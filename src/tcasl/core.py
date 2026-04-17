import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnc
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Dict, Type
import lava.lib.dl.slayer as slayer

class BaseSDNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(BaseSDNN, self).__init__()
        
        sdnn_params = {
            'threshold'     : 0.1,
            'tau_grad'      : 0.5,
            'scale_grad'    : 1,
            'requires_grad' : True,
            'shared_param'  : True,
            'activation'    : fnc.relu,
        }
        sdnn_cnn_params = {
                **sdnn_params,
                'norm' : slayer.neuron.norm.MeanOnlyBatchNorm,
        }
        sdnn_dense_params = {
                **sdnn_cnn_params,
                'dropout' : slayer.neuron.Dropout(p=0.2),
        }
        
        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params), 
            slayer.block.sigma_delta.Conv(sdnn_cnn_params,  1, 24, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0, stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0), stride=(2, 1), weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0, stride=1, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Flatten(),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 24192, 100, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params,   100,  50, weight_scale=2, weight_norm=True),
            
            slayer.block.sigma_delta.Dense(sdnn_dense_params,    50, num_classes, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Output(sdnn_dense_params, num_classes, num_classes, weight_scale=2, weight_norm=True)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks: 
            x = block(x)
        return x

MODEL_REGISTRY: Dict[str, Dict] = {
    "sdnn_v1": {
        "class": BaseSDNN,
        "url": "https://github.com/keshavshankar08/TCASLCore/releases/download/v1.0.0/sdnn_v1.pth" 
    }
}

class TCASL:
    def __init__(self, arch: str = "sdnn_v1", model_path: str = None, custom_registry: dict = None):
        """
        Initializes the TCASL classifier.

        :param arch: The architecture tag to use.
        :param model_path: Optional local path to the weights. If None, it auto-downloads.
        :param custom_registry: Optional dictionary to inject custom architectures dynamically.

        """
        self.classes = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_arch = arch
        registry = MODEL_REGISTRY.copy()
        if custom_registry:
            registry.update(custom_registry)

        if arch not in registry:
            raise ValueError(f"Architecture '{arch}' not found. Available: {list(registry.keys())}")
        
        arch_info = registry[arch]
        ModelClass = arch_info["class"]
        download_url = arch_info.get("url", "")

        self.model = ModelClass(num_classes=len(self.classes)).to(self.device)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((128, 128)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self._load_model(model_path, download_url)

    def _load_model(self, path: str, download_url: str) -> None:
        if path and os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        elif download_url:
            state_dict = torch.hub.load_state_dict_from_url(
                download_url, 
                map_location=self.device, 
                weights_only=True,
                check_hash=False
            )
        else:
            raise ValueError("No local path provided and no download URL found for this architecture.")

        if "sdnn" in self.current_arch.lower():
            model_state = self.model.state_dict()
            has_updates = False

            for key, val in state_dict.items():
                if key in model_state and val.shape != model_state[key].shape:
                    has_updates = True

            if has_updates:
                for name, module in self.model.named_modules():
                    if hasattr(module, 'running_mean') and f"{name}.running_mean" in state_dict:
                        target_shape = state_dict[f"{name}.running_mean"].shape
                        module.register_buffer('running_mean', torch.zeros(target_shape).to(self.device))
                    
                    if hasattr(module, 'running_var') and f"{name}.running_var" in state_dict:
                        target_shape = state_dict[f"{name}.running_var"].shape
                        module.register_buffer('running_var', torch.zeros(target_shape).to(self.device))

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Center-crops a standard video frame to a square and resizes it to 128x128.

        :param frame: A raw grayscale image array of any shape.

        :return: A 128x128 cropped and resized grayscale image array.
        """
        h, w = frame.shape
        min_dim = min(h, w)
        start_x = (w // 2) - (min_dim // 2)
        start_y = (h // 2) - (min_dim // 2)
        square_img = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
        return cv.resize(square_img, (128, 128))
    
    def compute_temporal_contrast(self, previous_frame: np.ndarray, current_frame: np.ndarray, threshold: int = 20) -> np.ndarray:
        """
        Computes the temporal contrast between two consecutive frames to emulate a DVS.

        :param previous_frame: The previous grayscale frame.
        :param current_frame: The current grayscale frame (must match previous_frame shape).
        :param threshold: The pixel intensity difference required to trigger an event. Defaults to 20.

        :return: A temporal contrast frame where 127 is neutral, 255 is positive polarity, and 0 is negative polarity.
        """
        temp_contrast_frame = np.full(previous_frame.shape, 127, dtype=np.uint8)
        diff_frame = current_frame.astype(np.float32) - previous_frame.astype(np.float32)
        temp_contrast_frame[diff_frame > threshold] = 255
        temp_contrast_frame[diff_frame < -threshold] = 0
        return temp_contrast_frame

    def predict(self, tc_frame: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Runs neural network inference on a temporal contrast frame.

        :param tc_frame: The temporal contrast frame generated by compute_temporal_contrast.
        :param top_k: The number of top predictions to return. Defaults to 5.

        :return: A list of tuples containing (class_label, confidence_percentage) sorted by highest confidence.
        """
        tc_image = Image.fromarray(tc_frame)
        input_tensor = self.transform(tc_image).unsqueeze(0).to(self.device)

        if "sdnn" in self.current_arch.lower():
            input_tensor = input_tensor.unsqueeze(-1)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            if isinstance(outputs, tuple): 
                logits = outputs[0].flatten(start_dim=1) 
            else:
                logits = outputs.flatten(start_dim=1) if len(outputs.shape) > 2 else outputs

            probs = torch.softmax(logits, dim=1)[0]
            
            k = min(top_k, len(self.classes))
            top_probs, top_indices = torch.topk(probs, k)

            predictions = [
                (self.classes[idx.item()], round(prob.item() * 100, 2)) 
                for prob, idx in zip(top_probs, top_indices)
            ]

        return predictions