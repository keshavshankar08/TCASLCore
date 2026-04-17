import cv2
from collections import deque
from tcasl import TCASL

def main():
    # Initialize class
    tcasl = TCASL()
    
    # Setup video capture
    cap = cv2.VideoCapture(1)

    # Buffer for predictions
    prediction_buffer = deque(maxlen=30)
    
    # Initial frame
    previous_frame = None

    while True:
        # Fetch frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip and convert frame to gray
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Preprocess frame
        processed_frame = tcasl.preprocess_frame(gray_frame)
        
        # Skip first iteration
        if previous_frame is None:
            previous_frame = processed_frame
            continue
            
        # Compute temporal contrast
        tc_frame = tcasl.compute_temporal_contrast(previous_frame, processed_frame, threshold=20)
        
        # Predict signed letter
        predictions = tcasl.predict(tc_frame, top_k=5)

        # Add prediction to buffer
        prediction_buffer.append(predictions[0])
        
        # Get majority vote prediction from buffer
        labels_only = [item[0] for item in prediction_buffer]
        majority_pred = max(set(labels_only), key=labels_only.count)
        
        # Get confidence associated with winning class
        majority_confs = [item[1] for item in prediction_buffer if item[0] == majority_pred]
        avg_confidence = sum(majority_confs) / len(majority_confs)
        
        # Show result
        display_text = f"{majority_pred.upper()} ({avg_confidence:.2f})"
        processed_frame_color = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        cv2.putText(processed_frame_color, display_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Grayscale Frame", processed_frame_color)
        cv2.imshow("Temporal Contrast Frame", tc_frame)
        
        # Update previous frame as current
        previous_frame = processed_frame
        
        # Quit if 'q' pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    # Release video capture
    cap.release()

    # Destroy windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()