import torch
import numpy as np
import os
import json
import cv2
import mediapipe as mp
from src.models.static_network import StaticSignClassifier
from src.preprocessing.hand_detector import create_detector, frame_to_HLResult
from src.preprocessing.tensor_manipulation import hlresult_to_tensor84
from src.utils.ema import probEMA

def main(refresh_rate = 2, ema_alpha = 0.2, confidence_threshold = 0.65, consistent_threshold = 3):
    model_load_path = "src/models/saved_models/static_sign.pth"
    data_load_path = "src/data/cache.pt"
    model_weights = torch.load(model_load_path, weights_only=True)
    data_info = torch.load(data_load_path)
    detector = create_detector()
    color = (0, 255, 0)

    CLASS_INDEX = { v : k for k, v in data_info["classes"].items()}
    model = StaticSignClassifier(num_categories=len(CLASS_INDEX))
    model.load_state_dict(model_weights)
    model.eval()

    ema = probEMA(len(CLASS_INDEX), alpha=ema_alpha)

    live_vid = cv2.VideoCapture(0)
    frame_index = 0
    consistent_amount = 0           # Setting default values
    actual_confidence = 0
    label = "None"
    if not live_vid.isOpened():
        raise RuntimeError("Cannot use camera")
    try:
        while True:
            ok, frame = live_vid.read()
            if not ok:
                print("Camera failed")
                break
            frame_index += 1
            if frame_index % refresh_rate == 0:
                hl_result = frame_to_HLResult(frame, detector)
                tensor = hlresult_to_tensor84(hl_result)
                tensor = tensor.unsqueeze(0) # To convert it from [84] to matrix [1, 84]
                with torch.no_grad():
                    logits = model(tensor)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                smooth_probabilities = ema.update(probabilities)
                predicted_index = int(np.argmax(smooth_probabilities)) # Returns the index of the highest probability
                predicted_confidence = float(smooth_probabilities[predicted_index]) # Returns the highest probability

                if predicted_confidence >= confidence_threshold:
                    consistent_amount += 1
                    if consistent_amount >= consistent_threshold:
                        label = CLASS_INDEX[predicted_index]
                        actual_confidence = predicted_confidence
                else:
                    consistent_amount = 0

            # Displays the video with the label and confidence level
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{label} ({actual_confidence:.2f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("Static Live (Python)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    except Exception as e:
        print(e)

    finally:
        live_vid.release()
        cv2.destroyAllWindows()
        print("Capture Stopped")

if __name__ == "__main__":
    main()
