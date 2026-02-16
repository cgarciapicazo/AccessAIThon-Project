import mediapipe as mp
from mediapipe.tasks import python
import cv2

def frame_to_HLResult(frame, detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(image)
    return detection_result

def img_to_HLResult(path, detector):
    try:
        image = mp.Image.create_from_file(path)
        detection_result = detector.detect(image)
    except Exception:
        print("Wrong image/path")
        return None
    return detection_result

def create_detector():
    base_options = python.BaseOptions(model_asset_path='src/models/saved_models/hand_landmarker.task')
    options = python.vision.HandLandmarkerOptions(base_options=base_options, num_hands = 2)
    detector = python.vision.HandLandmarker.create_from_options(options)
    return detector
