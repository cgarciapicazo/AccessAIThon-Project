import mediapipe as mp
from mediapipe.tasks import python

def img_to_HLResult(path):
    base_options = python.BaseOptions(model_asset_path='src/models/saved_models/hand_landmarker.task')
    options = python.vision.HandLandmarkerOptions(base_options=base_options, num_hands = 2)
    detector = python.vision.HandLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(path)
    detection_result = detector.detect(image)
    return detection_result