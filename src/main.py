import torch
import numpy as np
import cv2

from src.models.static_network import StaticSignClassifier
from src.preprocessing.hand_detector import create_detector, frame_to_HLResult
from src.preprocessing.tensor_manipulation import hlresult_to_tensor84
from src.utils.ema import probEMA

from src.postprocessing.text_to_speech import TTSService


def draw_accessible_ui(frame_bgr, label, confidence, detecting_text="Detecting..."):
    
    frame_bgr = cv2.flip(frame_bgr, 1)

    h, w = frame_bgr.shape[:2]

    panel_h = int(h * 0.30)          
    margin_x = int(w * 0.04)         
    margin_y = int(panel_h * 0.22)   

    panel = np.full((panel_h, w, 3), 255, dtype=np.uint8)

    if label == "None" or label.strip() == "" or confidence <= 0:
        main_text = detecting_text
        conf_text = ""
    else:
        main_text = str(label)
        conf_text = f"{confidence * 100:.0f}% confidence"

    font = cv2.FONT_HERSHEY_SIMPLEX
    main_scale = 2.2   
    main_thickness = 4

    max_text_width = int(w * 0.92)
    while True:
        (tw, th), _ = cv2.getTextSize(main_text, font, main_scale, main_thickness)
        if tw <= max_text_width or main_scale <= 0.9:
            break
        main_scale -= 0.1

    cv2.putText(
        panel,
        main_text,
        (margin_x, margin_y + int(panel_h * 0.25)),
        font,
        main_scale,
        (0, 0, 0),          
        main_thickness,
        cv2.LINE_AA
    )

    if conf_text:
        cv2.putText(
            panel,
            conf_text,
            (margin_x, margin_y + int(panel_h * 0.70)),
            font,
            1.1,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    cv2.line(panel, (0, 0), (w, 0), (0, 0, 0), 2)

    combined = np.vstack([frame_bgr, panel])
    return combined


def main(refresh_rate=2, ema_alpha=0.2, confidence_threshold=0.65, consistent_threshold=3):

    model_load_path = "src/models/saved_models/static_sign.pth"
    data_load_path = "src/data/cache.pt"

    model_weights = torch.load(model_load_path, weights_only=True)
    data_info = torch.load(data_load_path)

    detector = create_detector()
    CLASS_INDEX = {v: k for k, v in data_info["classes"].items()}

    model = StaticSignClassifier(num_categories=len(CLASS_INDEX))
    model.load_state_dict(model_weights)
    model.eval()

    ema = probEMA(len(CLASS_INDEX), alpha=ema_alpha)

    live_vid = cv2.VideoCapture(0)

    frame_index = 0
    consistent_amount = 0

    label = "None"
    actual_confidence = 0.0
    last_spoken_label = None

    tts = TTSService()

    if not live_vid.isOpened():
        tts.close()
        raise RuntimeError("Cannot use camera")

    try:
        while True:
            ok, frame = live_vid.read()
            if not ok:
                print("Camera failed")
                break

            frame_index += 1

            display_label = "Detecting..."
            display_conf = 0.0

            if frame_index % refresh_rate == 0:
                hl_result = frame_to_HLResult(frame, detector)
                tensor = hlresult_to_tensor84(hl_result).unsqueeze(0)

                with torch.no_grad():
                    logits = model(tensor)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

                smooth_probabilities = ema.update(probabilities)
                predicted_index = int(np.argmax(smooth_probabilities))
                predicted_confidence = float(smooth_probabilities[predicted_index])

                if predicted_confidence >= confidence_threshold:
                    consistent_amount += 1
                    if consistent_amount >= consistent_threshold:
                        label = CLASS_INDEX[predicted_index]
                        actual_confidence = predicted_confidence
                else:
                    consistent_amount = 0
                    label = "None"
                    actual_confidence = 0.0

            if label != "None" and actual_confidence >= confidence_threshold:
                display_label = label
                display_conf = actual_confidence

                if display_label != last_spoken_label:
                    tts.speak(display_label)
                    last_spoken_label = display_label

            ui_frame = draw_accessible_ui(frame, display_label if display_label != "Detecting..." else "None", display_conf)

            cv2.imshow("AccessAIThon - Sign to Speech", ui_frame)

            if cv2.waitKey(1) & 0xFF == 27:  
                break

    except Exception as e:
        print(e)

    finally:
        try:
            tts.close()
        except Exception:
            pass
        live_vid.release()
        cv2.destroyAllWindows()
        print("Capture Stopped")


if __name__ == "__main__":
    main()