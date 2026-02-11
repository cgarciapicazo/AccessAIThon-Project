import torch

def hlresult_to_tensor84(res, wrist_relative=True):
    features = torch.zeros(84, dtype=torch.float32)

    if res is None or not res.hand_landmarks:
        return features

    def write_hand(start_index, landmarks):
        coords = []

        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y

        for lm in landmarks:
            x = lm.x
            y = lm.y

            if wrist_relative:
                x -= wrist_x
                y -= wrist_y

            coords.append(x)
            coords.append(y)

        hand_tensor = torch.tensor(coords, dtype=torch.float32)
        features[start_index:start_index + 42] = hand_tensor

    for i, landmarks in enumerate(res.hand_landmarks):

        try:
            label = res.handedness[i][0].category_name.lower()
        except:
            label = ""

        if label == "left":
            write_hand(0, landmarks)
        elif label == "right":
            write_hand(42, landmarks)

    return features