import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from src.models.static_network import StaticSignClassifier
from src.preprocessing.image_hand_detector import img_to_HLResult, create_detector
from src.preprocessing.tensor_manipulation import hlresult_to_tensor84

# We used our own images of hand signals to train the static model

def main(n_steps = 200000, n_features = 84, lr=1e-3, test_size=0.2, batch_size=256, step_check = 500):
    """Train and save a `StaticSignClassifier`.

    Args:
        n_epochs: Number of training epochs
        n_features: Input feature size
        n_categories: Number of output classes
        lr: Learning rate for Adam
    """
    data_path = "src/data/images"
    cache_path = "src/data/cache.pt"

    # Loads the data set
    X = []
    y = []
    detector = create_detector()
    if not os.path.isfile(cache_path) or os.path.getsize(cache_path) == 0:

        # Sets up the labels and their corresponding index
        classes = os.listdir(data_path)
        try:
            classes.remove('.DS_Store')
        except Exception:
            pass
        classes.sort()
        n_categories = len(classes)
        CLASSES_TO_INDEX = {label : i for i, label in enumerate(classes)}

        # Loads all the images and convert them into X and y tensors with the appropiate values
        dir_count = 0
        for dir in os.listdir(data_path):
            if dir != '.DS_Store': # May read some unwanted files that are not a class
                class_index = CLASSES_TO_INDEX[dir]
                dir_count += 1
                for img in os.listdir(data_path + "/" + dir):
                    hl_result = img_to_HLResult(data_path + "/" + dir + "/" + img, detector)
                    tensor = hlresult_to_tensor84(hl_result)
                    X.append(tensor)
                    y.append(class_index)
                print(f" |||||||| The dir {dir} ({dir_count}/{n_categories}) has finished loading ||||||||")
        X = torch.stack(X, dim=0)
        y = torch.tensor(y, dtype=torch.long)

        torch.save({"X" : X, "y" : y, "classes" : CLASSES_TO_INDEX}, cache_path)
    else:
        d = torch.load(cache_path)
        X = d["X"]
        y = d["y"]
        CLASSES_TO_INDEX = d["classes"]
        n_categories = len(CLASSES_TO_INDEX)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                            shuffle=True, stratify=y)

    # Where the trained mode is saved
    save_path = "src/models/saved_models/static_sign.pth"

    model = StaticSignClassifier(n_categories, num_features=n_features)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    train_size = X_train.size(0)
    test_size = X_test.size(0)

    # trains the model
    for step in range(1, n_steps + 1):
        model.train()
        optimiser.zero_grad()
        # Forward pass
        idx = torch.randint(train_size, (batch_size,))
        predictions = model(X_train[idx])
        CEL = loss(predictions, y_train[idx])
        CEL.backward()
        optimiser.step()

        if step % step_check == 0:
            model.eval()
            with torch.no_grad():
                # Loss on test data
                idxt = torch.randint(test_size, (int(batch_size / 16),))
                test_predictions = model(X_test[idxt])
                test_CEL = loss(test_predictions, y_test[idxt])

                logits = model(X_test)
                pred = logits.argmax(dim=1)
                correct = (pred == y_test)
                accuracy = correct.float().mean()
            print(f"Epoch {step}: train_loss = {CEL.item():.4f}, test_loss = {test_CEL.item():.4f}, accuracy = {accuracy:.5f}")

    # Saves the model weights
    torch.save(model.state_dict(), save_path)
    print(f"Model successfuly saved at: {save_path}")

if __name__ == "__main__":
    main()