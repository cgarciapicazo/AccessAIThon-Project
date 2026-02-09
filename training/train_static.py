import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.models.static_network import StaticSignClassifier
# Import proper file when proper dataset
from src.data.examples.test_datasets import generate_static


def main(n_epochs = 20, n_features = 84, n_categories = 6, lr=1e-3, test_size=0.2):
    """Train and save a `StaticSignClassifier`.

    Args:
        n_epochs: Number of training epochs
        n_features: Input feature size
        n_categories: Number of output classes
        lr: Learning rate for Adam
    """
    model = StaticSignClassifier(n_categories, num_features=n_features)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    # Change depending on the dataset loaded
    X, y = generate_static(n_categories)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                            shuffle=True, stratify=y)
    
    CLASSES = ["C", "A", "Hello", "How are you?", "Good", "Bad"]
    CLASSES_TO_INDEX = {label : i for i, label in enumerate(CLASSES)}

    # Where the trained mode is saved
    save_path = "src/models/saved_models/static_sign.pth"

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()
        # Forward pass
        predictions = model(X_train)
        CEL = loss(predictions, y_train)
        CEL.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            # Loss on test data
            test_predictions = model(X_test)
            test_CEL = loss(test_predictions, y_test)

        print(f"Epoch {epoch}: train_loss = {CEL.item():.4f}, test_loss = {test_CEL.item():.4f}")

    # Saves the model weights
    torch.save(model.state_dict(), save_path)
    print(f"Model successfuly saved at: {save_path}")

if __name__ == "__main__":
    main()