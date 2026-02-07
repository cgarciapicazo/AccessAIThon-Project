import torch
import torch.nn as nn
from src.models.static_sign_classifier import StaticSignClassifier


def main(n_epochs = 20, n_features = 84, n_categories = 5, lr=1e-3):
    """Train and save a `StaticSignClassifier`.

    Args:
        n_epochs: Number of training epochs
        n_features: Input feature size
        n_categories: Number of output classes
        lr: Learning rate for Adam
    """
    model = StaticSignClassifier(n_features, n_categories)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    X = []
    y = []
    X_test = []
    y_test = []

    # Where the trained mode is saved.
    save_path = "src/models/saved_models/static_sign.pth"

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()
        # Forward pass
        predictions = model(X)
        CEL = loss(predictions, y)
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