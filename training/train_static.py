import torch
import torch.nn as nn
from src.models.static_sign_classifier import StaticSignClassifier

def main(n_epochs = 20, n_features = 84, n_categories = 5, lr=1e-3):
    model = StaticSignClassifier(n_features, n_categories)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    X = []
    y = []
    X_test = []
    y_test = []

    save_path = "src/models/saved_models/static_sign.pth"

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimiser.zero_grad()
        predictions = model(X)
        CEL = loss(predictions, y)
        CEL.backward()
        optimiser.step()

        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_CEL = loss(test_predictions, y_test)

        print(f"Epoch {epoch}: train_loss = {CEL.item():.4f}, test_loss = {test_CEL.item():.4f}")

    torch.save(model, save_path)
    print(f"Model successfuly saved at: {save_path}")

if __name__ == "__main__":
    main()