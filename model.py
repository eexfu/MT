# This code was completed with assistance from ChatGPT.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self, output_dim=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNClassifier:
    def __init__(self, input_shape, num_classes, lr=1e-3, device=None, patience=30, optimizer_type='adam'):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN(output_dim=num_classes).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.best_model_state_dict = None
        self.patience = patience
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer type")
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

    def train_model(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        """
        X_train, X_val: input features
        y_train, y_val: Label arrays
        """
        L, res = self.input_shape

        # --- Data Preparation ---
        def prepare(X, y):
            X = np.asarray(X, dtype=np.float32)
            N = X.shape[0]
            X = X.reshape(N, 1, L, res)
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            y = torch.tensor(y, dtype=torch.long).to(self.device)
            return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        train_loader = prepare(X_train, y_train)
        val_loader = prepare(X_val, y_val)

        # --- Training Main Loop ---
        best_val_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Validation set loss
            val_loss = self._evaluate_loss(val_loader)
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state_dict = self.model.state_dict()  # Save best model weights
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def _evaluate_loss(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                total_loss += loss.item()
        return total_loss / len(loader)

    def predict(self, X):
        L, res = self.input_shape
        N = X.shape[0]
        X = np.asarray(X, dtype=np.float32)
        X = X.reshape(N, 1, L, res)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        # Use best model
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
        else:
            print("Warning: Best model not set. Using current model state.")

        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total trainable parameters:", total_params)
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()
