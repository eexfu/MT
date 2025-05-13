import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# CBAM Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # [B, C, 1, 1]

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# Main CNN + CBAM Model
class CNN_CBAM(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(CNN_CBAM, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            CBAMBlock(32),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAMBlock(64),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CBAMBlock(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.layer1(x)  # [B, 32, H/2, W/2]
        out = self.layer2(out)  # [B, 64, H/4, W/4]
        out = self.layer3(out)  # [B, 128, 1, 1]
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out


class CNN_CBAM_L_resolution(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_CBAM, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            CBAMBlock(32),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CBAMBlock(64),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CBAMBlock(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 输入 x 的 shape 是 [B, L, resolution]，先变成 [B, 1, L, resolution]
        if x.ndim == 3:
            x = x.unsqueeze(1)

        out = self.layer1(x)         # -> [B, 32, L/2, resolution/2]
        out = self.layer2(out)       # -> [B, 64, L/4, resolution/4]
        out = self.layer3(out)       # -> [B, 128, 1, 1]
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class HeavyCNN(nn.Module):
    def __init__(self, output_dim: int = 4):
        """
        Args:
            output_dim (int): 分类类别数
        """
        super().__init__()
        # 输入为 [B, 1, L, resolution]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))  # 只对频率 resolution 降采样

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))  # 输出固定为 (B, 64, 4, 4)

        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, 1, L, resolution]
        Returns:
            logits: [B, output_dim]
        """
        x = self.conv1(x)         # [B, 1, L, R] → [B, 32, L, R]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)         # [B, 32, L, R//2]

        x = self.conv2(x)         # [B, 64, L, R//2]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)         # [B, 64, 4, 4]

        x = torch.flatten(x, 1)   # [B, 64*4*4]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
            nn.AdaptiveAvgPool2d((2, 2))  # 更小特征图
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


class SmallCNN(nn.Module):
    def __init__(self, output_dim=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),     # Conv1
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 下采样

            nn.Conv2d(8, 16, kernel_size=3, padding=1),    # Conv2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))                   # 输出变成 16×2×2 = 64
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 2 * 2, 32),                     # 64 -> 32
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, output_dim)
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
        self.input_shape = input_shape  # [L, resolution]
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
        X_train, X_val: [N, L, R]
        y_train, y_val: 标签数组
        """
        L, res = self.input_shape

        # --- 数据准备 ---
        def prepare(X, y):
            X = np.asarray(X, dtype=np.float32)
            N = X.shape[0]
            X = X.reshape(N, 1, L, res)
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            y = torch.tensor(y, dtype=torch.long).to(self.device)
            return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        train_loader = prepare(X_train, y_train)
        val_loader = prepare(X_val, y_val)

        # --- 训练主循环 ---
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

            # 验证集 loss
            val_loss = self._evaluate_loss(val_loader)
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping 判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state_dict = self.model.state_dict()  # 保存最优模型权重
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

        # 使用 best model
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


class MLP(nn.Module):
    def __init__(self, input_length: int, input_resolution: int, output_dim: int):
        """
        Args:
            input_length (int): 时间/序列维度长度 L
            input_resolution (int): 空间/特征维度大小 resolution
            output_dim (int): 输出维度
        """
        super().__init__()
        # 全连接层设计
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1 * input_length * input_resolution, 512)  # 包含通道维度1
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input Shape: (batch_size, 1, L, resolution)
        Output Shape: (batch_size, output_dim)
        """
        x = self.flatten(x)  # [B,1,L,resolution] → [B,1*L*resolution]

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x