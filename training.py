import os
import json

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

matplotlib.use('Agg')  # Prevents interactive display issues
from tqdm import tqdm
import sklearn
from sklearn.metrics import classification_report
import metrics
from model import Classifier
from model_pCNN import C_lenet
from getFeature import *
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
dataset_path = r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad\dataset'
xml_path = r'C:/Projects/occluded_vehicle_acoustic_detection/config/ourmicarray_56.xml'
model_path = 'best_model_cnn.pth'
scaler_path = 'scaler.pkl'
log_file = 'train_log.txt'
log_dir = 'runs/train_log'


def get_label_from_json(json_path):
    """直接从 JSON 文件中获取 class 字段"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data.get("class", 0)  # 默认为 0（none）类


class GetPrecomputedData(Dataset):
    def __init__(self, dataframe):
        self.data_df = dataframe

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        feature = torch.load(row["feature_path"], weights_only = True)  # [1, L, resolution] 已包含通道维度
        label = int(row["label"])
        return feature, label


def load_and_process_data():
    """加载预处理特征，并按 origin_id 分组划分，避免信息泄露"""

    dataset_csv = 'processed_features/metadata.csv'
    df = pd.read_csv(dataset_csv)

    # 提取 origin_id
    def extract_origin_id(path):
        filename = os.path.basename(path)
        parts = filename.split("_")[:3]
        return "_".join(parts)

    df["origin_id"] = df["feature_path"].apply(extract_origin_id)

    # 分组并划分（保持组不被拆散）
    group_keys = df["origin_id"].unique()
    train_keys, testval_keys = train_test_split(group_keys, test_size=0.4, random_state=42)
    val_keys, test_keys = train_test_split(testval_keys, test_size=0.5, random_state=42)

    train_df = df[df["origin_id"].isin(train_keys)].drop(columns=["origin_id"])
    val_df = df[df["origin_id"].isin(val_keys)].drop(columns=["origin_id"])
    test_df = df[df["origin_id"].isin(test_keys)].drop(columns=["origin_id"])

    # 创建数据集
    train_set = GetPrecomputedData(train_df.reset_index(drop=True))
    val_set = GetPrecomputedData(val_df.reset_index(drop=True))
    test_set = GetPrecomputedData(test_df.reset_index(drop=True))

    return train_set, val_set, test_set


def train_test_DNN(train_set, val_set, test_set, L, resolution, epoch, locs_in=['SAB'], save_cls=True):
    # do flip based data augmentation
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model = Classifier(L, resolution, in_channels=1, out_channels=16, num_class=4).to(device)
    # model = C_lenet(nn_number=3, class_number=4, lstm_layer=1).to(device)

    train_epoch(model, train_loader, val_loader, epoch)

    if save_cls:
        model_path = "models/" + "_".join(locs_in) + '_' + str(L) + '_' + str(resolution) + ".pth"
        torch.save(model.state_dict(), model_path)

    # Load the model's state dictionary
    # model.load_state_dict(torch.load('models/SAB_model.pth'))

    accuracy, conf_mat = test_model(model, test_loader)

    all_metrics = {"overall_accuracy": (accuracy, 0),
                   "per_class_accuracy": (metrics.getPCaccuracy(conf_mat), np.zeros(4)),
                   "per_class_precision": (metrics.getPCPrecision(conf_mat), np.zeros(4)),
                   "per_class_recall": (metrics.getPCRecall(conf_mat), np.zeros(4)),
                   "per_class_iou": (metrics.getPCIoU(conf_mat), np.zeros(4))}

    metrics.print_metrics(all_metrics, True)


def train_epoch(model, train_loader, val_loader, epoch, patience=5, save_path="best_model_earlystop.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_acc = 0.0
    counter = 0

    for i in range(epoch):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {i + 1}', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(loss=running_loss / len(train_loader))

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {i + 1}, Loss: {epoch_loss}")
        train_losses.append(epoch_loss)

        val_loss, val_acc = validate_model(model, val_loader, criterion)
        print(f"Epoch {i + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # ✅ Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"🔄 验证准确率提升，保存模型（Val Acc = {val_acc:.4f}）")
        else:
            counter += 1
            print(f"⏳ 没有提升（{counter}/{patience}）")

        if counter >= patience:
            print("⛔ 早停触发！训练提前结束。")
            break

    plot_training_history(train_losses, val_losses, val_accuracies, output_path="training_plot.png")


def validate_model(model, val_loader, criterion):
    """计算 Validation Loss 和 Accuracy"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc


def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    accuracy_score = sklearn.metrics.accuracy_score(all_preds, all_targets)

    # extract confusion matrix and metrics
    conf_mat = sklearn.metrics.confusion_matrix(all_preds, all_targets)

    return accuracy_score, conf_mat


def plot_training_history(train_losses, val_losses, val_accuracies, output_path="training_plot.png"):
    """
    绘制训练过程中的 Loss 和 Accuracy 曲线，并保存为图片
    """
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(12, 6))

    # -----------------------------
    # 左图：训练/验证 Loss 曲线
    # -----------------------------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o', linestyle='-')
    plt.plot(epochs, val_losses, label="Val Loss", marker='s', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)

    # -----------------------------
    # 右图：验证 Accuracy 曲线
    # -----------------------------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label="Val Accuracy", color='green', marker='^', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ 训练曲线已保存为：{output_path}")


if __name__ == '__main__':
    epoch = 20
    train_set, val_set, test_set = load_and_process_data()

    feature, label = train_set[0]
    _, L, resolution = feature.shape
    print(f"L = {L}, resolution = {resolution}")

    train_test_DNN(train_set, val_set, test_set, L, resolution, epoch, ['SAB'], True)

    # import torch
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")

    # model = Classifier(L, resolution, in_channels=1, out_channels=16, num_class=4).to(device)
    # model.load_state_dict(torch.load("models/SAB_2_30.pth"))
    # model.eval()
    # test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    # accuracy, conf_mat = test_model(model, test_loader)
    # all_metrics = {"overall_accuracy": (accuracy, 0),
    #                "per_class_accuracy": (metrics.getPCaccuracy(conf_mat), np.zeros(4)),
    #                "per_class_precision": (metrics.getPCPrecision(conf_mat), np.zeros(4)),
    #                "per_class_recall": (metrics.getPCRecall(conf_mat), np.zeros(4)),
    #                "per_class_iou": (metrics.getPCIoU(conf_mat), np.zeros(4))}
    #
    # metrics.print_metrics(all_metrics, True)