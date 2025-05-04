import argparse
import os
import json
import re

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from collections import Counter
matplotlib.use('Agg')  # Prevents interactive display issues
from tqdm import tqdm
import sklearn
from sklearn.metrics import classification_report
import metrics
from model_pCNN import C_lenet
from model import CNN, MLP, CNN_CBAM
from getFeature import *
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {
    "front": 0,
    "left": 1,
    "none": 2,
    "right": 3,
    "front_left": 4,
    "front_right": 5
}

# Paths
dataset_path = r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad\dataset'
xml_path = r'C:/Projects/occluded_vehicle_acoustic_detection/config/ourmicarray_56.xml'
model_path = 'best_model_cnn.pth'
scaler_path = 'scaler.pkl'
log_file = 'train_log.txt'
log_dir = 'runs/train_log'


class GetPrecomputedData(Dataset):
    def __init__(self, dataframe, L, resolution, device='cpu'):
        """
        直接一次性把特征和标签 tensor 化，避免 __getitem__ 每次转换浪费时间。

        Args:
            dataframe (pd.DataFrame): 输入的 dataframe
            L (int): 时间段数量
            resolution (int): 每段特征的维度
            device (str): 'cpu' 或 'cuda'
        """
        self.L = L
        self.resolution = resolution
        self.device = device

        feature_cols = [col for col in dataframe.columns if col.startswith("feat")]
        features = dataframe[feature_cols].values.astype('float32')
        features = features.reshape(-1, 1, L, resolution)
        self.features = torch.tensor(features, dtype=torch.float32)

        # label
        label_raw = dataframe["Class"].values
        self.labels = torch.tensor(label_raw, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_process_data(loc, root, filename):
    """
    加载预处理特征，并按 origin_id 分组划分，避免信息泄露。
    支持文件名如：metadata_samples_L2_r240_f50-1500_w1000_pre1000_post100_mid250_c6.csv
    """

    # 先通过正则提取关键参数（只提取前面的 L r fmin fmax）
    pattern = r"_L(\d+)_r(\d+)_f(\d+)-(\d+)_w(\d+)_pre(\d+)_post(\d+)_mid(\d+)_mw(\d+)_c(\d+)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"文件名格式错误，无法提取L/r/fmin/fmax：{filename}")

    L = int(match.group(1))
    resolution = int(match.group(2))
    fmin = int(match.group(3))
    fmax = int(match.group(4))
    mid_window_length = int(match.group(9))
    num_class = int(match.group(10))

    dataset_csv = os.path.join(root, filename)
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"文件不存在：{dataset_csv}")

    df = pd.read_csv(dataset_csv)

    # --- 筛选 location ---
    env_map = {
        "SA": ["SA1", "SA2"],
        "SB": ["SB1", "SB2", "SB3"],
        "SAB": ["SA1", "SA2", "SB1", "SB2", "SB3"],
        "DA": ["DA1", "DA2"],
        "DB": ["DB1", "DB2", "DB3"],
        "DAB": ["DA1", "DA2", "DB1", "DB2", "DB3"]
    }
    valid_envs = env_map[loc]
    df = df[df["Environment"].isin(valid_envs)]

    # --- 信息泄露处理 ---
    if "filename" in df.columns:
        print("✅ Detected 'filename' column，进行 origin_id 分组，避免信息泄露。")

        def extract_origin_id(path):
            basename = os.path.basename(path)
            parts = basename.split("_")[:3]  # 取前三段，比如 Sxx_Dxx_ID
            return "_".join(parts)

        df["origin_id"] = df["filename"].apply(extract_origin_id)

        group_keys = df["origin_id"].unique()
        train_keys, testval_keys = train_test_split(group_keys, test_size=0.4, random_state=42)
        val_keys, test_keys = train_test_split(testval_keys, test_size=0.5, random_state=42)

        train_df = df[df["origin_id"].isin(train_keys)].drop(columns=["origin_id"])
        val_df = df[df["origin_id"].isin(val_keys)].drop(columns=["origin_id"])
        test_df = df[df["origin_id"].isin(test_keys)].drop(columns=["origin_id"])
    else:
        print("⚠️ 没有 'filename' 列，直接随机划分 train/val/test。")
        train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    print_dataset_distribution_df(train_df)

    # --- 数据增强 ---
    train_df = do_data_augmentation_left_right(train_df, resolution, L)
    if num_class > 4:
        train_df = do_data_augmentation_front_none(train_df, resolution, L)

    # --- 创建DataLoader用的数据集 ---
    train_set = GetPrecomputedData(train_df.reset_index(drop=True), L, resolution)
    val_set = GetPrecomputedData(val_df.reset_index(drop=True), L, resolution)
    test_set = GetPrecomputedData(test_df.reset_index(drop=True), L, resolution)

    return train_set, val_set, test_set, L, resolution, fmin, fmax, mid_window_length, num_class


def do_data_augmentation_left_right(df, res, L):
    """
    一键完成数据增强：
    - front (0)：镜像翻转
    - none (2)：根据front增强量，下采样
    - left (1) ↔ right (3)：互相翻转增强
    - front_left / front_right：保留不变

    参数：
        df (pd.DataFrame): 原始数据，含 feat 列和 Class 列
        res (int): 每段分辨率
        L (int): 时间段数

    返回：
        df_augmented (pd.DataFrame): 增强后的新数据
    """
    df_orig = df.copy()
    feature_cols = [col for col in df.columns if col.startswith("feat")]

    # ========== LEFT / RIGHT 类增强 ==========
    left_df = df_orig[df_orig["Class"] == 1]
    right_df = df_orig[df_orig["Class"] == 3]

    augmented_lr = []

    # right ➝ left
    for _, row in right_df.iterrows():
        feature = row[feature_cols].values.reshape(L, res)
        flipped = np.flip(feature, axis=1).reshape(-1)

        new_row = row.copy()
        new_row[feature_cols] = flipped
        new_row["Class"] = 1
        augmented_lr.append(new_row)

    # left ➝ right
    for _, row in left_df.iterrows():
        feature = row[feature_cols].values.reshape(L, res)
        flipped = np.flip(feature, axis=1).reshape(-1)

        new_row = row.copy()
        new_row[feature_cols] = flipped
        new_row["Class"] = 3
        augmented_lr.append(new_row)

    df_lr_aug = pd.DataFrame(augmented_lr)

    # ========== 其余类保留 ==========
    keep_df = df_orig[df_orig["Class"].isin([4, 5])]  # front_left / front_right

    # ========== 合并全部 ==========
    df_augmented = pd.concat([
        df_orig[df_orig["Class"].isin([0, 1, 2, 3])],  # 原 front, left, right
        df_lr_aug,
        keep_df
    ], ignore_index=True)

    return df_augmented


def do_data_augmentation_front_none(df, res, L):
    """
    一键完成数据增强：
    - front (0)：镜像翻转
    - none (2)：根据front增强量，下采样
    - left (1) ↔ right (3)：互相翻转增强
    - front_left / front_right：保留不变

    参数：
        df (pd.DataFrame): 原始数据，含 feat 列和 Class 列
        res (int): 每段分辨率
        L (int): 时间段数

    返回：
        df_augmented (pd.DataFrame): 增强后的新数据
    """
    df_orig = df.copy()
    feature_cols = [col for col in df.columns if col.startswith("feat")]

    # ========== FRONT 类增强 ==========
    front_df = df_orig[df_orig["Class"] == 0]
    # augmented_front = []
    #
    # for _, row in front_df.iterrows():
    #     feature = row[feature_cols].values.reshape(L, res)
    #     flipped = np.flip(feature, axis=1).reshape(-1)
    #
    #     new_row = row.copy()
    #     new_row[feature_cols] = flipped
    #     augmented_front.append(new_row)
    #
    # df_front_aug = pd.DataFrame(augmented_front)

    # ========== NONE 类下采样 ==========
    none_df = df_orig[df_orig["Class"] == 2]
    target_none_count = len(front_df) # * 2

    if len(none_df) > target_none_count:
        df_none_down = none_df.sample(n=target_none_count, random_state=42)
    else:
        df_none_down = none_df.copy()

    # ========== 其余类保留 ==========
    keep_df = df_orig[df_orig["Class"].isin([4, 5])]  # front_left / front_right

    # ========== 合并全部 ==========
    df_augmented = pd.concat([
        df_orig[df_orig["Class"].isin([0, 1, 3])],  # 原 front, left, right
        # df_front_aug,
        df_none_down,
        keep_df
    ], ignore_index=True)

    return df_augmented


def train_test_DNN(train_set, val_set, test_set, L, resolution, num_class, loc, epoch, i, save_cls=True):
    # do flip based data augmentation
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model = CNN(num_class).to(device)
    # model = MLP(L, resolution, 4).to(device)
    # model = CNN(L, resolution, 4).to(device)
    # model = CNN_CBAM().to(device)

    train_epoch(model, train_loader, val_loader, epoch)

    if save_cls:
        model_path = "models/" + '_' + str(L) + '_' + str(resolution) + '_' + str(i) + ".pth"
        torch.save(model.state_dict(), model_path)

    # Load the model's state dictionary
    model.load_state_dict(torch.load('best_model_earlystop.pth', weights_only=True))
    print("Load early stop model!")

    accuracy, conf_mat = test_model(model, test_loader)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    all_metrics = {"overall_accuracy": (accuracy, 0),
                   "per_class_accuracy": (metrics.getPCaccuracy(conf_mat), np.zeros(6)),
                   "per_class_precision": (metrics.getPCPrecision(conf_mat), np.zeros(6)),
                   "per_class_recall": (metrics.getPCRecall(conf_mat), np.zeros(6)),
                   "per_class_iou": (metrics.getPCIoU(conf_mat), np.zeros(6))}

    metrics.print_metrics(all_metrics, True)

    save_confusion_matrix_data(conf_mat, f"confmat_L{L}_res{resolution}_run{i}_{loc}.csv")

    return all_metrics, conf_mat


def train_epoch(model, train_loader, val_loader, epoch, patience=20, save_path="best_model_earlystop.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = 5
    counter = 0

    for i in range(epoch):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {i + 1}', leave=False)
        for feature, labels in progress_bar:
            feature, labels = feature.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(feature)
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"🔄 Val Loss下降，保存模型（Val Loss = {val_loss:.4f}）")
        else:
            counter += 1
            print(f"⏳ 没有提升（{counter}/{patience}）")

        if counter >= patience:
            print("⛔ 早停触发！训练提前结束。")
            plot_training_history(train_losses, val_losses, val_accuracies, output_path="training_plot.png")
            # break
            return True

    plot_training_history(train_losses, val_losses, val_accuracies, output_path="training_plot.png")
    return False


def train_epoch_v2(model, train_loader, val_loader, epochs, patience=20, save_path="best_model_earlystop.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    scaler = torch.amp.GradScaler(device="cuda")

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)

        for feature, labels in progress_bar:
            feature, labels = feature.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(feature)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        val_loss, val_acc = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"\u2714\ufe0f 保存最佳模型，val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"\u23f3 没有提升，耐心值：{patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\u274c 早停触发，停止训练！")
            plot_training_history(train_losses, val_losses, val_accuracies, output_path="training_plot.png")
            return True

    plot_training_history(train_losses, val_losses, val_accuracies, output_path="training_plot.png")
    return False


def validate_model(model, val_loader, criterion):
    """计算 Validation Loss 和 Accuracy"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for feature, labels in val_loader:
            feature, labels = feature.to(device), labels.to(device)
            outputs = model(feature)
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
        for feature, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            feature, labels = feature.to(device), labels.to(device)
            outputs = model(feature)
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


def print_dataset_distribution(dataset, location_name, name):
    """
    打印数据集中每个类别的数量和总样本数。

    参数:
        dataset: GetPrecomputedData对象，具有 labels 属性。
        location_name: 字符串，比如 'SA', 'SB', 'SAB'。
    """
    from collections import Counter

    # 直接拿 labels tensor 并转成 list
    labels = dataset.labels.tolist()

    # 统计每个类别数量
    counter = Counter(labels)

    # 打印输出
    print(f"-- {location_name}({name}) --")
    print(f"  总样本数(after data augmentation): {len(dataset)}")
    for class_id in range(6):
        count = counter.get(class_id, 0)
        class_name = {
            0: "front",
            1: "left",
            2: "none",
            3: "right",
            4: "front_left",
            5: "front_right"
        }.get(class_id, "unknown")
        print(f"    ClassID {class_id} ({class_name}): {count}")


def print_dataset_distribution_df(df):
    counter = Counter(df["Class"].tolist())

    print(f"  总样本数(before data augmentation): {len(df)}")
    for class_id in range(6):
        count = counter.get(class_id, 0)
        class_name = {
            0: "front",
            1: "left",
            2: "none",
            3: "right",
            4: "front_left",
            5: "front_right"
        }.get(class_id, "unknown")
        print(f"    ClassID {class_id} ({class_name}): {count}")


def save_confusion_matrix_data(conf_mat, filename="confusion_matrix.csv", class_names=None):
    """
    保存混淆矩阵为 CSV 文件格式。
    参数：
        conf_mat (np.ndarray): 混淆矩阵 (shape: [num_classes, num_classes])
        filename (str): 保存路径
        class_names (list[str]): 类别名称列表（可选）
    """
    import pandas as pd
    if class_names is None:
        class_names = ["front", "left", "none", "right", "front_left", "front_right"]
        class_names = class_names[:conf_mat.shape[0]]

    df = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
    df.to_csv(filename)
    print(f"✅ 混淆矩阵已保存为：{filename}")


def CNN_train(locs_list, root):
    results = []
    filenames = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    for filename in filenames:
        if filename.startswith('metadata'):
            for loc in locs_list:
                epoch = 100
                train_set, val_set, test_set, L, resolution, fmin, fmax, mid_window_length, num_class = load_and_process_data(loc=loc, root=root, filename=filename)
                print_dataset_distribution(train_set, loc, "train set")
                print_dataset_distribution(val_set, loc, "val set")
                print_dataset_distribution(test_set, loc, "test set")
                print(f'L: {L}, resolution: {resolution}, fmin: {fmin}, fmax: {fmax}, loc: {loc}, filename: {filename}')
                times = 1
                for i in range(times):
                    print(f'start turn {i}: ----------------------------------------------------------------')
                    output_metrics, conf_mat = train_test_DNN(train_set, val_set, test_set, L, resolution, num_class, loc, epoch, i, True)
                    result_entry = {
                        "filename": filename,
                        "location": loc,
                        "L": L,
                        "resolution": resolution,
                        "fmin": fmin,
                        "fmax": fmax,
                        "mid_window_length": mid_window_length,
                        "num_class": num_class,
                        "overall_accuracy_mean": float(output_metrics["overall_accuracy"][0]),
                        "overall_accuracy_std": float(output_metrics["overall_accuracy"][1]),
                        "per_class_accuracy_mean": [float(x) for x in output_metrics["per_class_accuracy"][0]],
                        "per_class_accuracy_std": [float(x) for x in output_metrics["per_class_accuracy"][1]],
                        "per_class_precision_mean": [float(x) for x in output_metrics["per_class_precision"][0]],
                        "per_class_precision_std": [float(x) for x in output_metrics["per_class_precision"][1]],
                        "per_class_recall_mean": [float(x) for x in output_metrics["per_class_recall"][0]],
                        "per_class_recall_std": [float(x) for x in output_metrics["per_class_recall"][1]],
                        "per_class_iou_mean": [float(x) for x in output_metrics["per_class_iou"][0]],
                        "per_class_iou_std": [float(x) for x in output_metrics["per_class_iou"][1]]
                    }
                    results.append(result_entry)
    summary_df = pd.DataFrame(results)
    save_path = os.path.join(root, "summary_metrics.csv")
    summary_df.to_csv(save_path, index=False)
    print("Created summary_metrics.csv!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from wav files.')
    parser.add_argument('--locs_list', nargs="+", default=["SAB", "SA", "SB"], help='List of Location IDs to run')
    parser.add_argument('--root', default=None, help='Path to extracted samples')
    args = parser.parse_args()
    CNN_train(args.locs_list, root=args.root)
