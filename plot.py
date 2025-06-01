import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os

from sklearn.metrics import confusion_matrix


def loadCSVData(path = "./preprocess_test/result/summary_metrics_CNN_samples.csv"):
    df = pd.read_csv(path)

    def fix_and_eval(arr_str):
        try:
            fixed = '[' + ','.join(arr_str.strip('[]').split()) + ']'
            return ast.literal_eval(fixed)
        except:
            return []

    array_columns = [
        'per_class_accuracy_mean', 'per_class_accuracy_std',
        'per_class_precision_mean', 'per_class_precision_std',
        'per_class_recall_mean', 'per_class_recall_std',
        'per_class_iou_mean', 'per_class_iou_std'
    ]
    for col in array_columns:
        df[col] = df[col].apply(fix_and_eval)

    df['freq_range'] = df['fmin'].astype(str) + '-' + df['fmax'].astype(str)

    os.makedirs("figures", exist_ok=True)

    return df  # ✅ 不再返回 grouped


def line_accuracyVsL_by_location(df, location_name=None, filename_suffix="all"):
    if location_name:
        subset = df[df['location'] == location_name]
        title_loc = location_name
    else:
        subset = df
        title_loc = "All Locations"

    # 分组统计
    grouped = subset.groupby(['L', 'resolution', 'fmin', 'fmax'])['overall_accuracy_mean'].mean().reset_index()
    grouped['freq_range'] = grouped['fmin'].astype(str) + '-' + grouped['fmax'].astype(str)
    grouped['L'] = pd.Categorical(grouped['L'], categories=[1, 2, 4, 8, 16], ordered=True)

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=grouped,
        x='L',
        y='overall_accuracy_mean',
        hue='resolution',
        style='freq_range',
        markers=True
    )

    # 找出最佳点
    best_row = grouped.loc[grouped['overall_accuracy_mean'].idxmax()]
    best_L = best_row['L']
    best_res = best_row['resolution']
    best_freq = best_row['freq_range']
    best_acc = best_row['overall_accuracy_mean']

    # 标题与参数说明
    plt.title(f"Overall Accuracy vs L\n(Mean accuracy on {title_loc})")
    plt.ylabel("Accuracy")
    plt.xlabel("L (Segment Count)")
    plt.ylim(0, 1)
    plt.legend(title="Resolution / Freq Range", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.text(
        1.05, 0.25,
        f"Best Result:\nL = {best_L}\nres = {best_res}\nfreq = {best_freq}\nacc = {best_acc:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
    )

    plt.tight_layout()
    plt.savefig(f"figures/lineplot_accuracy_vs_L_{filename_suffix}.png")
    plt.close()


def bar_accuracyVsFreqRange(grouped, modelname, L=2, resolution=30, num_mics = 56, geo = 0, window_length = 1000):
    grouped = grouped[(grouped['location'] == 'SAB') & (grouped['num_mics'] == num_mics) & (grouped['geo'] == geo) & (grouped['L'] == L) & (grouped['resolution'] == resolution) & (grouped['window_length'] == window_length)]
    # 📊 柱状图：不同 freq_range 的平均准确率
    bar_data = grouped.groupby('freq_range')['overall_accuracy_mean'].mean().reset_index()

    # 设置顺序
    freq_order = ["20-50", "50-1500", "1500-3000", "3000-24000"]
    bar_data['freq_range'] = pd.Categorical(bar_data['freq_range'], categories=freq_order, ordered=True)
    bar_data = bar_data.sort_values('freq_range')

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=bar_data, x='freq_range', y='overall_accuracy_mean')

    plt.ylim(0, 1)

    # 添加柱顶数值标签
    for i, row in enumerate(bar_data.itertuples()):
        ax.text(
            i,
            row.overall_accuracy_mean + 0.01,
            f"{row.overall_accuracy_mean:.3f}",
            ha='center', va='bottom', fontsize=10, color='black'
        )

    plt.title(f'Mean Accuracy per Frequency Range({modelname})')
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Frequency Range (Hz)')
    plt.tight_layout()
    plt.savefig(f"figures/barplot_accuracy_freq_range_{modelname}.png")
    plt.close()


def heatMap_freq50_1500_by_location(df, location_name=None, filename_suffix="all", modelname="SVM", fmin=50, fmax=1500, num_mics=56, geo=0, window_length=1000):
    # 筛选 freq_range 为 50-1500 的数据
    df = df[(df['fmin'] == fmin) & (df['fmax'] == fmax) & (df['num_mics'] == num_mics) & (df['geo'] == geo) & (df['window_length'] == window_length) & (df['L'] != 1)]
    # 可选地筛选 location
    if location_name:
        df = df[df['location'] == location_name]
        title_suffix = f"Location: {location_name}({modelname})"
    else:
        title_suffix = "All Locations"

    # 分组并 pivot 成热力图数据
    grouped = df.groupby(['L', 'resolution'])['overall_accuracy_mean'].mean().reset_index()
    pivot_data = grouped.pivot(index='resolution', columns='L', values='overall_accuracy_mean')

    # 画图
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        vmin=0.6, vmax=1.0  # 固定颜色刻度范围
    )
    plt.title(f"Accuracy Heatmap (Freq {fmin}–{fmax})\n{title_suffix}")
    plt.ylabel("Resolution")
    plt.xlabel("L")
    plt.tight_layout()
    plt.savefig(f"figures/heatmap_accuracy_{fmin}_{fmax}_{filename_suffix}.png")
    plt.close()


def plot_window_comparison_by_location(df, model, L=2, resolution=30, fmin = 50, fmax = 1500, num_mics = 56, geo = 0):
    # 筛选所需列
    plot_df = df[(df['fmin'] == fmin) & (df['fmax'] == fmax) & (df['num_mics'] == num_mics) & (df['geo'] == geo) & (df['L'] == L) & (df['resolution'] == resolution)].copy()

    # 保证窗口长度是数值类型并排序
    plot_df['window_length'] = pd.to_numeric(plot_df['window_length'], errors='coerce')

    for loc in ['SAB', 'DAB']:
        subset = plot_df[plot_df['location'] == loc]

        grouped = subset.groupby('window_length')['overall_accuracy_mean'].mean().reset_index()

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=grouped, x='window_length', y='overall_accuracy_mean', marker='o')
        plt.title(f'Model Accuracy vs Window Length ({loc})\n(Freq range 50–1500Hz)')
        plt.xlabel('Window Length (ms)')
        plt.ylabel('Overall Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'figures/accuracy_vs_window_{loc}_{model}.png')
        plt.close()


def plot_window_comparison_dual(df_svm, df_cnn, L=2, resolution=30, fmin=50, fmax=1500, num_mics=56, geo=0):
    # 预处理两个模型的DataFrame
    def preprocess(df, model_name):
        df_filtered = df[
            (df['fmin'] == fmin) & (df['fmax'] == fmax) &
            (df['num_mics'] == num_mics) & (df['geo'] == geo) &
            (df['L'] == L) & (df['resolution'] == resolution)
            ].copy()
        df_filtered['model'] = model_name
        df_filtered['window_length'] = pd.to_numeric(df_filtered['window_length'], errors='coerce')
        return df_filtered[['location', 'window_length', 'overall_accuracy_mean', 'model']]

    df_svm_prep = preprocess(df_svm, 'SVM')
    df_cnn_prep = preprocess(df_cnn, 'CNN')

    # 合并两个DataFrame
    combined_df = pd.concat([df_svm_prep, df_cnn_prep], ignore_index=True)

    for loc in ['SAB', 'DAB']:
        subset = combined_df[combined_df['location'] == loc]

        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=subset,
            x='window_length',
            y='overall_accuracy_mean',
            hue='model',
            marker='o'
        )
        plt.title(f'Accuracy vs Window Length at {loc}\n(Freq range 50–1500Hz)')
        plt.xlabel('Window Length (ms)')
        plt.ylabel('Overall Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'figures/accuracy_vs_window_{loc}_compare.png')
        plt.close()


def plot_per_class_accuracy_comparison(df):
    # 筛选指定条件
    df_sub = df[
        (df['location'] == 'SAB') &
        (df['resolution'] == 240) &
        (df['fmin'] == 50) & (df['fmax'] == 1500) &
        (df['L'].isin([1, 2, 4, 8, 16]))
        ].copy()

    # 按照 class 顺序
    class_labels = ['front', 'left', 'none', 'right']
    df_sub['per_class_accuracy_mean'] = df_sub['per_class_accuracy_mean'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_sub = df_sub.sort_values('L')

    # 设置绘图
    fig, axs = plt.subplots(2, 2, figsize=(18, 6), sharey=True)
    axs = axs.flatten()

    for i, cls in enumerate(class_labels):
        acc_values = [acc[i] for acc in df_sub['per_class_accuracy_mean']]
        axs[i].plot(df_sub['L'], acc_values, marker='o', linewidth=2)
        axs[i].set_title(f'Class: {cls}')
        axs[i].set_xlabel('L (log scale)')
        axs[i].set_xscale('log', base=2)
        axs[i].set_xticks([1, 2, 4, 8, 16])  # 保证标签显示正常
        axs[i].get_xaxis().set_major_formatter(plt.ScalarFormatter())  # 显示原始整数
        if i == 0:
            axs[i].set_ylabel('Accuracy')
        axs[i].set_ylim(0.8, 1)
        axs[i].grid(True)

    plt.suptitle('Per-Class Accuracy vs L (SAB, res=240, freq=50–1500)')
    plt.tight_layout()
    plt.savefig("figures/per_class_accuracy_comparison_SAB_logL.png")
    plt.close()


def plot_srp_polar_for_sample(sample_id="3_00_0013", preprocess_dir="preprocess"):
    pattern = r"metadata_samples_L(\d+)_r240_f50-1500\.csv"
    os.makedirs("figures/srp_polar", exist_ok=True)

    for fname in os.listdir(preprocess_dir):
        match = re.match(pattern, fname)
        if not match:
            continue

        L = int(match.group(1))
        file_path = os.path.join(preprocess_dir, fname)
        df = pd.read_csv(file_path)

        # 查找指定 ID 的样本
        target_row = df[df["ID"] == sample_id]
        if target_row.empty:
            print(f"❌ Sample {sample_id} not found in {fname}")
            continue

        feat_cols = [col for col in df.columns if col.startswith("feat")]
        feature = target_row[feat_cols].values[0].astype(float)

        segment_len = len(feature) // L
        theta = np.linspace(-90, 90, segment_len)
        theta_rad = np.deg2rad(theta)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 4))
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        # 设置刻度标签从 -90 到 +90
        angles = np.array([-90, -60, -30, 0, 30, 60, 90])
        angles_for_plot = (angles + 360) % 360
        ax.set_thetagrids(angles_for_plot, labels=[f"{a}°" for a in angles])

        # 限制极坐标范围只显示上半部分（0 到 π 弧度）
        ax.set_ylim(0, None)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)

        for i in range(L):
            segment = feature[i * segment_len:(i + 1) * segment_len]
            ax.plot(theta_rad, segment, label=f"Segment {i+1}")

            # 红点标最大方向
            max_idx = np.argmax(segment)
            max_theta = theta_rad[max_idx]
            max_value = segment[max_idx]
            ax.plot(max_theta, max_value, 'ro')

        ax.set_title(f"SRP Energy Direction for Sample {sample_id} (L={L})", va='bottom')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.tight_layout()
        plt.savefig(f"figures/srp_polar/srp_polar_L{L}_{sample_id}_half.png")
        plt.close()


def plot_delta_histogram_by_class_separate(preprocess_dir="preprocess", target_L=2):
    pattern = rf"metadata_samples_L{target_L}_r240_f50-1500\.csv"

    delta_by_class = {'left': [], 'right': []}

    for fname in os.listdir(preprocess_dir):
        if not re.match(pattern, fname):
            continue

        df = pd.read_csv(os.path.join(preprocess_dir, fname))
        feat_cols = [col for col in df.columns if col.startswith("feat")]

        segment_len = len(feat_cols) // target_L
        theta = np.linspace(-90, 90, segment_len)

        for idx, row in df.iterrows():
            class_label = str(row['Class']).lower()
            if class_label not in delta_by_class:
                continue  # 只分析 left 和 right

            feature = row[feat_cols].values.astype(float)

            for i in range(target_L):
                segment = feature[i * segment_len:(i + 1) * segment_len]
                left_half = segment[theta < 0]
                right_half = segment[theta >= 0]

                if len(left_half) == 0 or len(right_half) == 0:
                    continue

                left_peak = np.max(left_half)
                right_peak = np.max(right_half)
                delta = right_peak - left_peak
                delta_by_class[class_label].append(delta)

    # 设置直方图
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bins = np.linspace(-1, 1, 40)

    for i, cls in enumerate(['left', 'right']):
        ax = axs[i]
        data = delta_by_class[cls]
        mean_val = np.mean(data)
        median_val = np.median(data)

        ax.hist(data, bins=bins, color='skyblue' if cls == 'left' else 'salmon', edgecolor='black', alpha=0.7)
        ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean Δ = {mean_val:.3f}")
        ax.axvline(median_val, color='blue', linestyle='-.', label=f"Median Δ = {median_val:.3f}")
        ax.set_title(f"Δ Histogram (Class = {cls})")
        ax.set_xlabel("Δ = right_peak - left_peak")
        if i == 0:
            ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

    plt.suptitle(f"SRP Δ Distribution by Class (L={target_L})")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/delta_histogram_by_class_L{target_L}_split.png")
    plt.close()


def draw_model_comparison_bar(
        cnn_df, svm_df,
        filename="barplot_cnn_vs_svm.png",
        average=True,
        L=None, resolution=None, fmin=None, fmax=None, location=None
):
    os.makedirs("figures", exist_ok=True)

    def is_valid_vector(arr):
        return isinstance(arr, list) and len(arr) == 4

    def extract_metrics(df):
        if not average:
            if L is not None:
                df = df[df['L'] == L]
            if resolution is not None:
                df = df[df['resolution'] == resolution]
            if fmin is not None:
                df = df[df['fmin'] == fmin]
            if fmax is not None:
                df = df[df['fmax'] == fmax]
            if location is not None:
                df = df[df['location'] == location]

        df_clean = df[
            df['per_class_accuracy_mean'].apply(is_valid_vector) &
            df['per_class_precision_mean'].apply(is_valid_vector) &
            df['per_class_recall_mean'].apply(is_valid_vector) &
            df['per_class_iou_mean'].apply(is_valid_vector)
            ]

        if df_clean.empty:
            print("❌ No valid entries found with the given parameters.")
            return None

        return {
            'accuracy_per_class': np.nanmean(np.vstack(df_clean['per_class_accuracy_mean']), axis=0),
            'precision_per_class': np.nanmean(np.vstack(df_clean['per_class_precision_mean']), axis=0),
            'recall_per_class': np.nanmean(np.vstack(df_clean['per_class_recall_mean']), axis=0),
            'iou_per_class': np.nanmean(np.vstack(df_clean['per_class_iou_mean']), axis=0)
        }

    cnn_metrics = extract_metrics(cnn_df)
    svm_metrics = extract_metrics(svm_df)

    if cnn_metrics is None or svm_metrics is None:
        return  # 无有效数据

    labels = ['front', 'left', 'none', 'right']
    x = np.arange(len(labels))
    width = 0.35

    metric_names = ['accuracy_per_class', 'precision_per_class', 'recall_per_class', 'iou_per_class']
    titles = ['Per-Class Accuracy', 'Per-Class Precision', 'Per-Class Recall', 'Per-Class IoU']

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.ravel()

    for i, metric in enumerate(metric_names):
        axs[i].bar(x - width/2, cnn_metrics[metric], width, label='CNN', color='tab:blue')
        axs[i].bar(x + width/2, svm_metrics[metric], width, label='SVM', color='tab:orange')
        axs[i].set_title(titles[i])
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(labels)
        axs[i].set_ylim(0, 1)
        axs[i].set_ylabel('Score')
        axs[i].legend()

    # 构造标题后缀
    if average:
        title_suffix = "Average Across All"
    else:
        parts = []
        if L: parts.append(f"L={L}")
        if resolution: parts.append(f"res={resolution}")
        if fmin and fmax: parts.append(f"f={fmin}-{fmax}")
        if location: parts.append(f"loc={location}")
        title_suffix = ", ".join(parts)

    plt.suptitle(f"CNN vs SVM Classification Metrics by Class\n({title_suffix})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"figures/{filename}")
    plt.close()


def draw_confusion_matrix_from_csv(csv_path, labels, title, out_path):
    df = pd.read_csv(csv_path)
    y_true = df['true_label']
    y_pred = df['predicted_label']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                vmin=0.0, vmax=1.0)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def draw_all_confusion_matrices(cnn_dir, svm_dir, labels, locations=["SAB", "DAB", "SA", "SB", "DA", "DB"]):
    for model, base_dir in [("CNN", cnn_dir), ("SVM", svm_dir)]:
        for loc in locations:
            filename = f"L2_r240_f50-1500_{loc}_predictions.csv"
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
                title = f"{model} Confusion Matrix ({loc})"
                out_img = os.path.join("figures", f"confmat_{model.lower()}_{loc}.png")
                draw_confusion_matrix_from_csv(path, labels, title, out_img)
            else:
                print(f"❌ Not found: {path}")


def plot_confusion_matrix_from_csv(csv_path, title="Confusion Matrix", save_path=None, figsize=(6, 5), cmap="Blues", normalize=True):
    """
    从 CSV 文件读取并绘制（可归一化）混淆矩阵热力图。

    参数:
        csv_path (str): 混淆矩阵 CSV 文件路径（行列为标签）
        title (str): 图表标题
        save_path (str): 若指定，则保存图像到该路径
        figsize (tuple): 图像尺寸
        cmap (str): 热力图配色
        normalize (bool): 是否按行归一化（每个类别总数为1）
    """
    conf_mat_df = pd.read_csv(csv_path, index_col=0)
    class_names = conf_mat_df.columns.tolist()

    if normalize:
        conf_mat_normalized = conf_mat_df.div(conf_mat_df.sum(axis=1).replace(0, 1), axis=0)
    else:
        conf_mat_normalized = conf_mat_df

    plt.figure(figsize=figsize)
    sns.heatmap(conf_mat_normalized, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1 if normalize else None)

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ 混淆矩阵已保存为：{save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrices_mydata(df, modelname, num_class=4, save_dir="./figures/conf_matrices"):
    '''
    对 DataFrame 中每一行的混淆矩阵绘图并保存，自动根据文件名识别类别数并生成 labels。

    参数：
        df (DataFrame): 包含 'conf_mat'、'filename' 和可选 'location' 列的 DataFrame。
        modelname (str): 模型名称，用于命名输出文件。
        save_dir (str): 输出图像保存的文件夹路径。
    '''
    df = df[(df['L'] == 2) & (df['resolution'] == 30) & (df['fmin'] == 50) & (df['fmax'] == 1500) & (df['num_class'] == num_class)]
    os.makedirs(save_dir, exist_ok=True)

    pattern = r"_L(\d+)_r(\d+)_f(\d+)-(\d+)_w(\d+)_pre(\d+)_post(\d+)_mid(\d+)_mw(\d+)_c(\d+)"

    for idx, row in df.iterrows():
        conf_raw = row["conf_mat"]
        if isinstance(conf_raw, str):
            try:
                conf_matrix = np.array(ast.literal_eval(conf_raw))
            except Exception as e:
                print(f"[Row {idx}] Error parsing conf_mat: {e}")
                continue
        elif isinstance(conf_raw, list):
            conf_matrix = np.array(conf_raw)
        else:
            print(f"[Row {idx}] Unsupported conf_mat type: {type(conf_raw)}")
            continue

        if "filename" not in row or not isinstance(row["filename"], str):
            print(f"[Row {idx}] Missing or invalid filename.")
            continue

        match = re.search(pattern, row["filename"])
        if not match:
            print(f"[Row {idx}] Filename pattern not matched.")
            continue

        try:
            num_class = int(match.group(10))
            if num_class < 4:
                print(f"[Row {idx}] num_class too small: {num_class}")
                continue

            orig_class_names = ["front_1"] + ["left", "none", "right"] + [f"front_{i}" for i in range(num_class - 3) if i != 1]
            if num_class == 6:
                class_names = ["left", "none", "right", "front_0", "front_1", "front_2"]
                reorder_indices = [1, 2, 3, 4, 0, 5]
                conf_matrix = conf_matrix[reorder_indices, :][:, reorder_indices]
            else:
                class_names = orig_class_names

        except Exception as e:
            print(f"[Row {idx}] Error extracting class names: {e}")
            continue

        with np.errstate(invalid='ignore', divide='ignore'):
            conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix_percent = np.nan_to_num(conf_matrix_percent) * 100
            conf_matrix_percent = np.around(conf_matrix_percent, decimals=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(conf_matrix_percent, cmap="Blues", vmin=0, vmax=100)
        fig.colorbar(cax, format='%d%%')

        for i in range(conf_matrix_percent.shape[0]):
            for j in range(conf_matrix_percent.shape[1]):
                ax.text(j, i, f"{conf_matrix_percent[i, j]:.1f}%",
                        ha="center", va="center",
                        color="black" if conf_matrix_percent[i, j] < 70 else "white")

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        base_name = os.path.splitext(os.path.basename(row["filename"]))[0]
        location_part = row.get("location", "unknown")
        save_path = os.path.join(save_dir, f"conf_mat_{base_name}_{location_part}_{modelname}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close()


def plot_confusion_matrices_samples(df, modelname, save_dir="./figures/conf_matrices"):
    """
    对 DataFrame 中每一行的混淆矩阵绘图并保存，自动根据文件名识别类别数并生成 labels。

    参数：
        df (DataFrame): 包含 'conf_mat'、'filename' 和可选 'location' 列的 DataFrame。
        modelname (str): 模型名称，用于命名输出文件。
        save_dir (str): 输出图像保存的文件夹路径。
    """
    df = df[(df['L'] == 2) & (df['resolution'] == 30) & (df['fmin'] == 50) & (df['fmax'] == 1500) & (df['window_length'] == 1000)]
    os.makedirs(save_dir, exist_ok=True)

    for idx, row in df.iterrows():
        conf_raw = row["conf_mat"]
        if isinstance(conf_raw, str):
            try:
                conf_matrix = np.array(ast.literal_eval(conf_raw))
            except Exception as e:
                print(f"[Row {idx}] Error parsing conf_mat: {e}")
                continue
        elif isinstance(conf_raw, list):
            conf_matrix = np.array(conf_raw)
        else:
            print(f"[Row {idx}] Unsupported conf_mat type: {type(conf_raw)}")
            continue

        # 归一化为百分比
        with np.errstate(invalid='ignore', divide='ignore'):
            conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix_percent = np.nan_to_num(conf_matrix_percent) * 100
            conf_matrix_percent = np.around(conf_matrix_percent, decimals=1)

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(conf_matrix_percent, cmap="Blues", vmin=0, vmax=100)
        fig.colorbar(cax, format='%d%%')

        for i in range(conf_matrix_percent.shape[0]):
            for j in range(conf_matrix_percent.shape[1]):
                ax.text(j, i, f"{conf_matrix_percent[i, j]:.1f}%",
                        ha="center", va="center",
                        color="black" if conf_matrix_percent[i, j] < 70 else "white")

        labels = ['front', 'left', 'none', 'right']

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.title(f"Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        # 保存文件名
        base_name = os.path.splitext(os.path.basename(row["filename"]))[0]
        location_part = row.get("location", "unknown")
        save_path = os.path.join(save_dir, f"conf_mat_{base_name}_{location_part}_{modelname}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
        plt.close()


def plot_learning_curve(df, metric="overall_accuracy", save_path="figures/learning_curve_plot.png"):
    """
    绘制学习曲线图
    参数:
        df: 包含 train_ratio, accuracy_mean, accuracy_std, model 等列的DataFrame
        metric: "accuracy", "iou", "recall" 等指标名
    """
    metric_mean = f"{metric}_mean"
    metric_std = f"{metric}_std"

    if not all(col in df.columns for col in [metric_mean, metric_std, 'train_ratio', 'model']):
        raise ValueError("缺少必要字段，请确认包含 train_ratio、model、accuracy_mean 和 accuracy_std 等列")

    df["train_ratio"] = df["train_ratio"].astype(float)
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="train_ratio", y=metric_mean, hue="model", marker="o", err_style="bars", errorbar=None)

    # 添加误差条
    # for model in df["model"].unique():
    #     sub_df = df[df["model"] == model]
    #     plt.errorbar(sub_df["train_ratio"], sub_df[metric_mean], yerr=sub_df[metric_std], fmt='none', capsize=5)

    plt.title(f"Learning Curve ({metric})")
    plt.xlabel("Training Set Ratio")
    plt.ylabel(f"Validation {metric.capitalize()}")
    plt.ylim(0, 1.0)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()


def line_accuracy_vs_num_class(df, location_name=None, filename_suffix="all"):
    """
    绘制 overall_accuracy_mean 随 num_class 变化的折线图。

    参数:
        df (DataFrame): 包含 num_class 和 overall_accuracy_mean 的 DataFrame。
        location_name (str, 可选): 若指定，则仅绘制该 location 的数据。
        filename_suffix (str): 保存图像时用作文件名后缀。
    """
    if location_name:
        subset = df[df['location'] == location_name]
        title_loc = location_name
    else:
        subset = df
        title_loc = "All Locations"

    # 按 num_class 计算平均准确率
    grouped = subset.groupby("num_class")["overall_accuracy_mean"].mean().reset_index()

    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(data=grouped, x="num_class", y="overall_accuracy_mean", marker="o")
    plt.title(f"Overall Accuracy vs Number of Classes\n(Mean accuracy on {title_loc})")
    plt.xlabel("Number of Classes")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)

    # 保存图像
    os.makedirs("figures", exist_ok=True)
    save_path = f"figures/lineplot_accuracy_vs_num_class_{filename_suffix}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved figure to {save_path}")
    plt.close()


def class_count():
    # 读取CSV文件
    df = pd.read_csv("./preprocess_mydata/thesis_mydata/metadata_samples_L2_r30_f50-1500_w100_pre100_post15_mid25_mw1000_c10.csv")

    # 添加筛选条件，仅保留指定的Environment
    target_envs = ['SA1', 'SA2', 'SB1', 'SB2', 'SB3']
    df = df[df['Environment'].isin(target_envs)]

    # 指定类别顺序与对应标签
    custom_order = [1, 2, 3, 4, 5, 6, 0, 7, 8, 9]
    class_labels = {
        1: 'left',
        2: 'none',
        3: 'right',
        4: 'front0',
        5: 'front1',
        6: 'front2',
        0: 'front3',
        7: 'front4',
        8: 'front5',
        9: 'front6'
    }

    # 统计每个类别的样本数量，并按照自定义顺序排序
    class_counts = df['Class'].value_counts().reindex(custom_order, fill_value=0)
    class_counts.index = [class_labels[i] for i in custom_order]  # 替换为字符串标签

    # 输出统计结果
    print("每个类别的样本数量：")
    print(class_counts)

    # 可视化绘图
    plt.figure(figsize=(8, 5))
    class_counts.plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples per Class')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('./figures/class_count.png')

    # 示例：按Environment与Class组合统计样本数量
    group_counts = df.groupby(['Environment', 'Class']).size().unstack(fill_value=0)
    print(group_counts)



if __name__ == '__main__':
    svm_4_class_df = loadCSVData(path="./preprocess_samples/features_samples/result/summary_results_SVM.csv")
    cnn_4_class_df = loadCSVData(path="./preprocess_samples/features_samples/result/summary_metrics_CNN.csv")
    cnn_multi_class_df = loadCSVData(path="./preprocess_mydata/thesis/summary_metrics.csv")

    heatMap_freq50_1500_by_location(df=svm_4_class_df, location_name="SAB", filename_suffix="SVM_SAB", modelname="SVM")
    heatMap_freq50_1500_by_location(df=cnn_4_class_df, location_name="SAB", filename_suffix="CNN_SAB", modelname="CNN")

    bar_accuracyVsFreqRange(grouped=svm_4_class_df, modelname="SVM")
    bar_accuracyVsFreqRange(grouped=cnn_4_class_df, modelname="CNN")

    plot_window_comparison_dual(df_svm=svm_4_class_df, df_cnn=cnn_4_class_df)

    line_accuracy_vs_num_class(df=cnn_multi_class_df, location_name="SAB", filename_suffix="SAB")

    plot_confusion_matrices_samples(df=svm_4_class_df, modelname="SVM_4_Class")
    plot_confusion_matrices_samples(df=cnn_4_class_df, modelname="CNN_4_Class")
    plot_confusion_matrices_mydata(df=cnn_multi_class_df, modelname="CNN_mydata_4_Class", num_class=4)
    plot_confusion_matrices_mydata(df=cnn_multi_class_df, modelname="CNN_mydata_6_Class", num_class=6)

    class_count()

    # plot_window_comparison_by_location(df=svm_4_class_df, model="SVM")
    # plot_window_comparison_by_location(df=cnn_4_class_df, model="CNN")

    # svm_df = loadCSVData(path="./preprocess_samples/result/summary_results_SVM_samples.csv")
    # cnn_df = loadCSVData(path="./preprocess_samples/result/summary_metrics_CNN_samples.csv")
    # svm_freq_df = loadCSVData(path="./preprocess_samples/current/freq/result/summary_results_SVM.csv")
    # cnn_freq_df = loadCSVData(path="./preprocess_samples/current/freq/result/summary_metrics_CNN.csv")
    # svm_window_df = loadCSVData(path="./preprocess_samples/current/window/result/summary_results_SVM.csv")
    # cnn_window_df = loadCSVData(path="./preprocess_samples/current/window/result/summary_metrics_CNN.csv")
    # cnn_mydata_df = loadCSVData(path="./preprocess_mydata/balanced/summary_metrics_balanced_CNN.csv")
    # heavycnn_mydata_df = loadCSVData(path="./preprocess_mydata/balanced/summary_metrics_balanced_HeavyCNN.csv")
    # cnn_mydata_unbalanced_df = loadCSVData(path="./preprocess_mydata/test/summary_metrics_unbalanced_100ms.csv")
    # svm_mydata_df = loadCSVData(path="./preprocess_mydata/summary_metrics_SVM.csv")
    # learning_curve_df = loadCSVData(path="./preprocess_test/result/learning_curve.csv")

    # accuracy vs num_class
    # line_accuracy_vs_num_class(df=cnn_mydata_unbalanced_df, location_name='SAB', filename_suffix='SAB')

    # Learning Curve
    # plot_learning_curve(learning_curve_df, metric="overall_accuracy")

    # Confusion Matrices
    # plot_confusion_matrices(df=cnn_mydata_df, modelname="CNN")
    # plot_confusion_matrices(df=heavycnn_mydata_df, modelname="HeavyCNN")
    # plot_confusion_matrices(df=cnn_mydata_unbalanced_df, modelname="CNN_Unbalanced")
    # plot_confusion_matrices(df=svm_df, modelname="SVM", labels=["front", "left", "none", "right"])
    # plot_confusion_matrices(df=cnn_df, modelname="CNN", labels=["front", "left", "none", "right"])
    # plot_confusion_matrices(df=svm_mydata_df, modelname="SVM_mydata")

    # draw_model_comparison_bar(cnn_df, svm_df, filename="barplot_cnn_vs_svm.png", average=False, L=2, resolution=240, fmax=1500, fmin=50, location=None)
    # draw_model_comparison_bar(cnn_df, svm_df, filename="barplot_cnn_vs_svm_average_within_freq.png", average=False, L=None, resolution=None, fmax=1500, fmin=50, location=None)
    # draw_model_comparison_bar(cnn_df, svm_df, filename="barplot_cnn_vs_svm_average.png")
    # for loc in cnn_df['location'].unique():
    #     draw_model_comparison_bar(cnn_df, svm_df, average=False, location=loc, filename=f"barplot_cnn_vs_svm_by_loc_{loc}.png")

    # freq_lists = [[20, 50], [50, 1500], [1500, 3000]]
    # for freq in freq_lists:
    #     fmin = freq[0]
    #     fmax = freq[1]
    #     draw_model_comparison_bar(cnn_df, svm_df, filename=f"cnn_vs_svm_{fmin}_{fmax}.png", average=False, fmax=fmax, fmin=fmin)

    # draw_all_confusion_matrices(
    #     cnn_dir="./preprocess_test/result",
    #     svm_dir="./preprocess/result",
    #     labels = ["front", "left", "none", "right"]
    # )

    # plot_confusion_matrix_from_csv(
    #     csv_path="confmat_L2_res240_run0_SAB.csv",
    #     title="Confusion Matrix (SAB, Normalized)",
    #     save_path="figures/confmat_SAB_normalized.png",
    #     normalize=True
    # )

    # # line_accuracyVsL_by_location(cnn_df, None, "all")
    # # line_accuracyVsL_by_location(cnn_df, "SAB", "SAB")
    # # line_accuracyVsL_by_location(cnn_df, "DAB", "DAB")

    # bar_accuracyVsFreqRange(cnn_freq_df, "CNN")
    # bar_accuracyVsFreqRange(svm_freq_df, "SVM")

    # # heatMap_freq50_1500_by_location(cnn_df, None, "all")
    # heatMap_freq50_1500_by_location(cnn_df, "SAB", "CNN_SAB", "CNN", 50, 1500)
    # heatMap_freq50_1500_by_location(cnn_df, "SAB", "CNN_SAB", "CNN", 1500, 3000)
    # heatMap_freq50_1500_by_location(cnn_df, "DAB", "CNN_DAB", "CNN", 50, 1500)
    # heatMap_freq50_1500_by_location(cnn_df, "DAB", "CNN_DAB", "CNN", 1500, 3000)

    # # heatMap_freq50_1500_by_location(cnn_df, None, "all")
    # heatMap_freq50_1500_by_location(svm_df, "SAB", "SVM_SAB", "SVM", 50, 1500)
    # heatMap_freq50_1500_by_location(svm_df, "DAB", "SVM_DAB", "SVM", 50, 1500)
    # heatMap_freq50_1500_by_location(svm_df, "SAB", "SVM_SAB", "SVM", 1500, 3000)
    # heatMap_freq50_1500_by_location(svm_df, "DAB", "SVM_DAB", "SVM", 1500, 3000)

    # plot_per_class_accuracy_comparison(cnn_df)
    # plot_srp_polar_for_sample()
    # plot_delta_histogram_by_class_separate()

    # Window Length
    # plot_window_comparison_by_location(cnn_window_df, "CNN")
    # plot_window_comparison_by_location(svm_window_df, "SVM")