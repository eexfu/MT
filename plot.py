# This code was completed with assistance from ChatGPT.
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os


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

    return df


def bar_accuracyVsFreqRange(grouped, modelname, L=2, resolution=30, num_mics = 56, geo = 0, window_length = 1000):
    grouped = grouped[(grouped['location'] == 'SAB') & (grouped['num_mics'] == num_mics) & (grouped['geo'] == geo) & (grouped['L'] == L) & (grouped['resolution'] == resolution) & (grouped['window_length'] == window_length)]
    # Bar chart: Mean accuracy for different freq_ranges
    bar_data = grouped.groupby('freq_range')['overall_accuracy_mean'].mean().reset_index()

    # Set order
    freq_order = ["20-50", "50-1500", "1500-3000", "3000-24000"]
    bar_data['freq_range'] = pd.Categorical(bar_data['freq_range'], categories=freq_order, ordered=True)
    bar_data = bar_data.sort_values('freq_range')

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=bar_data, x='freq_range', y='overall_accuracy_mean')

    plt.ylim(0, 1)

    # Add value labels on top of bars
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
    # Filter data for freq_range 50-1500
    df = df[(df['fmin'] == fmin) & (df['fmax'] == fmax) & (df['num_mics'] == num_mics) & (df['geo'] == geo) & (df['window_length'] == window_length) & (df['L'] != 1)]
    # Optionally filter by location
    if location_name:
        df = df[df['location'] == location_name]
        title_suffix = f"Location: {location_name}({modelname})"
    else:
        title_suffix = "All Locations"

    # Group and pivot data for heatmap
    grouped = df.groupby(['L', 'resolution'])['overall_accuracy_mean'].mean().reset_index()
    pivot_data = grouped.pivot(index='resolution', columns='L', values='overall_accuracy_mean')

    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        vmin=0.6, vmax=1.0  # Fixed color scale range
    )
    plt.title(f"Accuracy Heatmap (Freq {fmin}–{fmax})\n{title_suffix}")
    plt.ylabel("Resolution")
    plt.xlabel("L")
    plt.tight_layout()
    plt.savefig(f"figures/heatmap_accuracy_{fmin}_{fmax}_{filename_suffix}.png")
    plt.close()


def plot_window_comparison_by_location(df, model, L=2, resolution=30, fmin = 50, fmax = 1500, num_mics = 56, geo = 0):
    # Filter required columns
    plot_df = df[(df['fmin'] == fmin) & (df['fmax'] == fmax) & (df['num_mics'] == num_mics) & (df['geo'] == geo) & (df['L'] == L) & (df['resolution'] == resolution)].copy()

    # Ensure window_length is numeric and sorted
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
    # Preprocess DataFrames for both models
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

    # Combine both DataFrames
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


def plot_confusion_matrices_mydata(df, modelname, num_class=4, save_dir="./figures/conf_matrices"):
    '''
    Plot and save confusion matrices for each row in the DataFrame, automatically identifying number of classes from filename and generating labels.

    Parameters:
        df (DataFrame): DataFrame containing 'conf_mat', 'filename' and optional 'location' columns.
        modelname (str): Model name for output file naming.
        save_dir (str): Directory path for saving output images.
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


def line_accuracy_vs_num_class(df, location_name=None, filename_suffix="all"):
    """
    Plot line chart of overall_accuracy_mean vs num_class.

    Parameters:
        df (DataFrame): DataFrame containing num_class and overall_accuracy_mean.
        location_name (str, optional): If specified, only plot data for this location.
        filename_suffix (str): Suffix for saved image filename.
    """
    if location_name:
        subset = df[df['location'] == location_name]
        title_loc = location_name
    else:
        subset = df
        title_loc = "All Locations"

    # Calculate mean accuracy by num_class
    grouped = subset.groupby("num_class")["overall_accuracy_mean"].mean().reset_index()

    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(data=grouped, x="num_class", y="overall_accuracy_mean", marker="o")
    plt.title(f"Overall Accuracy vs Number of Classes\n(Mean accuracy on {title_loc})")
    plt.xlabel("Number of Classes")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)

    # Save figure
    os.makedirs("figures", exist_ok=True)
    save_path = f"figures/lineplot_accuracy_vs_num_class_{filename_suffix}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved figure to {save_path}")
    plt.close()


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

    plot_confusion_matrices_mydata(df=cnn_multi_class_df, modelname="CNN_mydata_6_Class", num_class=6)

    # plot_window_comparison_by_location(df=svm_4_class_df, model="SVM")
    # plot_window_comparison_by_location(df=cnn_4_class_df, model="CNN")