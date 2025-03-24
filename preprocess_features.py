import os
import torch
import pandas as pd
from tqdm import tqdm
from getFeature import extract_srp_phat_features

# 路径
dataset_path = r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad\dataset'
output_dir = 'processed_features'
os.makedirs(output_dir, exist_ok=True)

metadata = []

for file in tqdm(os.listdir(dataset_path)):
    if file.endswith('.wav'):
        wav_path = os.path.join(dataset_path, file)

        # ✅ 从文件名中提取 classid
        parts = os.path.splitext(file)[0].split("_")
        if len(parts) < 4:
            print(f"⚠️ 文件名格式不正确，跳过：{file}")
            continue
        try:
            label = int(parts[3])
        except ValueError:
            print(f"⚠️ 无法解析 label，跳过：{file}")
            continue

        # 提取特征
        features = extract_srp_phat_features(wav_path)
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # 保存为 .pt 文件
        out_filename = os.path.splitext(file)[0] + '.pt'
        out_path = os.path.join(output_dir, out_filename)
        torch.save(features, out_path)

        metadata.append([out_path, label])

# 保存 metadata.csv
df = pd.DataFrame(metadata, columns=["feature_path", "label"])
df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
print("✅ 所有特征提取完毕，metadata.csv 已生成。")