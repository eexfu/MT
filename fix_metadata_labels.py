import pandas as pd
import os

# 原始 metadata 路径
metadata_path = 'processed_features/metadata.csv'

# 加载 CSV
df = pd.read_csv(metadata_path)

# 替换 label
def extract_classid_from_filename(path):
    filename = os.path.basename(path)
    parts = filename.split("_")
    if len(parts) >= 4:
        try:
            return int(parts[3])  # 第 4 段是 classid
        except ValueError:
            return 0
    return 0

df['label'] = df['feature_path'].apply(extract_classid_from_filename)

# 保存修正后的 metadata.csv
df.to_csv(metadata_path, index=False)
print("✅ metadata.csv 中的 label 已根据文件名成功更新！")
