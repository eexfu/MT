import os
import shutil
from collections import defaultdict

# 根路径（你的ovad_dataset的根目录）
path_root = r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad'

# 目标路径（所有文件要集中放到这里）
target_folder = r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad\dataset'
os.makedirs(target_folder, exist_ok=True)

# 图像宽度
image_width = 1936
center_line = image_width / 2

CLASS_MAP = {
    0: "none",
    1: "left",
    2: "right",
    3: "front"
}

def cut_files():
    """从所有result文件夹剪切wav和json到目标文件夹（实际仍为复制）"""
    file_count = 0

    for root, dirs, files in os.walk(path_root):
        if 'result' in dirs:
            result_folder = os.path.join(root, 'result')

            for file in os.listdir(result_folder):
                if file.endswith('.wav') or file.endswith('.json'):
                    source_file = os.path.join(result_folder, file)
                    target_file = os.path.join(target_folder, file)

                    shutil.move(source_file, target_file)
                    file_count += 1

    print(f"✅ 总共迁移了 {file_count} 个文件到 {target_folder}")


def count_classid_and_ids():
    """根据文件名统计分类ID和原始ID，同时打印总文件数量"""
    # 初始化统计字典
    classid_stats = defaultdict(int)
    original_id_stats = defaultdict(int)
    classid_mapping = defaultdict(set)

    total_files = 0
    total_json = 0
    total_wav = 0

    for filename in os.listdir(target_folder):
        if filename.endswith('.json'):
            total_json += 1
        elif filename.endswith('.wav'):
            total_wav += 1
        else:
            continue  # 跳过其他非目标文件

        total_files += 1

        if not filename.endswith('.json'):
            continue

        # 解析文件名各部分
        parts = filename.split('_')
        try:
            # 示例文件名：1_00_001_1_0.json
            # 原始ID: 1_00_001 (前三个部分)
            # classid: 第4部分 (索引3)
            # 窗口ID: 第5部分 (索引4)
            original_id = '_'.join(parts[:3])
            classid = int(parts[3])
            window_id = parts[4].split('.')[0]

            # 统计分类ID
            classid_stats[classid] += 1

            # 统计原始ID出现次数
            original_id_stats[original_id] += 1

            # 记录classid与实际类别的映射关系
            class_mapped = CLASS_MAP.get(classid, "unknown")
            classid_mapping[classid].add(class_mapped)

        except (IndexError, ValueError) as e:
            print(f"⚠️ 文件名解析失败: {filename} - {str(e)}")
            continue

    # 打印统计结果
    print(f"\n📁 Total number of files: {total_files} ( {total_wav} .wav files, {total_json} .json files)")

    print("\n📊 Count：")
    for cid, count in sorted(classid_stats.items()):
        mapped_classes = ', '.join(classid_mapping[cid])
        print(f"  ClassID {cid} ({mapped_classes}): {count} 个样本")

    print(f"\n Number of original ID: {len(original_id_stats)}")

    print("\n📋 Original ID appear(Top 10): ")
    for idx, (oid, count) in enumerate(sorted(original_id_stats.items(), key=lambda x: x[1], reverse=True)):
        if idx >= 10:
            break
        print(f"  {oid}: {count} ")

    # 验证分类映射一致性
    print("\n🔍 Class ID Map: ")
    for cid, classes in classid_mapping.items():
        if len(classes) > 1:
            print(f"⚠️ ClassID {cid} 存在冲突映射: {classes}")
        else:
            print(f"✅ ClassID {cid} 统一映射为: {classes.pop()}")


if __name__ == '__main__':
    cut_files()
    count_classid_and_ids()
