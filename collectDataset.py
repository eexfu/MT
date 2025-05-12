import os
import shutil
from collections import defaultdict

# 图像宽度
image_width = 1936
center_line = image_width / 2

def cut_files(path_root, target_folder):
    """从所有result文件夹剪切wav和json到目标文件夹"""
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
        print(f"🧹 已清空目标文件夹: {target_folder}")

    os.makedirs(target_folder, exist_ok=True)

    file_count = 0

    for root, dirs, files in os.walk(path_root):
        if 'result' in dirs:
            result_folder = os.path.join(root, 'result')

            for file in os.listdir(result_folder):
                if file.endswith('.wav') or file.endswith('.json'):
                    source_file = os.path.join(result_folder, file)
                    target_file = os.path.join(target_folder, file)

                    if os.path.exists(target_file):
                        os.remove(target_file)

                    shutil.move(source_file, target_file)
                    file_count += 1

    print(f"✅ 总共迁移了 {file_count} 个文件到 {target_folder}")


def count_classid_and_ids(target_folder):
    """根据文件名统计分类ID和原始ID，同时按子数据集统计"""

    # 总体统计
    classid_stats_total = defaultdict(int)
    original_id_stats = defaultdict(int)
    classid_mapping = defaultdict(set)

    # 每个子数据集的统计
    subdatasets = ['SA', 'SB', 'SAB', 'DA', 'DB', 'DAB']
    subdataset_stats = {key: defaultdict(int) for key in subdatasets}

    # 映射LOC_ID到子数据集
    LOC_MAP = {
        "00": "SA1",
        "01": "SA2",
        "02": "SB1",
        "03": "SB2",
        "04": "SB3",
        "05": "DA1",
        "06": "DA2",
        "07": "DB1",
        "08": "DB2",
        "09": "DB3"
    }

    LOC_TO_GROUP = {
        "SA1": "SA",
        "SA2": "SA",
        "SB1": "SB",
        "SB2": "SB",
        "SB3": "SB",
        "DA1": "DA",
        "DA2": "DA",
        "DB1": "DB",
        "DB2": "DB",
        "DB3": "DB",
    }


    CLASS_MAP = {
        2: "none",
        1: "left",
        3: "right",
        0: "front",
        4: "front_left",
        5: "front_right"
    }

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

        parts = filename.split('_')
        try:
            loc_id = parts[1]  # 是 "00" 这种
            original_id = '_'.join(parts[:3])
            classid = int(parts[3])

            location = LOC_MAP.get(loc_id)
            if location is None:
                print(f"⚠️ 无法识别的LOC ID: {loc_id} in {filename}")
                continue

            group = LOC_TO_GROUP.get(location)
            if group is None:
                print(f"⚠️ 无法归属到子数据集: {location} in {filename}")
                continue

            # 统计总体
            classid_stats_total[classid] += 1
            original_id_stats[original_id] += 1
            class_mapped = CLASS_MAP.get(classid, "unknown")
            classid_mapping[classid].add(class_mapped)

            # 统计每个子数据集
            subdataset_stats[group][classid] += 1

            # SAB 和 DAB 特别处理：SAB=SA+SB，DAB=DA+DB
            if group in ["SA", "SB"]:
                subdataset_stats["SAB"][classid] += 1
            if group in ["DA", "DB"]:
                subdataset_stats["DAB"][classid] += 1

        except (IndexError, ValueError) as e:
            print(f"⚠️ 文件名解析失败: {filename} - {str(e)}")
            continue

    # 打印统计结果
    print(f"\n📁 总文件数: {total_files} ( {total_wav} .wav, {total_json} .json)")

    print("\n📊 总体 ClassID 分布：")
    for cid, count in sorted(classid_stats_total.items()):
        mapped_classes = ', '.join(classid_mapping[cid])
        print(f"  ClassID {cid} ({mapped_classes}): {count}")

    print(f"\n📋 原始ID数量: {len(original_id_stats)}")

    print("\n📋 原始ID出现次数（前10个）:")
    for idx, (oid, count) in enumerate(sorted(original_id_stats.items(), key=lambda x: x[1], reverse=True)):
        if idx >= 10:
            break
        print(f"  {oid}: {count}")

    print("\n🔍 Class ID映射检查:")
    for cid, classes in classid_mapping.items():
        if len(classes) > 1:
            print(f"⚠️ ClassID {cid} 存在冲突映射: {classes}")
        else:
            print(f"✅ ClassID {cid} 统一映射为: {classes.pop()}")

    print("\n📊 子数据集内 ClassID 分布：")
    for subset in subdatasets:
        print(f"\n-- {subset} --")
        subset_stats = subdataset_stats[subset]
        total = sum(subset_stats.values())
        print(f"  总样本数: {total}")
        for cid in sorted(subset_stats.keys()):
            print(f"    ClassID {cid} ({CLASS_MAP.get(cid, 'unknown')}): {subset_stats[cid]}")


if __name__ == '__main__':
    cut_files()
    count_classid_and_ids()
