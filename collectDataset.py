# This code was completed with assistance from ChatGPT.
import os
import shutil
from collections import defaultdict

# Image width
image_width = 1936
center_line = image_width / 2

def cut_files(path_root, target_folder):
    """Cut wav and json files from all result folders to target folder"""
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
        print(f"Target folder cleared: {target_folder}")

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

    print(f"Total {file_count} files moved to {target_folder}")


def count_classid_and_ids(target_folder):
    """Count classification IDs and original IDs based on filenames, also count by sub-dataset"""

    # Overall statistics
    classid_stats_total = defaultdict(int)
    original_id_stats = defaultdict(int)
    classid_mapping = defaultdict(set)

    # Statistics for each sub-dataset
    subdatasets = ['SA', 'SB', 'SAB', 'DA', 'DB', 'DAB']
    subdataset_stats = {key: defaultdict(int) for key in subdatasets}

    # Map LOC_ID to sub-dataset
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
            continue  # Skip other non-target files

        total_files += 1

        if not filename.endswith('.json'):
            continue

        parts = filename.split('_')
        try:
            loc_id = parts[1]  # Format like "00"
            original_id = '_'.join(parts[:3])
            classid = int(parts[3])

            location = LOC_MAP.get(loc_id)
            if location is None:
                print(f" Unrecognized LOC ID: {loc_id} in {filename}")
                continue

            group = LOC_TO_GROUP.get(location)
            if group is None:
                print(f" Cannot assign to sub-dataset: {location} in {filename}")
                continue

            # Overall statistics
            classid_stats_total[classid] += 1
            original_id_stats[original_id] += 1
            class_mapped = CLASS_MAP.get(classid, "unknown")
            classid_mapping[classid].add(class_mapped)

            # Statistics for each sub-dataset
            subdataset_stats[group][classid] += 1

            # Special handling for SAB and DAB: SAB=SA+SB, DAB=DA+DB
            if group in ["SA", "SB"]:
                subdataset_stats["SAB"][classid] += 1
            if group in ["DA", "DB"]:
                subdataset_stats["DAB"][classid] += 1

        except (IndexError, ValueError) as e:
            print(f" Filename parsing failed: {filename} - {str(e)}")
            continue

    # Print statistics
    print(f"\n Total files: {total_files} ( {total_wav} .wav, {total_json} .json)")

    print("\n Overall ClassID distribution:")
    for cid, count in sorted(classid_stats_total.items()):
        mapped_classes = ', '.join(classid_mapping[cid])
        print(f"  ClassID {cid} ({mapped_classes}): {count}")

    print(f"\n Number of original IDs: {len(original_id_stats)}")

    print("\n Original ID occurrence count (top 10):")
    for idx, (oid, count) in enumerate(sorted(original_id_stats.items(), key=lambda x: x[1], reverse=True)):
        if idx >= 10:
            break
        print(f"  {oid}: {count}")

    print("\n Class ID mapping check:")
    for cid, classes in classid_mapping.items():
        if len(classes) > 1:
            print(f" ClassID {cid} has conflicting mappings: {classes}")
        else:
            print(f" ClassID {cid} consistently mapped to: {classes.pop()}")

    print("\n ClassID distribution in sub-datasets:")
    for subset in subdatasets:
        print(f"\n-- {subset} --")
        subset_stats = subdataset_stats[subset]
        total = sum(subset_stats.values())
        print(f"  Total samples: {total}")
        for cid in sorted(subset_stats.keys()):
            print(f"    ClassID {cid} ({CLASS_MAP.get(cid, 'unknown')}): {subset_stats[cid]}")


if __name__ == '__main__':
    cut_files()
    count_classid_and_ids()
