import os
import shutil

def copy_static_to_dynamic(base_path):
    r"""
    将 S 开头文件夹中的 camera_baseline_results.json 复制到对应 D 开头的文件夹中
    :param base_path: 数据集根目录（例如：r'C:\temp\...\ovad'）
    """
    print(f"🔍 扫描根目录: {base_path}")

    # 遍历所有以 S 开头的文件夹（SA1, SB3 等）
    for src_env in os.listdir(base_path):
        src_env_path = os.path.join(base_path, src_env)
        print(f"\n检查项: {src_env}")

        if not src_env.startswith('S'):
            print(f"  跳过非 S 开头项: {src_env}")
            continue
        if not os.path.isdir(src_env_path):
            print(f"  跳过非文件夹项: {src_env}")
            continue

        # 构建目标文件夹名（S -> D）
        dst_env = 'D' + src_env[1:]
        dst_env_path = os.path.join(base_path, dst_env)
        print(f"  映射目标文件夹: {src_env} → {dst_env}")

        # 自动创建目标文件夹（如果不存在）
        os.makedirs(dst_env_path, exist_ok=True)

        # 递归遍历源文件夹中的 JSON 文件
        for root_dir, _, files in os.walk(src_env_path):
            print(f"\n  扫描子目录: {os.path.relpath(root_dir, base_path)}")
            for file in files:
                if file == 'camera_baseline_detections.json':
                    src_file = os.path.join(root_dir, file)
                    relative_path = os.path.relpath(root_dir, src_env_path)
                    dst_dir = os.path.join(dst_env_path, relative_path)
                    dst_file = os.path.join(dst_dir, file)

                    os.makedirs(dst_dir, exist_ok=True)
                    try:
                        shutil.copy2(src_file, dst_file)
                        print(f'    ✅ 复制成功: {os.path.relpath(src_file, base_path)} → {os.path.relpath(dst_file, base_path)}')
                    except Exception as e:
                        print(f'    ❌ 复制失败: {os.path.relpath(src_file, base_path)}\n      错误信息: {str(e)}')
                else:
                    print(f"    忽略文件: {file}")

if __name__ == '__main__':
    base_path = r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad'

    if not os.path.exists(base_path):
        print(f"错误：根目录不存在 -> {base_path}")
    else:
        copy_static_to_dynamic(base_path)
        print("\n处理完成 ✅")