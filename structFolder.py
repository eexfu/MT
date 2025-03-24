import os
import shutil

# 定义音频和视频的根目录
audio_root = r'D:\ovad\ovad_dataset_audio\ovad_dataset'
video_root = r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad'

# 遍历音频文件夹
for root, dirs, files in os.walk(audio_root):
    for file in files:
        # 构建完整的源文件路径
        audio_file_path = os.path.join(root, file)

        # 计算相对于audio_root的相对路径
        relative_path = os.path.relpath(root, audio_root)

        # 构建目标文件夹的路径
        video_folder_path = os.path.join(video_root, relative_path)

        # 确保目标文件夹存在
        os.makedirs(video_folder_path, exist_ok=True)

        # 构建目标文件路径
        video_file_path = os.path.join(video_folder_path, file)
        # 执行文件复制
        shutil.copy2(audio_file_path, video_file_path)

print("All audio files have been successfully copied to the video folder structure.")
