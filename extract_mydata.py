import os

from extract_clips import AudioProcessor
from collectDataset import cut_files, count_classid_and_ids
from preprocess_srp_mydata import loadMicarray, extract_from_wav_folder

if __name__ == '__main__':
    # window_length_list = [1000]
    # step_pre_t0_list = [1000]
    # step_mid_t0_list = [250]
    # step_post_t0_list = [150]

    window_length_list = [100]
    step_pre_t0_list = [100]
    step_mid_t0_list = [100]
    step_post_t0_list = [100]

    mid_window_length = 1000
    num_class = [4, 5, 6, 7, 8, 9, 10]
    # num_class = [5]

    for j in range(len(num_class)):
        for i in range(len(window_length_list)):
            PATH_ROOT = r'D:\ovad\ovad_dataset'
            processor = AudioProcessor(
                path_root=PATH_ROOT,
                audio_rate=48000,
                video_rate=10,
                window_length_ms=window_length_list[i],     # 每个剪切片段
                step_pre_t0_ms=step_pre_t0_list[i],       # 前段窗口步长
                step_post_t0_ms=step_post_t0_list[i],      # T0后1.5秒窗口步长
                step_mid_t0_ms=step_mid_t0_list[i],       # T0之后窗口步长
                mid_window_length=mid_window_length,
                image_width=1936,
                num_class=num_class[j]
            )

            # 遍历整个数据集目录
            for root, dirs, files in os.walk(PATH_ROOT):
                if processor.should_skip(root):
                    continue
                if 'out_multi.wav' in files:
                    print(f"\n{'=' * 40}")
                    print(f"处理目录: {root}")
                    processor.process_folder(root)

            print("\n✅ 全部处理完成！")

            path_root = r'D:\ovad\ovad_dataset'# r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad'
            target_folder = r'D:\ovad\mydata'
            os.makedirs(target_folder, exist_ok=True)

            cut_files(path_root, target_folder)
            count_classid_and_ids(target_folder)

            L_list = [2]
            resolution_list = [30]
            freq_range_list = [[50, 1500]]

            mic_array = loadMicarray()
            nfft = 512

            for L in L_list:
                for resolution in resolution_list:
                    for freq_range in freq_range_list:
                        fmin, fmax = freq_range
                        print(f"Processing with L = {L}, resolution = {resolution}, freq_range = {fmin}-{fmax}...")

                        extract_from_wav_folder(
                            folder_path=target_folder,
                            save_path=rf".\preprocess_mydata\thesis\metadata_samples_L{L}_r{resolution}_f{fmin}-{fmax}_w{window_length_list[i]}_pre{step_pre_t0_list[i]}_post{step_post_t0_list[i]}_mid{step_mid_t0_list[i]}_mw{mid_window_length}_c{num_class[j]}.csv",
                            mic_array=mic_array,
                            resolution=resolution,
                            freq_range=freq_range,
                            nfft=nfft,
                            L=L
                        )

                        print(f"Finish: L = {L}, resolution = {resolution}, freq_range = {fmin}-{fmax}, window length: {window_length_list[i]}, step pre: {step_pre_t0_list[i]}, step post: {step_post_t0_list[i]}, step mid: {step_mid_t0_list[i]}")
