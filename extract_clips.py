import os
import json
import shutil
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple

# =====================
# 全局配置
# =====================
PATH_ROOT = r'C:\temp\KU Leuven\Master\Master Thesis\Database\ovad'
AUDIO_RATE = 48000    # 48kHz
VIDEO_RATE = 10       # 视频帧率（用于帧计算）
WINDOW_LENGTH = 1.0   # 1.0秒窗口
STEP_PRE_T0 = 0.1     # t0前区域步长（100ms）
STEP_POST_T0 = 0.03    # t0后区域步长（50ms）
IMAGE_WIDTH = 1936    # 图像宽度（必须与实际数据一致）

# =====================
# 核心处理类
# =====================
class AudioProcessor:
    def __init__(self):
        self.datalog = self.load_datalog()

    def load_datalog(self) -> pd.DataFrame:
        """加载并预处理DataLog.csv（T0为帧号）"""
        datalog_path = os.path.join(PATH_ROOT, 'DataLog.csv')
        try:
            df = pd.read_csv(datalog_path)
            # 标准化列名并过滤无效数据
            df = df.rename(columns={
                'Class': 'direction',
                'Environment': 'env',
                'ID': 'id',
                'T0': 't0_frame'  # 明确字段含义
            }).dropna(subset=['t0_frame'])

            # 强制转换为整数帧号
            df['t0_frame'] = pd.to_numeric(df['t0_frame'], errors='coerce').astype(int)
            return df
        except Exception as e:
            print(f"DataLog加载错误: {str(e)}")
            return pd.DataFrame()

    def should_skip(self, folder_path: str) -> bool:
        """检查是否需要跳过none目录"""
        return 'none' in folder_path.split(os.sep)

    def get_dataset_info(self, folder_path: str) -> Optional[dict]:
        """动态解析路径信息"""
        parts = folder_path.split(os.sep)
        try:
            # 从后向前搜索关键目录
            env = None
            direction = None
            id = parts[-1]  # 最后一段是ID

            # 查找环境和方向
            for part in reversed(parts[:-1]):
                if part in ['left', 'right', 'none']:
                    direction = part
                elif part.startswith(('S', 'D')) and part[1:].isalnum():
                    env = part
                    break  # 找到环境后停止搜索

            return {
                'env': env,
                'direction': direction,
                'id': id
            }
        except IndexError:
            return None

    def clear_results(self, folder_path: str):
        """清理历史结果"""
        result_dir = os.path.join(folder_path, 'result')
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
            print(f"已清除历史结果: {result_dir}")

    def extract_car_boxes(self, json_path: str, start: float, end: float) -> Dict[str, List[dict]]:
        """提取指定时间段的车辆边界框"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        start_frame = int(start * VIDEO_RATE)
        end_frame = int(end * VIDEO_RATE)
        car_boxes = {}

        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx >= len(data['detections_per_frame']):
                continue

            frame_data = data['detections_per_frame'][frame_idx]
            boxes = [
                {
                    "bbox": frame_data['boxes'][i],
                    "score": float(frame_data['scores'][i])
                }
                for i, cls in enumerate(frame_data['classes_str'])
                if cls == 'car' and frame_data['scores'][i] > 0.85
            ]
            if boxes:
                car_boxes[str(frame_idx)] = boxes
        return car_boxes

    def generate_windows(self, duration: float, t0_sec: float) -> list:
        """生成时间窗口（严格遵循时间段划分）"""
        windows = []

        # Phase 1: 0到t0_sec-1秒（使用大步长）
        if t0_sec > 1:
            pre_end = max(0.0, t0_sec - 1 - WINDOW_LENGTH)
            windows += [
                (t, t + WINDOW_LENGTH)
                for t in np.arange(0.0, pre_end, STEP_PRE_T0)
            ]

        # Phase 2: t0_sec-1秒到t0_sec（使用小步长）
        post_start = max(0.0, t0_sec - 1)
        post_end = min(t0_sec, duration - WINDOW_LENGTH)
        if post_end > post_start:
            windows += [
                (t, t + WINDOW_LENGTH)
                for t in np.arange(post_start, post_end, STEP_PRE_T0)
            ]

        # Phase 3: t0_sec到结束（可选策略，此处保持小步长）
        final_start = max(t0_sec, 0.0)
        final_end = duration - WINDOW_LENGTH
        if final_end > final_start:
            windows += [
                (t, t + WINDOW_LENGTH)
                for t in np.arange(final_start, final_end, STEP_POST_T0)
            ]

        return windows

    def process_folder(self, folder_path: str):
        """处理单个数据文件夹"""
        # 跳过none目录
        if self.should_skip(folder_path):
            print(f"⏩ 跳过none目录: {folder_path}")
            return

        # 解析路径信息
        info = self.get_dataset_info(folder_path)
        if not info:
            print(f"⚠️ 无效路径结构: {folder_path}")
            return

        # 获取DataLog记录
        try:
            record = self.datalog[
                (self.datalog['env'] == info['env']) &
                (self.datalog['id'] == info['id'])
                ]
            if record.empty:
                print(f"⚠️ 无DataLog记录: {info}")
                return

            # 直接获取整数帧号（T0已为帧号）
            t0_frame = int(record['t0_frame'].values[0])
            class_label = str(record['direction'].values[0])

        except Exception as e:
            print(f"数据查询错误: {str(e)}")
            return

        # 准备输出目录
        self.clear_results(folder_path)
        result_dir = os.path.join(folder_path, 'result')
        os.makedirs(result_dir, exist_ok=True)

        # 获取音频信息（用于计算总时长）
        audio_path = os.path.join(folder_path, 'out_multi.wav')
        try:
            with sf.SoundFile(audio_path) as f:
                duration_sec = float(f.frames / f.samplerate)
        except Exception as e:
            print(f"音频文件错误: {str(e)}")
            return

        # 生成时间窗口（单位：秒）
        try:
            # 将帧号转换为秒数用于窗口生成
            t0_sec = t0_frame / VIDEO_RATE
            windows = self.generate_windows(duration_sec, t0_sec)
        except Exception as e:
            print(f"时间窗口生成失败: {str(e)}")
            return

        # 处理每个窗口
        for idx, (start_sec, end_sec) in enumerate(tqdm(windows, desc=f"处理 {info['env']}-{info['id']}")):
            metadata = {
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "t0_frame": t0_frame,
                "dataset_type": "dynamic" if info['env'].startswith('D') else "static",
                "class_id": 0,
                "class_name": "",
                "car_boxes": {}
            }

            if info['env'].startswith('D'):
                # 动态数据集：直接使用DataLog中的方向标签
                metadata.update({
                    "class_id": 1 if class_label.lower() == 'left' else 2,
                    "class_name": class_label.lower(),
                })
            # 修改静态数据集处理部分
            else:
                try:
                    json_path = os.path.join(folder_path, 'camera_baseline_detections.json')
                    if not os.path.exists(json_path):
                        print(f"⚠️ 缺少检测文件: {json_path}")
                        continue

                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                    # 转换时间窗口为秒单位
                    t0_sec = t0_frame / VIDEO_RATE

                    # 判断当前窗口所属时间段
                    if end_sec <= t0_sec - 1:
                        # 时间段：0 < window_end < t0-1 → none
                        class_id, class_name = 0, 'none'
                    elif start_sec >= t0_sec - 1 and end_sec <= t0_sec:
                        # 时间段：t0-1 < window < t0 → 直接使用DataLog中的方向标签
                        class_id = 1 if class_label.lower() == 'left' else 2
                        class_name = class_label.lower()
                    else:
                        # 时间段：window > t0 → 使用窗口结束帧判断left/right/front
                        target_frame = int(start_sec * VIDEO_RATE)
                        class_id, class_name = self.determine_class(json_data, target_frame)

                    # 提取车辆框信息（根据实际需要保留）
                    car_boxes = self.extract_car_boxes(json_path, start_sec, end_sec)

                    metadata.update({
                        "class_id": class_id,
                        "class_name": class_name,
                        "time_period": "pre-t0" if end_sec <= t0_sec -1 else "around-t0" if end_sec <= t0_sec else "post-t0",
                        "car_boxes": car_boxes
                    })
                except Exception as e:
                    print(f"分类处理失败: {str(e)}")
                    continue

            # 构建文件名
            filename = f"{info['id']}_{metadata['class_id']}_{idx}"

            # 剪切音频
            if self.cut_audio(audio_path, start_sec, end_sec, os.path.join(result_dir, f"{filename}.wav")):
                # 保存元数据
                try:
                    with open(os.path.join(result_dir, f"{filename}.json"), 'w') as f:
                        json.dump(metadata, f, indent=2, default=self.json_serializer)
                except Exception as e:
                    print(f"元数据保存失败: {str(e)}")

    def determine_class(self, json_data: dict, t0_frame: int) -> Tuple[int, str]:
        """根据t0帧的车辆位置判断分类"""
        # 边界检查
        if t0_frame < 0 or t0_frame >= len(json_data['detections_per_frame']):
            print(f"⚠️ 帧号越界: {t0_frame} (总帧数: {len(json_data['detections_per_frame'])})")
            return 0, 'none'

        frame_data = json_data['detections_per_frame'][t0_frame]
        boxes = [
            frame_data['boxes'][i]
            for i, cls in enumerate(frame_data['classes_str'])
            if cls == 'car' and frame_data['scores'][i] > 0.85
        ]

        if not boxes:
            print("no boxes")
            return 0, 'none'

        # 计算车辆中心水平位置
        x1, _, x2, _ = boxes[0]
        x_center = (x1 + x2) / 2
        print(f"调试信息 - 帧号: {t0_frame}, 中心位置: {x_center}/{IMAGE_WIDTH}")

        # 分类判断
        if x_center < IMAGE_WIDTH / 3:
            return 1, 'left'
        elif x_center > 2 * IMAGE_WIDTH / 3:
            return 2, 'right'
        else:
            return 3, 'front'

    def json_serializer(self, obj):
        """处理特殊类型的JSON序列化"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def cut_audio(self, audio_path: str, start_sec: float, end_sec: float, output_path: str) -> bool:
        """精确剪切音频（秒单位）"""
        try:
            with sf.SoundFile(audio_path) as f:
                sr = f.samplerate
                start_frame = int(start_sec * sr)
                end_frame = int(end_sec * sr)

                if start_frame < 0 or end_frame > f.frames:
                    print(f"⚠️ 音频剪切范围越界: {start_sec}-{end_sec}s (总时长: {f.frames/sr:.2f}s)")
                    return False

                f.seek(start_frame)
                data = f.read(end_frame - start_frame)
                sf.write(output_path, data, sr)
                return True
        except Exception as e:
            print(f"音频处理失败: {str(e)}")
            return False

# =====================
# 主执行逻辑
# =====================
if __name__ == '__main__':
    processor = AudioProcessor()

    # 遍历数据集目录
    for root, dirs, files in os.walk(PATH_ROOT):
        if processor.should_skip(root):
            continue

        if 'out_multi.wav' in files:
            print(f"\n{'='*40}")
            print(f"处理目录: {root}")
            processor.process_folder(root)

    print("\n处理完成 ✅")