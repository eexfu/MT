# This code was completed with assistance from ChatGPT.
import os
import json
import shutil
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple

class AudioProcessor:
    def __init__(self,
                 path_root: str,
                 audio_rate: int = 48000,
                 video_rate: int = 10,
                 window_length_ms: int = 100,
                 step_pre_t0_ms: int = 100,
                 step_post_t0_ms: int = 100,
                 step_mid_t0_ms: int = 100,
                 mid_window_length: int = 1000,
                 image_width: int = 1936,
                 num_class: int = 4):
        self.path_root = path_root
        self.audio_rate = audio_rate
        self.video_rate = video_rate
        self.window_length = window_length_ms
        self.step_pre_t0 = step_pre_t0_ms
        self.step_post_t0 = step_post_t0_ms
        self.step_mid_t0 = step_mid_t0_ms
        self.mid_window_length = mid_window_length
        self.image_width = image_width
        self.num_class = num_class
        self.datalog = self.load_datalog()

    def load_datalog(self) -> pd.DataFrame:
        datalog_path = os.path.join(self.path_root, 'DataLog.csv')
        try:
            df = pd.read_csv(datalog_path)
            df = df.rename(columns={
                'Class': 'direction',
                'Environment': 'env',
                'ID': 'id',
                'T0': 't0_frame'
            }).dropna(subset=['t0_frame'])
            df['t0_frame'] = pd.to_numeric(df['t0_frame'], errors='coerce').astype(int)
            return df
        except Exception as e:
            print(f"DataLog loading error: {str(e)}")
            return pd.DataFrame()

    def should_skip(self, folder_path: str) -> bool:
        return 'none' in folder_path.split(os.sep)

    def get_dataset_info(self, folder_path: str) -> Optional[dict]:
        parts = folder_path.split(os.sep)
        try:
            env, direction, id = None, None, parts[-1]
            for part in reversed(parts[:-1]):
                if part in ['left', 'right', 'none']:
                    direction = part
                elif part.startswith(('S', 'D')) and part[1:].isalnum():
                    env = part
                    break
            return {'env': env, 'direction': direction, 'id': id}
        except IndexError:
            return None

    def clear_results(self, folder_path: str):
        result_dir = os.path.join(folder_path, 'result')
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
            print(f"Cleared previous results: {result_dir}")

    def extract_car_boxes(self, json_path: str, start: float, end: float) -> Dict[str, List[dict]]:
        with open(json_path, 'r') as f:
            data = json.load(f)

        start_frame = int(start * self.video_rate)
        end_frame = int(end * self.video_rate)
        car_boxes = {}

        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx >= len(data['detections_per_frame']):
                continue
            frame_data = data['detections_per_frame'][frame_idx]
            boxes = [
                {"bbox": frame_data['boxes'][i], "score": float(frame_data['scores'][i])}
                for i, cls in enumerate(frame_data['classes_str'])
                if cls == 'car' and frame_data['scores'][i] > 0.75
            ]
            if boxes:
                car_boxes[str(frame_idx)] = boxes
        return car_boxes

    def generate_windows(self, duration: float, t0_sec: float) -> list:
        windows = []
        win_len = self.window_length
        t0_ms = int(t0_sec * 1000)
        duration_ms = int(duration * 1000)

        # Phase 1: 0 to t0 - 1s
        pre_end_ms = max(0, t0_ms - 1000 - win_len)
        for t_ms in range(0, pre_end_ms, self.step_pre_t0):
            windows.append((t_ms / 1000, (t_ms + win_len) / 1000))

        # Phase 2: t0 - 1s to t0
        post_start_ms = max(0, t0_ms - self.mid_window_length)
        post_end_ms = min(t0_ms, duration_ms - win_len)
        for t_ms in range(post_start_ms, post_end_ms, self.step_mid_t0):
            windows.append((t_ms / 1000, (t_ms + win_len) / 1000))

        # Phase 3: t0 + 1.5s to end
        final_start_ms = max(t0_ms + 1500, 0)
        final_end_ms = duration_ms - win_len
        for t_ms in range(final_start_ms, final_end_ms, self.step_post_t0):
            windows.append((t_ms / 1000, (t_ms + win_len) / 1000))

        return windows

    def determine_front_subclass(self, car_boxes: Dict[str, List[dict]], image_width: int, num_class: int = 6) -> Tuple[str, int]:
        """
        Dynamically divide the front region into subclasses based on detection box positions, with num_class - 3 subclasses.
        Returns subclass name (e.g., front, front_0, front_1, etc.) and corresponding class ID.
        Class ID definition:
          - "front" is 0
          - "none" is 2 (no car detected)
          - "front_0" is 4, and subsequent IDs increment
        """
        assert num_class >= 3, "num_class must be at least 3"
        num_front_subclasses = num_class - 3
        region_counts = [0] * num_front_subclasses

        for frame_idx, boxes in car_boxes.items():
            for box_info in boxes:
                x_min, y_min, x_max, y_max = box_info['bbox']
                center_x = (x_min + x_max) / 2
                region_idx = int(center_x / image_width * num_front_subclasses)
                region_idx = min(region_idx, num_front_subclasses - 1)  # Prevent out of bounds
                region_counts[region_idx] += 1

        total = sum(region_counts)
        if total == 0:
            return "none", 2  # No car detected

        dominant_idx = region_counts.index(max(region_counts))

        # If it's the middle region, call it "front", others as "front_0", etc.
        subclass_names = [f"front_{i}" for i in range(num_front_subclasses)]
        mid_idx = num_front_subclasses // 2
        subclass_names[mid_idx] = "front"

        subclass_name = subclass_names[dominant_idx]
        if subclass_name == "front":
            class_id = 0
        else:
            if dominant_idx < mid_idx:
                class_id = 4 + dominant_idx
            else:
                class_id = 4 + dominant_idx - 1

        return subclass_name, class_id


    def process_folder(self, folder_path: str):
        if self.should_skip(folder_path):
            print(f"Skipping none directory: {folder_path}")
            return

        info = self.get_dataset_info(folder_path)
        if not info:
            print(f"Invalid path structure: {folder_path}")
            return

        try:
            record = self.datalog[
                (self.datalog['env'] == info['env']) &
                (self.datalog['id'] == info['id'])
                ]
            if record.empty:
                print(f"No DataLog record: {info}")
                return
            t0_frame = int(record['t0_frame'].values[0])
            class_label = str(record['direction'].values[0])
        except Exception as e:
            print(f"Data query error: {str(e)}")
            return

        self.clear_results(folder_path)
        result_dir = os.path.join(folder_path, 'result')
        os.makedirs(result_dir, exist_ok=True)

        audio_path = os.path.join(folder_path, 'out_multi.wav')
        try:
            with sf.SoundFile(audio_path) as f:
                duration_sec = float(f.frames / f.samplerate)
        except Exception as e:
            print(f"Audio file error: {str(e)}")
            return

        try:
            t0_sec = t0_frame / self.video_rate
            if info['env'].startswith('D'):
                start = max(0.0, t0_sec - 1.5)
                end = t0_sec - self.window_length / 1000
                windows = [
                    (t, t + self.window_length / 1000)
                    for t in np.arange(start, end, self.step_post_t0)
                    if t + self.window_length / 1000 <= t0_sec
                ]
            else:
                windows = self.generate_windows(duration_sec, t0_sec)
        except Exception as e:
            print(f"Time window generation failed: {str(e)}")
            return

        for idx, (start_sec, end_sec) in enumerate(tqdm(windows, desc=f"Processing {info['env']}-{info['id']}")):
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
                metadata.update({
                    "class_id": 1 if class_label.lower() == 'left' else 3,
                    "class_name": class_label.lower(),
                })
            else:
                try:
                    json_path = os.path.join(folder_path, 'camera_baseline_detections.json')
                    car_boxes = self.extract_car_boxes(json_path, start_sec, end_sec)
                    if not os.path.exists(json_path):
                        print(f"Missing detection file: {json_path}")
                        continue

                    if end_sec <= t0_sec - 1:
                        class_id, class_name = 2, 'none'
                    elif start_sec >= t0_sec - self.mid_window_length/1000 and end_sec <= t0_sec:
                        class_id = 1 if class_label.lower() == 'left' else 3
                        class_name = class_label.lower()
                    elif start_sec < t0_sec + 1.5:
                        continue
                    else:
                        if self.num_class == 4:
                            class_id, class_name = 0, 'front'
                        else:
                            match info['env']:
                                case 'SA1':
                                    image_width = 1264
                                case 'SA2':
                                    image_width = 1679
                                case 'SB1':
                                    image_width = 1285
                                case 'SB2':
                                    image_width = 1591
                                case 'SB3':
                                   image_width = 1407
                                case _:
                                    print("Unknown env")
                                    raise ValueError(f"Unknown env: {info['env']}")

                            print(image_width)
                            class_name, class_id = self.determine_front_subclass(car_boxes=car_boxes, image_width=image_width, num_class=self.num_class)

                    metadata.update({
                        "class_id": class_id,
                        "class_name": class_name,
                        "time_period": "pre-t0" if end_sec <= t0_sec -1 else "around-t0" if end_sec <= t0_sec else "post-t0",
                        "car_boxes": car_boxes
                    })
                except Exception as e:
                    print(f"Classification processing failed: {str(e)}")
                    continue

            filename = f"{info['id']}_{metadata['class_id']}_{idx}"
            if self.cut_audio(audio_path, start_sec, end_sec, os.path.join(result_dir, f"{filename}.wav")):
                try:
                    with open(os.path.join(result_dir, f"{filename}.json"), 'w') as f:
                        json.dump(metadata, f, indent=2, default=self.json_serializer)
                except Exception as e:
                    print(f"Metadata save failed: {str(e)}")

    def json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def cut_audio(self, audio_path: str, start_sec: float, end_sec: float, output_path: str) -> bool:
        try:
            with sf.SoundFile(audio_path) as f:
                sr = f.samplerate
                start_frame = int(start_sec * sr)
                end_frame = int(end_sec * sr)
                if start_frame < 0 or end_frame > f.frames:
                    print(f"Audio cut range out of bounds: {start_sec}-{end_sec}s (total duration: {f.frames/sr:.2f}s)")
                    return False
                f.seek(start_frame)
                data = f.read(end_frame - start_frame)
                sf.write(output_path, data, sr)
                return True
        except Exception as e:
            print(f"Audio processing failed: {str(e)}")
            return False


if __name__ == '__main__':
    PATH_ROOT = r'D:\ovad\ovad_dataset'  # Change to your dataset root directory

    processor = AudioProcessor(
        path_root=PATH_ROOT,
        audio_rate=48000,
        video_rate=10,
        window_length_ms=200,     # Each clip is 200ms
        step_pre_t0_ms=100,       # Pre-t0 window step 100ms
        step_post_t0_ms=100,      # T0-1s window step 100ms
        step_mid_t0_ms=100,       # Post-t0 window step 100ms
        image_width=1936
    )

    # Traverse the entire dataset directory
    for root, dirs, files in os.walk(PATH_ROOT):
        if processor.should_skip(root):
            continue
        if 'out_multi.wav' in files:
            print(f"\n{'=' * 40}")
            print(f"Processing directory: {root}")
            processor.process_folder(root)

    print("\nAll processing completed!")
