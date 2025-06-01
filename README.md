# MT - Acoustic Signal Processing and Classification Project

This project is a modified version based on the master thesis repository [occluded_vehicle_acoustic_detection](https://github.com/tudelft-iv/occluded_vehicle_acoustic_detection.git) from TU Delft. The original dataset can be found in that repository.

This repository contains the code for my master thesis project, which focuses on acoustic signal processing and classification using microphone arrays. The project extends the original work by implementing additional features and improvements.

This is a Python project for acoustic signal processing and classification, primarily used for processing and analyzing acoustic data collected from microphone arrays.

## Main Features

- Acoustic feature extraction
- Data preprocessing and augmentation
- SVM model training
- CNN model training
- CNN model training for fine-grained class

## Usage

### 1. Feature Extraction from samples
```bash
# Basic feature extraction
python training_samples.py --extract_feats --input <input_data_path> --save_path <save_path>

# Feature extraction with different microphone configurations
python training_samples.py --num_mics 56 32 16 8 4 --extract_feats --input <input_data_path> --save_path <save_path> --L 2 4 8 16 --res 30 60 120 240 360 --fmin 20 50 1500 --fmax 50 1500 3000

# Feature extraction with specific geometric configuration
python training_samples.py --geo col0 col3 row4 row7 --extract_feats --input <input_data_path> --save_path <save_path> --L 2 4 8 16 --res 30 60 120 240 360 --fmin 20 50 1500 --fmax 50 1500 3000
```

### 2. SVM model training
```bash
python training_samples.py --run_cross_val --locs_list SAB SA SB DAB DA DB --root <preprocessed_data_path>
```

### 3. CNN model training
```bash
python training_samples.py --run_measure_CNN --locs_list SAB SA SB DAB DA DB --root <preprocessed_data_path>
```

### 4.1 extract fine-grained class from recordings
Run extract_mydata.py

### 4.2 CNN model training for fine-grained class
```bash
python training.py --locs_list SAB SA SB --root <preprocessed_data_path>
```

## Parameter Description

- `--num_mics`: Microphone quantity configuration
- `--geo`: Microphone array geometric configuration
- `--L`: Time segmentation parameter
- `--res`: Resolution parameter
- `--fmin/fmax`: Frequency range parameters
- `--locs_list`: Location list parameter

## Dependencies

- Python 3.12.0