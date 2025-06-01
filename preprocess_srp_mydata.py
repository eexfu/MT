# This file is modified from the original code of the paper.
import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import scipy.io.wavfile as wavf
import pyroomacoustics as pra
from scipy import signal
import xml.etree.ElementTree as ET

ENV_MAP = {
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

def extract_from_wav_folder(folder_path, save_path, mic_array, resolution, freq_range, nfft, L,):
    extracted_data = []
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    for filename in tqdm(filenames, desc="Extracting features"):
        try:
            sample_rate, data = wavf.read(os.path.join(folder_path, filename))

            feature = extractSRPFeature(data, sample_rate, mic_array, resolution, freq_range, nfft, L)

            # Extract classid
            parts = filename.split('_')
            classid = int(parts[3])
            env_code = parts[1]
            environment = ENV_MAP.get(env_code, "Unknown")

            # Concatenate into a row: feature + class + filename
            row = np.concatenate([feature, np.array([classid, filename, environment], dtype=object)])
            extracted_data.append(row)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    if not extracted_data:
        print("No matching files found for the specified location list.")
        return

    # Create DataFrame
    feat_len = len(feature)
    columns = [f"feat{i}" for i in range(feat_len)] + ["Class", "filename", "Environment"]
    df = pd.DataFrame(extracted_data, columns=columns)

    # Save
    df.to_csv(save_path, index=False)
    print(f"Saved metadata to {save_path}")


def extractSRPFeature(dataIn, sampleRate, micArray, resolution, freqRange=[10,1200], nfft=2*256, L=2):
    doaProcessor = pra.doa.algorithms['SRP'](micArray.transpose(), sampleRate, nfft, azimuth=np.linspace(-90.,90., resolution)*np.pi/180, max_four=4)

    container = []
    for i in range(dataIn.shape[1]):
        _, _, stft = signal.stft(dataIn[:,i], sampleRate, nperseg=nfft)
        container.append(stft)
    container = np.stack(container)

    segments = []
    delta_t = container.shape[-1] // L
    for i in range(L):
        segments.append(container[:, :, i*delta_t:(i+1)*delta_t])

    feature = []
    for i in range(L):
        doaProcessor.locate_sources(segments[i], freq_range=freqRange)
        feature.append(doaProcessor.grid.values)

    return np.concatenate(feature)


def loadMicarray():
    ar_x = []
    ar_y = []
    root = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/config/ourmicarray_56.xml').getroot()
    for type_tag in root.findall('pos'):
        ar_x.append(type_tag.get('x'))
        ar_y.append(type_tag.get('y'))

    micArray = np.zeros([len(ar_x), 3])
    micArray[:,1] = ar_x
    micArray[:,2] = ar_y
    return micArray


if __name__ == '__main__':
    # L_list = [1, 2, 4, 8, 16]
    # resolution_list = [30, 60, 120, 240, 360]
    # freq_range_list = [[20, 50], [50, 1500], [1500, 3000]]

    L_list = [2]
    resolution_list = [240]
    freq_range_list = [[50, 1500]]

    mic_array = loadMicarray()
    nfft = 512

    for L in L_list:
        for resolution in resolution_list:
            for freq_range in freq_range_list:
                fmin, fmax = freq_range
                print(f"Processing with L = {L}, resolution = {resolution}, freq_range = {fmin}-{fmax}...")

                extract_from_wav_folder(
                    folder_path=r"D:\ovad\dataset_100ms",
                    save_path=rf".\preprocess_mydata\metadata_samples_L{L}_r{resolution}_f{fmin}-{fmax}.csv",
                    mic_array=mic_array,
                    resolution=resolution,
                    freq_range=freq_range,
                    nfft=nfft,
                    L=L
                )

                print(f"Finish: L = {L}, resolution = {resolution}, freq_range = {fmin}-{fmax}")