import xml.etree.ElementTree as ET

import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wavf
from scipy import signal


def loadMicarray(xml_path="config/ourmicarray_56.xml"):
    """
    解析ourmicarray_56.xml文件，读取麦克风位置。
    :param xml_path: XML文件路径
    :return: 麦克风位置数组 (shape: [num_mics, 3])
    """
    ar_x = []
    ar_y = []

    # 读取XML文件
    root = ET.parse(xml_path).getroot()
    for type_tag in root.findall('pos'):
        ar_x.append(float(type_tag.get('x')))
        ar_y.append(float(type_tag.get('y')))

    # 构建麦克风阵列坐标 (X, Y, Z)
    micArray = np.zeros([len(ar_x), 3])
    micArray[:, 0] = ar_x  # X坐标
    micArray[:, 1] = ar_y  # Y坐标
    micArray[:, 2] = 0.0   # Z全部假设为0（2D平面）

    return micArray

def compress_features_to_n_bins(features_360, target_bins=30):
    """
    把360维的SRP-PHAT特征压缩到指定维度（如30或60）
    :param features_360: (num_seconds, 360)的原始特征
    :param target_bins: 目标维度（默认30）
    :return: (num_seconds, target_bins)的压缩特征
    """
    assert features_360.shape[1] == 360, "输入特征不是360维"

    compression_ratio = 360 // target_bins
    features_compressed = np.zeros((features_360.shape[0], target_bins))

    for i in range(target_bins):
        start = i * compression_ratio
        end = (i + 1) * compression_ratio
        features_compressed[:, i] = features_360[:, start:end].mean(axis=1)

    return features_compressed

def extract_srp_phat_features(audio_path, xml_path="config/ourmicarray_56.xml", nfft=512, c=343.0, L=2, resolution=30):
    """
    每1秒提取L个SRP-PHAT特征向量，并压缩到resolution维
    """

    # 读取麦克风阵列
    micArray = loadMicarray(xml_path)

    # 读取音频
    fs_file, wavdata = wavf.read(audio_path)
    num_mics = wavdata.shape[1]
    total_length = wavdata.shape[0]  # 总帧数

    # 初始化SRP-PHAT的DoA分析器
    doa = pra.doa.algorithms['SRP'](micArray.T, fs_file, nfft, c=c)

    # 计算每秒的样本数
    total_samples = wavdata.shape[0]

    # **将1秒数据分成L个子窗口**
    samples_per_segment = total_samples // L

    # **存储 L 个窗口的 360 维特征**
    features_360 = np.zeros((L, doa.grid.n_points))

    for i in range(L):
        # **取出当前窗口的所有采样点**
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = wavdata[start:end, :]

        # **每个麦克风STFT**
        Zxx_list = []
        for m in range(num_mics):
            f, t, Zxx_m = signal.stft(segment[:, m], fs=fs_file, window='hann', nperseg=nfft, noverlap=nfft//2, boundary=None)
            Zxx_list.append(Zxx_m)

        Zxx = np.array(Zxx_list)  # shape (num_mics, num_freq_bins, num_time_frames)

        # **对当前窗口计算 SRP-PHAT**
        doa.locate_sources(Zxx)

        # **存储当前窗口的 360 维特征**
        features_360[i, :] = doa.grid.values

    # **压缩到 (L, resolution) 维度**
    features_compressed = compress_features_to_n_bins(features_360, resolution)

    return features_compressed


if __name__ == '__main__':
    audio_file = "C:/Projects/occluded_vehicle_acoustic_detection/ovad_example/left.wav"
    xml_file = "C:/Projects/occluded_vehicle_acoustic_detection/config/ourmicarray_56.xml"

    # 设置目标方向bins数（可以改为60或其他）
    target_bins = 30

    features = extract_srp_phat_features(audio_file, xml_file, resolution=target_bins)

    print(f"提取的特征形状：{features.shape}")  # (num_seconds, target_bins)
