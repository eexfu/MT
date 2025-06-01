# This file is modified from the original code of the paper.
import os
import warnings

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import metrics
import pickle
import json

import numpy as np
import xml.etree.ElementTree as ET
import pyroomacoustics as pra
import pandas as pd
import scipy.io.wavfile as wavf
import sklearn as skl

from scipy import signal
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import pdb

from model import CNNClassifier


class AudioData:

    def __init__(self, res=30, freq_range=[50,1500], nfft=2*256, L=2, window_length=1000, mic_array=None, data_df_save_path='./config/extracted_features.csv'):

        # set parameters for feature extraction
        self.mic_array = mic_array

        if mic_array is None:
            self.mic_array = loadMicarray()

        self.resolution = res
        self.freq_range = freq_range 
        self.nfft = nfft
        self.L = L
        self.window_length = window_length

        self.data = None
        self.data_df_save_path = data_df_save_path

    def random_chose_mic(self, num, save_path, filename):
        global selected_indices
        """
        Randomly select num microphones and update self.mic_array
        """
        if num > self.mic_array.shape[0]:
            raise ValueError(f"Requested number of microphones {num} exceeds available microphones {self.mic_array.shape[0]}")
        random_indices = np.random.choice(self.mic_array.shape[0], num, replace=False)
        selected_indices = np.sort(random_indices)
        self.mic_array = self.mic_array[selected_indices]
        print(selected_indices)
        print(f"Randomly selected {num} microphones, updated microphone array:")
        print(self.mic_array)
        self.plot_mic_array(save_path, filename)


    def select_geometry(self, shape_name, save_path, filename):
        """
        Select a subset of microphones with specific geometry.

        Args:
            shape_name: str, geometry name, e.g. "circle1", "cross", "x_shape", "center_block"
        """

        global selected_indices

        # Define preset geometries (can be extended as needed)
        shapes = {
            "col0": np.array([0, 4, 44, 42]),
            "col1": np.array([1, 2, 5, 45, 43]),
            "col2": np.array([3, 6, 40, 41, 46]),
            "col3": np.array([7, 39, 47, 48]),
            "col4": np.array([8, 9, 10, 38, 55]),
            "col5": np.array([11, 12, 37, 49, 54]),
            "col6": np.array([13, 15, 33, 36, 52]),
            "col7": np.array([14, 17, 35, 50, 53]),
            "col8": np.array([16, 18, 22, 23, 32, 34, 51]),
            "col9": np.array([19, 20, 24, 26, 27, 29, 31]),
            "col10":  np.array([21, 25, 28, 30]),
            "row0": np.array([0, 1, 8, 14, 18, 19]),
            "row1": np.array([2, 3, 9, 15, 16, 20, 21]),
            "row2": np.array([4, 6, 7, 10, 11, 17, 22]),
            "row3": np.array([5, 12, 13, 23, 24]),
            "row4": np.array([25, 46, 48, 49, 50]),
            "row5": np.array([26, 44, 51, 52, 54]),
            "row6": np.array([27, 45, 47, 53, 55]),
            "row7": np.array([28, 29, 32, 33, 38, 39, 40]),
            "row8": np.array([34, 36, 42, 43]),
            "row9": np.array([30, 31, 35, 37, 41]),

            "rhombus": np.array([8, 24, 25, 37, 44, 45]),

            "center_cross": np.array([1, 12, 13, 18, 31, 41]),

            "rectangle": np.array([0, 15, 21, 30, 37, 42]),

            "rectangle1": np.array([0, 9, 21, 30, 35, 42]),

            "diagonal": np.array([0, 6, 29, 30, 49, 53]),

            "diagonal1": np.array([21, 22, 40, 42, 50, 55]),
        }

        if shape_name not in shapes:
            raise ValueError("Shape not found")

        selected_indices = shapes[shape_name]
        self.mic_array = self.mic_array[selected_indices]

        print(f"Selected shape: '{shape_name}'")
        print(f"Microphone indices: {selected_indices}")
        self.plot_mic_array(save_path, filename)

    def plot_mic_array(self, save_path, filename):
        global all_mic_array
        plt.figure(figsize=(8, 6))
        all_mic_array = loadMicarray()
        # Plot all microphones
        plt.scatter(all_mic_array[:, 1], all_mic_array[:, 2], c='blue', marker='o', label="All Microphones")

        # Label all microphone numbers
        for i in range(len(all_mic_array)):
            plt.text(all_mic_array[i, 1], all_mic_array[i, 2], str(i + 1), fontsize=9, ha='right', va='bottom')

        # Plot selected microphones (in red)
        if selected_indices is not None:
            plt.scatter(all_mic_array[selected_indices, 1], all_mic_array[selected_indices, 2], c='red', marker='o',
                        label="Selected Microphones")

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Microphone Array Layout")
        plt.grid(True)
        plt.legend()
        # Create directory (if it doesn't exist)
        os.makedirs(save_path, exist_ok=True)

        # Construct full save path
        full_path = os.path.join(save_path, filename)

        # Save image as file
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Mic array plot saved to {full_path}")

        plt.close()

    def extract_data(self, data_path, save=False):
        label_data = pd.read_csv(os.path.join(data_path, 'SampleLog.csv'))

        # Move Class column to the front
        label_data = label_data[["Class"] + [col for col in label_data.columns if col != "Class"]]

        data_columns = None
        rows_list = []

        num_samples = dict.fromkeys(["front", "left", "none", "right"], 0)

        for idx, row in tqdm(label_data.iterrows(), desc='Extracting features: ', total=label_data.shape[0]):
            sample_path = os.path.join(data_path, row["Class"], row["ID"] + '.wav')
            sample_rate, data = wavf.read(sample_path)
            selected_data = data[:, selected_indices]  # Channel selection

            window_len = self.window_length
            window_samples = int(sample_rate * (window_len / 1000.0))
            total_samples = selected_data.shape[0]
            num_windows = total_samples // window_samples

            for w in range(num_windows):
                start = w * window_samples
                end = start + window_samples
                window_data = selected_data[start:end]
                feature = extractSRPFeature(window_data, sample_rate, self.mic_array,
                                            self.resolution, self.freq_range, self.nfft, self.L)

                if data_columns is None:
                    data_columns = ['feat' + str(i) for i in range(feature.shape[0])] + list(label_data.columns) + ['window_id']

                row_values = np.concatenate((feature, row.to_numpy(), [w]))
                rows_list.append(row_values)
                num_samples[row["Class"]] += 1

        # Create DataFrame at once (faster)
        extracted_data = pd.DataFrame(rows_list, columns=data_columns)
        self.data = extracted_data

        if save:
            extracted_data.to_csv(self.data_df_save_path, index=False)


    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def read_csv(self, csv_path=None):
        if csv_path is None:
            self.data = pd.read_csv(self.data_df_save_path)
        else:
            print(" --- Please ensure csv is of the format given by the file: {} --- ".format(self.data_df_save_path))
            self.data = pd.read_csv(csv_path)


def get_locations(locs_in=["SAB"]):

    temp_locs_in = locs_in.copy()

    loc_ids = {"type_A": ["A1", "A2"],  
               "type_B": ["B1", "B2", "B3"]}

    for loc in temp_locs_in:
        if loc == "SAB" or loc == "DAB":
            temp_ids = loc_ids["type_A"] + loc_ids["type_B"]
            locs_in.remove(loc)
            locs_in.extend([loc[0] + id for id in temp_ids])

        elif loc == "DA" or loc == "SA":
            temp_ids = loc_ids["type_A"]
            locs_in.remove(loc)
            locs_in.extend([loc[0] + id for id in temp_ids])
            
        elif loc == "DB" or loc == "SB":
            temp_ids = loc_ids["type_B"]
            locs_in.remove(loc)
            locs_in.extend([loc[0] + id for id in temp_ids])

    return list(set(locs_in))


def prepare_skl_interface(data_in, classifier):

    #prepare the sklearn interface
    le = skl.preprocessing.LabelEncoder()
    le.fit(data_in["Class"].unique())
    scaler = skl.preprocessing.StandardScaler()
    pipeline = Pipeline([('transformer', scaler), ('estimator', classifier)])
    
    return le, pipeline


def prepare_skl_interface_CNN(data_in, classifier):

    #prepare the sklearn interface
    le = skl.preprocessing.LabelEncoder()
    le.fit(data_in["Class"].unique())
    pipeline = classifier  # CNN is not suitable to use with StandardScaler

    return le, pipeline


def partitionPanda(vector, fold, k):
    size = vector.shape[0]
    start = (size//k)*fold
    end = (size//k)*(fold+1)
    validation = vector.iloc[start:end,:]

    indices = range(start, end)
    mask = np.ones(vector.shape[0], dtype=bool)
    mask[indices] = False
    training = vector.iloc[mask,:]

    return training, validation


def cross_validation(pipeline, n_folds, data, le, srp_dict, data_aug=True):

    un_classes = data["Class"].unique()   # get classes
    
    # initialize variables that hold metrics
    per_class_cm = []              # Per class confusion matrix
    per_class_acc = []             # Per class accuracy
    per_class_prec = []            # Per class precision
    per_class_rec = []             # Per class recall
    per_class_iou = []             # Per class IoU
    validation_folds_score = []    # Overall accuracy on Validation folds 
    CO = None                      # Confusion matrix summed over folds           


    # iterate over the folds
    for fold in range(0, n_folds):
        training_set = pd.DataFrame(columns=data.columns)
        validation_set = pd.DataFrame(columns=data.columns)

        # exception for the loo case
        if n_folds == data.shape[0]:
            training_set, validation_set = partitionPanda(data, fold, n_folds)
        else:
            # otherwise make sure that classes are equally distributed
            train_list = []
            val_list = []

            for single_label in un_classes:
                df_sl = data[data["Class"] == single_label].reset_index(drop=True)
                train_snippet, validation_snippet = partitionPanda(df_sl, fold, n_folds)
                train_list.append(train_snippet)
                val_list.append(validation_snippet)

            training_set = pd.concat(train_list, ignore_index=True)
            validation_set = pd.concat(val_list, ignore_index=True)

        # train classifier and get predictions on validation
        accuracy, C = train_and_test(training_set, validation_set, pipeline, le, srp_dict, data_aug=data_aug)

        # aggregate the metrics
        validation_folds_score.append(accuracy)

        if CO is None:
            CO = C
        else:
            CO = CO + C

        per_class_acc.extend([metrics.getPCaccuracy(C)])
        per_class_prec.extend([metrics.getPCPrecision(C)])
        per_class_rec.extend([metrics.getPCRecall(C)])
        per_class_iou.extend([metrics.getPCIoU(C)])

    metrics_dict = {"overall_accuracy" : (np.mean(validation_folds_score), np.std(validation_folds_score)), 
                    "per_class_accuracy": (np.mean(per_class_acc, axis=0), np.std(per_class_acc, axis=0)), 
                    "per_class_precision": (np.mean(per_class_prec, axis=0), np.std(per_class_prec, axis=0)), 
                    "per_class_recall": (np.mean(per_class_rec, axis=0), np.std(per_class_rec, axis=0)), 
                    "per_class_iou": (np.mean(per_class_iou, axis=0), np.std(per_class_iou, axis=0)),
                    "conf_mat": CO}


    return metrics_dict


def cross_validation_CNN(pipeline, n_folds, data, le, srp_dict, data_aug=True, L=2, res=240, num_classes=4):

    un_classes = data["Class"].unique()   # get classes

    # initialize variables that hold metrics
    per_class_acc = []             # Per class accuracy
    per_class_prec = []            # Per class precision
    per_class_rec = []             # Per class recall
    per_class_iou = []             # Per class IoU
    validation_folds_score = []    # Overall accuracy on Validation folds
    CO = None                      # Confusion matrix summed over folds


    # iterate over the folds
    for fold in range(0, n_folds):
        train_list = []
        val_list = []

        # exception for the loo case
        if n_folds == data.shape[0]:
            training_set, validation_set = partitionPanda(data, fold, n_folds)
        else:
            # otherwise make sure that classes are equally distributed
            for single_label in un_classes:
                df_sl = data[data["Class"] == single_label]
                df_sl = df_sl.reset_index(drop=True)
                train_snippet, validation_snippet = partitionPanda(df_sl, fold, n_folds)
                # training_set = training_set.append(train_snippet, ignore_index=True)
                # validation_set = validation_set.append(validation_snippet, ignore_index=True)
                if not train_snippet.empty:
                    train_list.append(train_snippet)
                if not validation_snippet.empty:
                    val_list.append(validation_snippet)

            training_set = pd.concat(train_list, ignore_index=True)
            validation_set = pd.concat(val_list, ignore_index=True)

        # Reinitialize CNNClassifier for each fold
        pipeline = CNNClassifier(
            input_shape=(L, res),
            num_classes=num_classes,
            lr=1e-2,
            patience=30,
            optimizer_type='adamw'
        )

        # train classifier and get predictions on validation
        accuracy, C = train_and_test_CNN(training_set, validation_set, pipeline, le, srp_dict, data_aug=data_aug, epochs=300)

        # aggregate the metrics
        validation_folds_score.append(accuracy)

        if CO is None:
            CO = C
        else:
            CO = CO + C

        per_class_acc.extend([metrics.getPCaccuracy(C)])
        per_class_prec.extend([metrics.getPCPrecision(C)])
        per_class_rec.extend([metrics.getPCRecall(C)])
        per_class_iou.extend([metrics.getPCIoU(C)])

    metrics_dict = {"overall_accuracy" : (np.mean(validation_folds_score), np.std(validation_folds_score)),
                    "per_class_accuracy": (np.mean(per_class_acc, axis=0), np.std(per_class_acc, axis=0)),
                    "per_class_precision": (np.mean(per_class_prec, axis=0), np.std(per_class_prec, axis=0)),
                    "per_class_recall": (np.mean(per_class_rec, axis=0), np.std(per_class_rec, axis=0)),
                    "per_class_iou": (np.mean(per_class_iou, axis=0), np.std(per_class_iou, axis=0)),
                    "conf_mat": CO.tolist()}


    return metrics_dict


def do_data_augmentation(data_in, res, nsegs):
    """
    Perform mirror flip augmentation for left and right classes, with symmetric label flipping.

    Args:
        data_in: pd.DataFrame, original data (features + Class + other info)
        res: int, feature length per segment (resolution)
        nsegs: int, number of segments (time windows)

    Returns:
        data_out: pd.DataFrame, containing original + augmented data
    """
    columns = data_in.columns
    data_out = data_in.copy()
    right = data_in[data_in["Class"] == 3]
    left = data_in[data_in["Class"] == 1]
    # Collect new samples in lists
    augmented_right_rows = []
    augmented_left_rows = []

    # Flip right → become left
    for _, row in right.iterrows():
        flipped_segments = []
        for i in range(nsegs):
            segment = row[i * res:(i + 1) * res]
            flipped_segments.append(np.flip(segment.to_numpy(), 0))
        flipped_feature = np.concatenate(flipped_segments)

        label = np.array([1])
        suffix = row[nsegs * res + 1:].to_numpy()  # Skip the class label column
        new_row = np.concatenate((flipped_feature, label, suffix))
        augmented_right_rows.append(new_row)

    # Flip left → become right
    for _, row in left.iterrows():
        flipped_segments = []
        for i in range(nsegs):
            segment = row[i * res:(i + 1) * res]
            flipped_segments.append(np.flip(segment.to_numpy(), 0))
        flipped_feature = np.concatenate(flipped_segments)

        label = np.array([3])
        suffix = row[nsegs * res + 1:].to_numpy()
        new_row = np.concatenate((flipped_feature, label, suffix))
        augmented_left_rows.append(new_row)

    # Build new DataFrame (if there are augmented samples)
    if augmented_right_rows:
        df_right = pd.DataFrame(augmented_right_rows, columns=columns)
        data_out = pd.concat([data_out, df_right], ignore_index=True)

    if augmented_left_rows:
        df_left = pd.DataFrame(augmented_left_rows, columns=columns)
        data_out = pd.concat([data_out, df_left], ignore_index=True)

    return data_out


def train_and_test(train_set, test_set, pipeline, le, srp_dict=None, save_cls=False, out_folder=None, data_aug=True):

    # do flip based data augmentation
    if data_aug:
        if srp_dict is not None:
            train_set = do_data_augmentation(train_set, srp_dict['res'], srp_dict['nsegs'])
    
    # check until which column features are stored
    i_max = 1
    for i, col in enumerate(train_set.columns):
        if 'feat' in col:
            i_max = i + 1
    
    # split the dataframe to get features and append the transformed labels
    data_train = np.split(train_set.to_numpy(), [i_max], axis=1)
    data_train[1] = le.transform(train_set["Class"])  

    data_test = np.split(test_set.to_numpy(), [i_max], axis=1)
    data_test[1] = le.transform(test_set["Class"])

    # fit the classifier and predict on the test set
    pipeline.fit(data_train[0], data_train[1])
    test_predicted = pipeline.predict(data_test[0])
    svm_model = pipeline.named_steps["estimator"]
    print("Support vector count:", svm_model.support_vectors_.shape[0])
    print("Total parameters (approx.):", svm_model.support_vectors_.size)

    accuracy_score = skl.metrics.accuracy_score(data_test[1], test_predicted)

    # extract confusion matrix and metrics
    conf_mat = skl.metrics.confusion_matrix(data_test[1], test_predicted, labels=le.transform(le.classes_))

    if save_cls:
        if out_folder is None:
            save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_classifier')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join(out_folder, 'saved_classifier/')
            os.makedirs(save_dir, exist_ok=True)

        print("Saving Classifier to {} ... ".format(save_dir))

        locs_in_train = train_set["Environment"].unique()
        save_string = "_".join(locs_in_train)

        pickle.dump((pipeline), open(os.path.join(*[save_dir, save_string + '_classifier.obj']), "wb"))
        test_set = test_set.drop_duplicates(subset=["Recording ID"])
        test_set["ID"].to_csv(os.path.join(*[save_dir, save_string + '_test_bags.csv']), index=False, header=True)

    return accuracy_score, conf_mat


def train_and_test_CNN(train_set, test_set, pipeline, le, srp_dict=None, save_cls=False, out_folder=None, data_aug=True, epochs=200):

    # do flip based data augmentation
    if data_aug:
        if srp_dict is not None:
            train_set = do_data_augmentation(train_set, srp_dict['res'], srp_dict['nsegs'])

    # check until which column features are stored
    i_max = 1
    for i, col in enumerate(train_set.columns):
        if 'feat' in col:
            i_max = i + 1

    # Split train_set into train/val
    train_df, val_df = train_test_split(
        train_set, test_size=0.2, stratify=train_set["Class"], random_state=42
    )

    # Split features and labels
    def process_data(df):
        data = np.split(df.to_numpy(), [i_max], axis=1)
        data[1] = le.transform(df["Class"])
        return data

    train_data = process_data(train_df)
    val_data = process_data(val_df)
    test_data = process_data(test_set)

    X_train = train_data[0]
    X_val   = val_data[0]
    X_test  = test_data[0]

    pipeline.train_model(X_train, train_data[1], X_val, val_data[1], epochs=epochs)
    test_predicted = pipeline.predict(X_test)

    accuracy_score = skl.metrics.accuracy_score(test_data[1], test_predicted)

    # extract confusion matrix and metrics
    conf_mat = skl.metrics.confusion_matrix(test_data[1], test_predicted, labels=le.transform(le.classes_))

    if save_cls:
        if out_folder is None:
            save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_classifier')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join(out_folder, 'saved_classifier/')
            os.makedirs(save_dir, exist_ok=True)

        print("Saving Classifier to {} ... ".format(save_dir))

        locs_in_train = train_set["Environment"].unique()
        save_string = "_".join(locs_in_train)

        pickle.dump((pipeline), open(os.path.join(*[save_dir, save_string + '_classifier.obj']), "wb"))
        test_set = test_set.drop_duplicates(subset=["Recording ID"])
        test_set["ID"].to_csv(os.path.join(*[save_dir, save_string + '_test_bags.csv']), index=False, header=True)

    return accuracy_score, conf_mat


def makeDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loadMicarray():
    ar_x = []
    ar_y = []
    
    # iterate through the xml to get all locations
    root = ET.parse(os.path.dirname(os.path.abspath(__file__)) + '/config/ourmicarray_56.xml').getroot()
    for type_tag in root.findall('pos'):
        ar_x.append(type_tag.get('x'))
        ar_y.append(type_tag.get('y'))

    # set up the array vector
    micArray = np.zeros([len(ar_x), 3])
    micArray[:,1] = ar_x
    micArray[:,2] = ar_y

    micArrayConfig = r"""
  _______________________________________________________________
   Loading microphone Array with {} microphones.  
                                            -O  |
                                -O              |
                    -O                          |
        -O               |Z                     |            ┌ ┐
                         |    _Y            -O  |            |X|
                         |___/  -O              | micArray = |Y|
                    -O    \                     |            |Z|
        -O                 \X                   |            └ ┘
                                            -O  |
                                -O              |
                    -O                          |
        -O                                      | 
  _______________________________________________________________\n\n
        """.format(micArray.shape[0])

    # print(micArrayConfig)

    return micArray


def extractSRPFeature(dataIn, sampleRate, micArray, resolution, freqRange=[10,1200], nfft=2*256, L=2):
    # generate fft lengths and filter mics and create doa algorithm
    doaProcessor = pra.doa.algorithms['SRP'](micArray.transpose(), sampleRate, nfft, azimuth=np.linspace(-90.,90., resolution)*np.pi/180, max_four=4)
    
    # extract the stft from parameters
    container = []
    for i in range(dataIn.shape[1]):
        _, _, stft = signal.stft(dataIn[:,i], sampleRate, nperseg=nfft)
        container.append(stft)
    container = np.stack(container)
    
    # split the stft into L segments
    segments = []
    delta_t = container.shape[-1] // L 
    for i in range(L):
        segments.append(container[:, :, i*delta_t:(i+1)*delta_t])
    # pdb.set_trace()
    # container = [container[:, :, 0:94], container[:, :, 94:94+94]]

    # apply the doa algorithm for each specified segment according to parameters
    feature = []
    for i in range(L):
        doaProcessor.locate_sources(segments[i], freq_range=freqRange)
        feature.append(doaProcessor.grid.values)

    return np.concatenate(feature)


def detectionFolder(folder, score_threshold=0, height_threshold=0):
    """
    score_threshold : range 0-1.0
    """
    detection_fpath = os.path.join(folder, "camera_baseline_detections.json")
    if not os.path.isfile(detection_fpath):
        raise ValueError("No file {} found.")

    with open("{}/camera_baseline_detections.json".format(folder), 'r') as f:
        detection_summary = json.load(f)
    detections_per_frame = detection_summary['detections_per_frame']

    filtered_detections_per_frame = []
    for detections in detections_per_frame:
        filter_detections = [(box, score)
                             for box, score, class_str
                             in zip(detections['boxes'], detections['scores'],
                                    detections['classes_str'])
                             if ((class_str == 'car' or class_str == 'motorcycle')
                                 and score >= score_threshold
                                 and np.abs(box[1]-box[3]) > height_threshold)]
        filtered_detections_per_frame.append(filter_detections)

    return filtered_detections_per_frame


