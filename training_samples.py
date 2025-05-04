import os
import re
import sys
import argparse
import utilities
import metrics

import sklearn as skl
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from model import CNNClassifier

class Expts:

    def __init__(self, data, output_folder, data_aug=True, classifier=None, random_state=0):

        self.classifier = classifier
        self.data = data
        self.random_state = random_state   #set this to None for randomizing the data shuffle and fold split.
        self.paper_metrics_only = True
        self.output_folder = output_folder
        self.data_aug = data_aug

        #feature related parameters
        self.srp_dict = dict(res=data.resolution, nsegs=data.L)

        #for cross validation expt.
        self.folds = 5

        if classifier is None:
            self.classifier = SVC(kernel='linear', C=1, random_state=self.random_state)


    def run_train_and_save_classifier(self, save_classifier=True, locs_in=["SAB"], data_split_ratio=0.15):

        locations = utilities.get_locations(locs_in)

        print(" \n\n--- Train a classifier for locations: {}\n".format(locations))
        print("Data split into train and test set at ratio: {}\n".format(data_split_ratio))

        data_in = self.data.get_data()
        data_in = data_in[data_in["Environment"].isin(locations)]

        label_encoder, pipeline = utilities.prepare_skl_interface(data_in, self.classifier)

        # stratified split data into train and test - stratification considers location and the class labels

        temp_df = data_in[['Environment', 'Recording ID', 'Class']]
        temp_df = temp_df.drop(temp_df[temp_df['Class'] == 'front'].index)   # just to avoid repeated Recording IDs

        train_bags, test_bags = train_test_split(temp_df,
                                                 test_size=data_split_ratio,
                                                 random_state=self.random_state,
                                                 stratify=temp_df[['Environment', 'Class']])

        # check if samples from same recordings are present in both train and test
        for bag in list(test_bags['Recording ID']):
            if bag in list(train_bags['Recording ID']):
                print("Error: {}".format(bag))

        train_data = data_in[data_in['Recording ID'].isin(train_bags['Recording ID'])]
        test_data = data_in[data_in['Recording ID'].isin(test_bags['Recording ID'])]

        accuracy, conf_mat = utilities.train_and_test(train_data, test_data, pipeline, label_encoder, self.srp_dict,
                                                      save_cls=save_classifier, out_folder=self.output_folder)

        all_metrics = {"overall_accuracy" : (accuracy, 0),
                       "per_class_accuracy": (metrics.getPCaccuracy(conf_mat), np.zeros(4)),
                       "per_class_precision": (metrics.getPCPrecision(conf_mat), np.zeros(4)),
                       "per_class_recall": (metrics.getPCRecall(conf_mat), np.zeros(4)),
                       "per_class_iou": (metrics.getPCIoU(conf_mat), np.zeros(4))}

        metrics.print_metrics(all_metrics, self.paper_metrics_only)


    def run_cross_validation(self, locs_in=["SAB"]):

        print(" \n\n--- Cross Validation for locations: {}\n".format(locs_in))
        loc = locs_in[0]

        locations = utilities.get_locations(locs_in)

        data_in = self.data.get_data()
        data_in = data_in[data_in["Environment"].isin(locations)]

        label_encoder, pipeline = utilities.prepare_skl_interface(data_in, self.classifier)

        # shuffle with random seed if specified
        if self.random_state is not None:
            data_in_shuffled = skl.utils.shuffle(data_in, random_state=self.random_state)
        else:
            data_in_shuffled = skl.utils.shuffle(data_in)

        # get metrics
        output_metrics = utilities.cross_validation(pipeline, self.folds, data_in_shuffled, label_encoder, self.srp_dict, data_aug=self.data_aug)

        metrics.print_metrics(output_metrics, self.paper_metrics_only)
        return output_metrics

    def run_generalisation(self, train_locs=["DA"], test_locs=["DB"], save_classifier=True):

        print("\n\n --- Generalization across locations --- \n")
        print("Locations in train set: {}".format(train_locs))
        print("Locations in test set: {}".format(test_locs))

        train_locs = utilities.get_locations(train_locs)
        test_locs = utilities.get_locations(test_locs)

        data_in = self.data.get_data()
        label_encoder, pipeline = utilities.prepare_skl_interface(data_in, self.classifier)

        train_data = data_in[data_in["Environment"].isin(train_locs)]
        test_data  = data_in[data_in["Environment"].isin(test_locs)]

        accuracy, conf_mat = utilities.train_and_test(train_data, test_data, pipeline, label_encoder, self.srp_dict,  save_cls=save_classifier, out_folder=self.output_folder)

        all_metrics = {"overall_accuracy" : (accuracy, 0),
                       "per_class_accuracy": (metrics.getPCaccuracy(conf_mat), np.zeros(4)),
                       "per_class_precision": (metrics.getPCPrecision(conf_mat), np.zeros(4)),
                       "per_class_recall": (metrics.getPCRecall(conf_mat), np.zeros(4)),
                       "per_class_iou": (metrics.getPCIoU(conf_mat), np.zeros(4))}

        metrics.print_metrics(all_metrics, self.paper_metrics_only)

    def run_measure(self, locs_in=["SAB"]):

        print(" \n\n--- Cross Validation for locations: {}\n".format(locs_in))

        locations = utilities.get_locations(locs_in)

        data_in = self.data.get_data()
        data_in = data_in[data_in["Environment"].isin(locations)]

        label_encoder, pipeline = utilities.prepare_skl_interface(data_in, self.classifier)

        # shuffle with random seed if specified
        if self.random_state is not None:
            data_in_shuffled = skl.utils.shuffle(data_in, random_state=self.random_state)
        else:
            data_in_shuffled = skl.utils.shuffle(data_in)

        # get metrics
        output_metrics = utilities.cross_validation(pipeline, self.folds, data_in_shuffled, label_encoder, self.srp_dict, data_aug=self.data_aug)

        metrics.print_metrics(output_metrics, self.paper_metrics_only)

        return output_metrics


    def run_measure_CNN(self, locs_in=["SAB"], L=2, res=240, num_classes=4):

        print(" \n\n--- Cross Validation for locations: {}\n".format(locs_in))

        locations = utilities.get_locations(locs_in)

        data_in = self.data.get_data()
        data_in = data_in[data_in["Environment"].isin(locations)]

        label_encoder, pipeline = utilities.prepare_skl_interface_CNN(data_in, self.classifier)

        # shuffle with random seed if specified
        if self.random_state is not None:
            data_in_shuffled = skl.utils.shuffle(data_in, random_state=self.random_state)
        else:
            data_in_shuffled = skl.utils.shuffle(data_in)

        # get metrics
        output_metrics = utilities.cross_validation_CNN(pipeline, self.folds, data_in_shuffled, label_encoder, self.srp_dict, data_aug=self.data_aug, L=L, res=res, num_classes=num_classes)

        metrics.print_metrics(output_metrics, self.paper_metrics_only)

        return output_metrics


def parseArgs():
    class ExtractorArgsParser(argparse.ArgumentParser):
        def error(self, message):
            self.print_help()
            sys.exit(2)
        def format_help(self):
            formatter = self._get_formatter()
            formatter.add_text(self.description)
            formatter.add_usage(self.usage, self._actions,
                                self._mutually_exclusive_groups)
            for action_group in self._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.add_text(self.epilog)

            return formatter.format_help()

    usage = """
    To run cross validation experiment: 
        
        python classificationExpts.py --run_cross_val --locs_list DAB DA DB
    
    To run generalization experiment:

        python classificationExpts.py --run_gen --train_locs_list SA --test_locs_list SB --save_cls

    To train and save a classifier on a data subset:

        python classificationExpts.py --train_save_cls --locs_list SAB --split_ratio 0.15 --save_cls

    """
    parser = ExtractorArgsParser(description='Run classification experiments on the acoustic_data', usage=usage)

    parser.add_argument('--input',  dest='input',  default=None, help='Path to extracted samples')
    parser.add_argument('--output', dest='output', default=None, help='Output folder to store the saved classifier.')

    parser.add_argument('--num_mics', nargs="+", type=int, default=None, help='Number of microphones to randomly choose')
    parser.add_argument("--geo", nargs="+", type=int, default=None, help="the geometry of microphone array")

    parser.add_argument('--extract_feats', dest='extract_feats', action='store_true',
                        help='If specified, features are extracted rather than read from a csv file stored in ./config folder')
    parser.add_argument("--save_path", type=str, help="Directory to save extracted features")

    parser.add_argument('--run_cross_val', action='store_true', help='Runs the cross validation experiment. (Table 3 in the paper)')
    parser.add_argument('--locs_list', nargs="+", default=["DAB", "DA", "DB"], help='List of Location IDs to run cross validation / train and test a classifier. To specify multiple subsets, separate the arguments with a space e.g --locs_list SA1 SA2')

    parser.add_argument('--run_gen',  action='store_true', help='Runs the generalization experiment. (Table 4 in the paper)')
    parser.add_argument('--train_locs_list', nargs="+", default=["SA"], help='List of locations in the training set')
    parser.add_argument('--test_locs_list', nargs="+", default=["SB"], help='List of locations in the training set')

    parser.add_argument('--train_save_cls', action='store_true', help='Train a classifier on specified subset and optionally save it.')
    parser.add_argument('--split_ratio', default=0.15, type=float, help='Ratio to split the data into train and test')
    parser.add_argument('--save_cls', action='store_true', help='Save the trained classifier.')

    parser.add_argument('--run_measure', action='store_true', help='Train a classifier on dataset with different parameter')
    parser.add_argument('--run_measure_CNN', action='store_true', help='Train a CNN classifier on dataset with different parameter')
    parser.add_argument('--test_cuda', action='store_true', help='Test cuda')
    parser.add_argument('--root',  default=None, help='Path to samples')

    #hyperparameters
    parser.add_argument('--C',  default=1, type=float, help='Classifier regularization')
    parser.add_argument('--no_data_aug', action='store_true', help='Disable data augmentation')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--L',  nargs="+", type=int, default=2, help='Number of segments')
    parser.add_argument('--res', nargs="+", type=int, default=30, help='Resolution of segments')
    parser.add_argument('--fmin', nargs="+", type=int, default=50, help='Min Frequency')
    parser.add_argument('--fmax', nargs="+", type=int, default=1500, help='Max Frequency')
    parser.add_argument('--window_length', nargs="+", type=int, default=1000, help='The length of window(ms)')


    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(2)

    try:
        parsed = parser.parse_args()
    except:
        sys.exit(2)

    return parsed


if __name__ == '__main__':
    parsed = parseArgs()

    if parsed.num_mics is None and parsed.geo is None:
        parsed.num_mics = 56

    def to_int_list(x):
        if isinstance(x, list):
            return [int(i) for i in x]
        elif isinstance(x, str):
            return [int(x)]
        elif isinstance(x, int):
            return [x]
        else:
            raise ValueError(f"Unsupported type for argument: {type(x)}")

    parsed.L = to_int_list(parsed.L)
    parsed.res = to_int_list(parsed.res)
    parsed.fmin = to_int_list(parsed.fmin)
    parsed.fmax = to_int_list(parsed.fmax)
    parsed.window_length = to_int_list(parsed.window_length)
    if parsed.num_mics is not None:
        parsed.num_mics = to_int_list(parsed.num_mics)
    if parsed.geo is not None:
        parsed.geo = to_int_list(parsed.geo)

    if len(parsed.fmin) != len(parsed.fmax):
        raise ValueError("Please make suer the length of fmin and fmax are the same")

    if parsed.extract_feats:
        if parsed.input is None:
            raise ValueError("Please specify path to extracted one second audio samples at the flag --input.")

        os.makedirs(parsed.save_path, exist_ok=True)  # 确保输出目录存在

        if parsed.num_mics is not None:
            for i in range(len(parsed.L)):
                for j in range(len(parsed.res)):
                    for k in range(len(parsed.fmin)):
                        for l in range(len(parsed.window_length)):
                            for mics in parsed.num_mics:
                                L_val = parsed.L[i]
                                res_val = parsed.res[j]
                                fmin_val = parsed.fmin[k]
                                fmax_val = parsed.fmax[k]
                                window_length_val = parsed.window_length[l]

                                save_path = os.path.join(
                                    parsed.save_path,
                                    f"extracted_features_m{mics}_g0_L{L_val}_r{res_val}_f{fmin_val}-{fmax_val}_w{window_length_val}.csv"
                                )

                                data = utilities.AudioData(
                                    L=L_val,
                                    res=res_val,
                                    freq_range=[fmin_val, fmax_val],
                                    window_length=window_length_val,
                                    data_df_save_path=save_path
                                )
                                data.random_chose_mic(mics, save_path=parsed.save_path, filename=f"extracted_features_m{mics}_g0_L{L_val}_r{res_val}_f{fmin_val}-{fmax_val}.png")
                                data.extract_data(data_path=parsed.input, save=True)
        else:
            print("Didn't do microphone array.")

        if parsed.geo is not None:
            for i in range(len(parsed.L)):
                for j in range(len(parsed.res)):
                    for k in range(len(parsed.fmin)):
                        for l in range(len(parsed.window_length)):
                            for g in parsed.geo:
                                L_val = parsed.L[i]
                                res_val = parsed.res[j]
                                fmin_val = parsed.fmin[k]
                                fmax_val = parsed.fmax[k]
                                window_length_val = parsed.window_length[l]

                                save_path = os.path.join(
                                    parsed.save_path,
                                    f"extracted_features_m0_g{g}_L{L_val}_r{res_val}_f{fmin_val}-{fmax_val}_w{window_length_val}.csv"
                                )

                                data = utilities.AudioData(
                                    L=L_val,
                                    res=res_val,
                                    freq_range=[fmin_val, fmax_val],
                                    window_length=window_length_val,
                                    data_df_save_path=save_path
                                )
                                data.select_geometry(shape_name=g, save_path=parsed.save_path, filename=f"extracted_features_m0_g{g}_L{L_val}_r{res_val}_f{fmin_val}-{fmax_val}.png")
                                data.extract_data(data_path=parsed.input, save=True)
        else:
            print("Didn't do default geo.")


    if parsed.run_cross_val:
        folder_path = parsed.root
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        results = []
        for filename in filenames:
            pattern = r"m(\d+)_g(\d+)_L(\d+)_r(\d+)_f(\d+)-(\d+)_w(\d+)"
            match = re.search(pattern, filename)

            m = int(match.group(1))
            g = match.group(2)
            L = int(match.group(3))
            res = int(match.group(4))
            fmin = int(match.group(5))
            fmax = int(match.group(6))
            window_length = int(match.group(7))
            csv_path = os.path.join(folder_path, filename)
            data = utilities.AudioData(L=L, res=res, freq_range=[fmin, fmax])
            data.read_csv(csv_path=csv_path)
            classifier = SVC(kernel='linear', C=parsed.C, random_state=parsed.seed, probability=True)
            expts = Expts(data, parsed.output, classifier=classifier, random_state=parsed.seed, data_aug=(not parsed.no_data_aug))
            for loc in parsed.locs_list:
                output_metrics = expts.run_cross_validation([loc])
                result_entry = {
                    "filename": filename,
                    "location": loc,
                    "L": L,
                    "resolution": res,
                    "fmin": fmin,
                    "fmax": fmax,
                    "num_mics": m,
                    "geo": g,
                    "window_length": window_length,
                    "overall_accuracy_mean": output_metrics["overall_accuracy"][0],
                    "overall_accuracy_std": output_metrics["overall_accuracy"][1],
                    "per_class_accuracy_mean": output_metrics["per_class_accuracy"][0].tolist(),
                    "per_class_accuracy_std": output_metrics["per_class_accuracy"][1].tolist(),
                    "per_class_precision_mean": output_metrics["per_class_precision"][0].tolist(),
                    "per_class_precision_std": output_metrics["per_class_precision"][1].tolist(),
                    "per_class_recall_mean": output_metrics["per_class_recall"][0].tolist(),
                    "per_class_recall_std": output_metrics["per_class_recall"][1].tolist(),
                    "per_class_iou_mean": output_metrics["per_class_iou"][0].tolist(),
                    "per_class_iou_std": output_metrics["per_class_iou"][1].tolist(),
                    "conf_mat": output_metrics["conf_mat"].tolist()
                }

                results.append(result_entry)

        # 保存为一个总表
        save_path = os.path.join(parsed.root, "result/summary_results_SVM.csv")
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(save_path, index=False)
        print("✅ Saved summary_results_SVM.csv")


    if parsed.run_measure_CNN:
        results = []
        folder_path = parsed.root
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for filename in filenames:
            pattern = r"m(\d+)_g(\d+)_L(\d+)_r(\d+)_f(\d+)-(\d+)_w(\d+)"
            match = re.search(pattern, filename)

            m = int(match.group(1))
            g = match.group(2)
            L = int(match.group(3))
            res = int(match.group(4))
            fmin = int(match.group(5))
            fmax = int(match.group(6))
            window_length = int(match.group(7))
            csv_path = os.path.join(folder_path, filename)
            data = utilities.AudioData(L=L, res=res, freq_range=[fmin, fmax])
            data.read_csv(csv_path=csv_path)
            input_shape = (L, res)  # L, resolution
            num_classes = 4
            classifier = CNNClassifier(input_shape=input_shape, num_classes=num_classes)
            expts = Expts(data, parsed.output, classifier=classifier, random_state=parsed.seed, data_aug=(not parsed.no_data_aug))
            print(f'--- Cross Validation of csv file: {filename}')
            for loc in parsed.locs_list:
                output_metrics = expts.run_measure_CNN([loc], L, res, num_classes)
                result_entry = {
                    "filename": filename,
                    "location": loc,
                    "L": L,
                    "resolution": res,
                    "fmin": fmin,
                    "fmax": fmax,
                    "num_mics": m,
                    "geo": g,
                    "window_length": window_length,
                    "overall_accuracy_mean": output_metrics["overall_accuracy"][0],
                    "overall_accuracy_std": output_metrics["overall_accuracy"][1],
                    "per_class_accuracy_mean": output_metrics["per_class_accuracy"][0],
                    "per_class_accuracy_std": output_metrics["per_class_accuracy"][1],
                    "per_class_precision_mean": output_metrics["per_class_precision"][0],
                    "per_class_precision_std": output_metrics["per_class_precision"][1],
                    "per_class_recall_mean": output_metrics["per_class_recall"][0],
                    "per_class_recall_std": output_metrics["per_class_recall"][1],
                    "per_class_iou_mean": output_metrics["per_class_iou"][0],
                    "per_class_iou_std": output_metrics["per_class_iou"][1],
                    "conf_mat": output_metrics["conf_mat"]
                }
                results.append(result_entry)
        summary_df = pd.DataFrame(results)
        save_path = os.path.join(parsed.root, "result/summary_metrics_CNN.csv")
        os.makedirs(os.path.join(parsed.root, "result"), exist_ok=True)
        summary_df.to_csv(save_path, index=False)
        print("Created summary_metrics_CNN.csv!")