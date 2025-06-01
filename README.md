# MT

- python training_samples.py --extract_feats --input D:\ovad\ovad_dataset_samples\samples --save_path .\preprocess_samples
- python training_samples.py --num_mics 56 32 16 8 4 --extract_feats --input D:\ovad\ovad_dataset_samples\samples --save_path .\preprocess_samples --L 2 4 8 16 --res 30 60 120 240 360 --fmin 20 50 1500 --fmax 50 1500 3000
- python training_samples.py --geo col0 col3 row4 row7  --extract_feats --input D:\ovad\ovad_dataset_samples\samples --save_path .\preprocess_samples --L 2 4 8 16 --res 30 60 120 240 360 --fmin 20 50 1500 --fmax 50 1500 3000
- python training_samples.py --run_cross_val --locs_list SAB SA SB DAB DA DB --root .\preprocess_samples
- python training_samples.py --run_measure_CNN --locs_list SAB SA SB DAB DA DB --root .\preprocess_samples
- python training.py --locs_list SAB SA SB --root .\preprocess_mydata
- python training_SVM.py --locs_list SAB SA SB --root .\preprocess_mydata