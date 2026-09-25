import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import time
import shutil
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
import utils as utils
from parse_args import parse_args  # Import the argument parser
from recurrent_models_reversioned_100625_ori import clfOnlyTowers
import finetune_models as ftm

# --- After parsing args ---
def ensure_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def main():
    args = parse_args()  # Parse the command line arguments

    # Table of results  
    table = pd.DataFrame(columns=['time', 'dataset', 'n_epochs', 'lr','do','batch', 'layers', 
                                  'h_units', 'lstm_reg', 'clf_reg', 'clipvalue', 'train', 'train_bal', 
                                  'f1_score_train_we', 'kappa_train',
                                  'val', 'val_bal', 'f1_score_val_we', 'kappa_val',
                                  'test', 'test_bal', 'f1_score_test_we', 'kappa_test', 'obs'])
    

    
    # Set the GPU based on the command-line argument
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Set GPU number

    # Print out the GPU being used for verification
    print(f"Using GPU: {args.gpu}")

    # Check that TensorFlow is using the correct GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"Available physical devices: {physical_devices}")
    try:
        tf.config.set_visible_devices(physical_devices[int(args.gpu)], 'GPU')
    except ValueError as e:
        print(f"Error setting GPU {args.gpu}: {e}")
        exit(1)

    # DEFAULT PARAMETERS
    # # Use the parsed arguments here
    # n_batch = [128]
    # model_type_list = ['autoencoder_s']
    # dataset_list = ['eatdrinkanother_94u_10']
    # finetune_models_list = ['13']

    # # Use the parsed arguments here
    # test_type = [args.test_type]
    # k_folds = [args.k_folds]
    # norm_method_list = [args.norm_method_list]
    # n_epochs = [args.n_epochs]
    # learning_rate = [args.learning_rate]
    # dropout_rate = [args.dropout_rate]
    # overlap_shift = [args.overlap_shift]
    # LSTM_layers = [args.LSTM_layers]
    # sensors = [args.sensors]
    # seg5 = [args.seg5]
    # gpu = str(args.gpu)
    # finetune_model_n = args.finetune_model

    # PARSER
    # Use the parsed arguments here
    n_batch              = ensure_list(args.n_batch)
    model_type_list      = ensure_list(args.model_type_list)
    dataset_list         = ensure_list(args.dataset_list)
    finetune_models_list = ensure_list(args.finetune_models_list)
    activation           = ensure_list(args.activation)
    n_dense              = ensure_list(args.n_dense)
    learning_rate        = ensure_list(args.learning_rate)
    mode                 = ensure_list(args.modee)

    # Use the parsed arguments here
    test_type = [args.test_type]
    k_folds = [args.k_folds]
    norm_method_list = [args.norm_method_list]
    n_epochs = [args.n_epochs]
    dropout_rate = [args.dropout_rate]
    overlap_shift = [args.overlap_shift]
    LSTM_layers = [args.LSTM_layers]
    sensors = [args.sensors]
    seg5 = [args.seg5]

    gpu = str(args.gpu)

    # Hyperparameters
    a = [model_type_list, test_type, k_folds, norm_method_list, dataset_list, 
         learning_rate, dropout_rate, overlap_shift, n_batch, 
         n_epochs, LSTM_layers, sensors, seg5, finetune_models_list, activation, n_dense, mode]
    
    combs = list(itertools.product(*a))

    # Directory
    time_ = time.strftime("%Y%m%d-%H%M%S")
    directory = os.getcwd() + '/saved_models/' + 'recurrent_models_' + time_
    print("Directory: ", directory)

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Base path for your scripts
    base_path = '/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/'

    # Save the code
    source_file = base_path + 'Recurrence/recurrent_models_reversioned_100625_ori.py'
    destination_folder = directory + "/recurrent_models_reversioned_100625_ori.py"
    print("Saving the code at: ", destination_folder)
    shutil.copy(source_file, destination_folder)

    destination_folder = directory + "/utils.py"
    shutil.copy(base_path + 'Recurrence/utils.py', destination_folder)

    destination_folder = directory + "/models.py"
    shutil.copy(base_path + 'Recurrence/models.py', destination_folder)

    destination_folder = directory + "/recurrent_models_main.py"
    shutil.copy(base_path + 'Recurrence/recurrent_models_main.py', destination_folder)

    destination_folder = directory + "/parse_args.py"
    shutil.copy(base_path + 'Recurrence/parse_args.py', destination_folder)

    # Dataset and utils function
    dataset = dataset_list[0]
    _, _, _, file_full_raws = utils.get_raw_datasets(dataset, magni=False)
    print(file_full_raws)

    if dataset == 'vivabem12_lying':
        df = utils.read_full_raws_without_tv(file_full_raws)
    elif dataset == 'vivabem12_tv':
        df = utils.read_full_raws_without_lying(file_full_raws)
    else:
        df = utils.read_full_raws(file_full_raws, dataset)

    # Running experiments with combinations of hyperparameters
    print("combs: ", len(combs))
    i = 0
    for comb in combs:
        # Extract hyperparameters from comb
        modeltype = comb[0]
        test_type = comb[1]
        k_folds = comb[2]
        norm_method = comb[3]
        dataset = comb[4]
        learning_rate = comb[5]
        dropout_rate = comb[6]
        overlap_shift = comb[7]
        n_batch = comb[8]
        n_epochs = comb[9]
        LSTM_layers = comb[10]
        sensors = comb[11]
        seg5 = comb[12]
        finetune_model_n = comb[13]
        activation = comb[14]
        n_dense = comb[15]
        mode = comb[16]
    
        finetune_model = ftm.fine_tune_models[finetune_model_n]
        print("Pre-trained model:", finetune_model)
        print()

        hyperparams = f"{modeltype}_{gpu}_{test_type}_{k_folds}_{overlap_shift}_{norm_method}_{dataset}_{n_dense}"
        hyperparams = hyperparams + f"_{learning_rate}_{dropout_rate}_{n_batch}_{n_epochs}_{LSTM_layers}_{sensors}_{activation}_{mode}"
        print("Comb: ", comb)

        i = i + 1

        #exit()

        # Set random seeds
        seed = 30
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Initialize performance tracking variables
        scores = []
        cms = []

        LABELS = utils.get_class_names(dataset)
        fold = 1
        time_ = time.strftime("%Y%m%d-%H%M%S")

        subdirectory = f"{directory}/{dataset}_{fold}f_{modeltype}_{time_}"
        print("subdirectory: ", subdirectory)

        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        # Save hyperparameters to file
        file_path = os.path.join(subdirectory, "hyperparams_"+str(i)+".txt")
        with open(file_path, "w") as f:
            f.write(hyperparams)
            f.write("\n")
            f.write(finetune_model)

        plots_loss_dir = os.path.join(subdirectory, 'plots_loss_dir')
        if not os.path.exists(plots_loss_dir):
            os.makedirs(plots_loss_dir)

        dict_arrays = utils.get_processed_fold(df, dataset, modeltype, subdirectory, sensors, fold,
                                                seg5, normalize=True, overlap=False, overlap_shift=overlap_shift)                    

        table = clfOnlyTowers(directory, subdirectory, finetune_model, plots_loss_dir, dict_arrays, '', table, 
                              time_, dataset, n_epochs, scores, n_batch, float(learning_rate), cms, LABELS, int(n_dense), mode, activation,
                              testing=False, ftune=True, modelname=modeltype, modeltype=modeltype, saveModel=True)

        print("Iteration ", i, " of ", len(combs), " completed.")
        print()
    

    # Define the directories to search
    dirs = [
        directory
    ]

    # Regex to match the desired filename pattern and score line
    filename_pattern = re.compile(r"^metrics_test_eatdrinkanother_94u_.*\.txt$")
    score_pattern = re.compile(r"balanced_accuracy_score:\s*([\d.]+)")

    # Store results as (path, score)
    results = []

    for base_dir in dirs:
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if filename_pattern.match(fname):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    match = score_pattern.search(content)
                    if match:
                        score = float(match.group(1))
                        results.append((fpath, score))

    # Print results with only the last 9 characters of the path
    for path, score in results:
        print(f"{path[-13:]}: {score:.4f}")

    # End of the program
    time_ = time.strftime("%Y%m%d-%H%M%S")
    print("End of the training at: ", time_)

if __name__ == "__main__":
    main()