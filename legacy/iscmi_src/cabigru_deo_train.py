import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import utils_cab as utils

import shutil
import argparse
import csv
import pandas as pd
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#from embedding import *
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Layer, Bidirectional
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score, cohen_kappa_score
from sklearn.metrics import f1_score, mean_squared_error

from tensorflow.keras import layers
from tensorflow.keras import backend as K

import tensorflow as tf
import itertools

seed = 30 # 1, 10 , 15
tf.random.set_seed(seed)
np.random.seed(seed)

# Import custom modules
import model_builder
import plotting_utils
import training_utils


"""
Training step
"""
def training(directory, subdirectory, plots_loss_dir, dict_arrays, obs, table, 
                  time_, dataset, n_epochs, scores, n_batch, cms, LABELS, ftune = False, 
                  modelname = '', modeltype = '', saveModel = True):
    
    # Load "X" and "y" and "epochs" (the neural network's training and testing inputs)
    X_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    X_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    X_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']

    # Number of classes
    n_classes = int(max(y_train.max(), y_val.max())+1)
    
    # Training Hyperparameters
    learning_rate = 1e-4
    dropout_rate = 0.5
    n_batch = n_batch
    LSTM_layers = -1
    lstm_hidden_units = 100
    fconn_units = None
    ff_dim = 9
    lstm_reg = 1e-4
    clf_reg = 1e-4
    clipvalue = 5

    n_epochs = n_epochs # n_epochs # Loop 500 times on the dataset
    verbose = 0

    print("\nTraining the model...")

    # Build the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    hyperparameters = {
        'dropout_rate': dropout_rate,
        'lstm_hidden_units': lstm_hidden_units,
        'lstm_reg': lstm_reg,
        'clf_reg': clf_reg,
        'n_epochs': n_epochs,
        'learning_rate': learning_rate,
        'clipvalue': clipvalue,
        'n_batch': n_batch,
        'LSTM_layers': LSTM_layers,
        'fconn_units': fconn_units,
        'verbose': verbose
    }
    builder = model_builder.ModelBuilder(modeltype, input_shape, n_classes, hyperparameters)
    model = builder.build()
    
    # Create callbacks
    best_model = "best_model_"+dataset+"_"+time_+".h5"
    path_best_model = subdirectory+"/"+best_model
    print("path_best_model: ", path_best_model)
    
    callbacks = training_utils.create_callbacks(path_best_model)


    # Compile the model
    training_utils.compile_model(model, learning_rate, clipvalue)

    # Train the model
    history=model.fit(X_train, 
                    utils.one_hot(y_train, n_classes), 
                    batch_size=n_batch,
                    epochs=n_epochs,
                    callbacks = callbacks,
                    validation_data=(X_val, utils.one_hot(y_val, n_classes)),
                    verbose = 1)

    # Save the history
    training_utils.save_history(history, subdirectory, dataset, modelname, time_)
    
    # Load the best model
    model = training_utils.load_best_model(path_best_model)

    # Predictions
    predictions_test = model.predict(X_test)
    predictions_val = model.predict(X_val)
    predictions_train = model.predict(X_train)

    # Evaluate the model
    predictions = model.predict(X_test)    
    
    # Plot training history
    plotting_utils.plot_training_history(history, subdirectory, time_, dataset)

    # Compute all metrics
    metric_results = [[], [], []]
    print("Test:")
    utils.compute_all_metrics(y_test, predictions_test, dataset, time_, subdirectory, metric_results[0], split='test')
    print("Val:")
    utils.compute_all_metrics(y_val, predictions_val, dataset, time_, subdirectory, metric_results[1], split='val')
    print("Train:")
    utils.compute_all_metrics(y_train, predictions_train, dataset, time_, subdirectory, metric_results[2], split='train')

    # Compute metrics
    metric_results = utils.build_metrics(model, dict_arrays)
    
    # Fill the table
    new_row = { 'time' : time_,
                'modeltype' : modeltype,
                'dataset': dataset,
                'uuid_val': dict_arrays['uuid_val'],
                'uuid_test': dict_arrays['uuid_test'],
                'n_epochs': n_epochs, 
                'lr' : str(learning_rate),
                'do' : str(dropout_rate),
                'ov' : "",
                'batch' : str(n_batch),
                'layers' : str(LSTM_layers),
                'h_units' : str(lstm_hidden_units),
                'lstm_reg' : str(lstm_reg),
                'clf_reg' : str(clf_reg),
                'clipvalue' : str(clipvalue),

                'train': round(metric_results[0][0], 4),
                'train_bal' : round(metric_results[0][1], 4),
                'f1_score_train_we': round(metric_results[0][2], 4),
                'kappa_train': round(metric_results[0][4], 4),

                'val': round(metric_results[1][0], 4),
                'val_bal': round(metric_results[1][1], 4),
                'f1_score_val_we': round(metric_results[1][2], 4),
                'kappa_val': round(metric_results[1][4], 4),
                'test': round(metric_results[2][0], 4),
                'test_bal': round(metric_results[2][1], 4),
                'f1_score_test_we': round(metric_results[2][2], 4),
                'kappa_test': round(metric_results[2][4], 4),
                'obs' : obs,
                'path': subdirectory
            }

	# Append [acc_test, bal_acc_test, f1_test, k_test]
    scores.append([metric_results[2][0], metric_results[2][1], metric_results[2][2], metric_results[2][4]])

    # Append to the table
    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True) 

    # Saving the table
    table_name = subdirectory+"/table_"+dataset+"_"+time_+".csv"
    print("Saving table in...", table_name)
    table.to_csv(table_name, sep=',', encoding='utf-8', index=False)

    # SAVE THE MODEL
    if saveModel:
        # Saving the model
        training_utils.save_model_and_metadata(model, subdirectory, time_, dataset, obs, hyperparameters, history)
        print("Model saved !")

    else:
        print("Model not saved.")
        
    return table


# Main function
def main():

    # Table of results
    table = pd.DataFrame(columns=['time', 'dataset', 'n_epochs', 'lr','do','batch', 'layers', 
                                        'h_units', 'lstm_reg', 'clf_reg', 'clipvalue', 'train', 'train_bal', 
                                        'f1_score_trai n_we', 'kappa_train',
                                        'val', 'val_bal', 'f1_score_val_we', 'kappa_val',
                                        'test', 'test_bal', 'f1_score_test_we', 'kappa_test', 'obs'])
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train HAR model')
    parser.add_argument('--model_types', type=str, default='attnbigru', help='Comma-separated list of model types')
    parser.add_argument('--test_types', type=str, default='nusers', help='Comma-separated list of test types')
    parser.add_argument('--k_folds', type=str, default='1', help='Comma-separated list of k folds')
    parser.add_argument('--norm_methods', type=str, default='0', help='Comma-separated list of normalization methods')
    parser.add_argument('--n_epochs', type=str, default='1', help='Comma-separated list of number of epochs')
    parser.add_argument('--datasets', type=str, default='eatdrinkanother', help='Comma-separated list of datasets')
    parser.add_argument('--learning_rates', type=str, default='1e-4', help='Comma-separated list of learning rates')
    parser.add_argument('--dropout_rates', type=str, default='0.5', help='Comma-separated list of dropout rates')
    parser.add_argument('--overlap_shifts', type=str, default='0.5', help='Comma-separated list of overlap shifts')
    parser.add_argument('--n_batches', type=str, default='1024', help='Comma-separated list of batch sizes')
    parser.add_argument('--lstm_layers', type=str, default='-1', help='Comma-separated list of LSTM layers')
    parser.add_argument('--sensors', type=str, default='3', help='Comma-separated list of sensors')
    parser.add_argument('--seg5', type=str, default='True', help='Comma-separated list of seg5 flags')
    args = parser.parse_args()

    # Example usage: python src/cabigru_deo_train.py --n_epochs 10,50 --learning_rates 1e-3,1e-4 --datasets eatdrinkanother,otherdataset
    
    model_type_list = args.model_types.split(',')
    test_type = args.test_types.split(',')
    k_folds = [int(x) for x in args.k_folds.split(',')]
    norm_method_list = [int(x) for x in args.norm_methods.split(',')]
    n_epochs = [int(x) for x in args.n_epochs.split(',')]
    dataset_list = args.datasets.split(',')
    learning_rate = [float(x) for x in args.learning_rates.split(',')]
    dropout_rate = [float(x) for x in args.dropout_rates.split(',')]
    overlap_shift = [float(x) for x in args.overlap_shifts.split(',')]
    n_batch = [int(x) for x in args.n_batches.split(',')]
    LSTM_layers = [int(x) for x in args.lstm_layers.split(',')]
    sensors = [int(x) for x in args.sensors.split(',')]
    seg5 = [x.lower() == 'true' for x in args.seg5.split(',')]
    
    # Hyperparameters
    a = [model_type_list, test_type, k_folds, norm_method_list, dataset_list, 
                       learning_rate, dropout_rate, overlap_shift, n_batch, 
                       n_epochs, LSTM_layers, sensors, seg5]
    combs = list(itertools.product(*a))    
        
    #with tf.device('/gpu:'+device):
    for comb in combs:
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

        hyperparams = modeltype+"_"+test_type+"_"+str(k_folds)+"_"+str(overlap_shift)+"_"+str(norm_method)
        hyperparams = hyperparams+"_"+ dataset+"_"+str(learning_rate)+"_"+str(dropout_rate)
        hyperparams = hyperparams+"_"+str(n_batch)+"_"+str(n_epochs)+"_"+str(LSTM_layers)+"_"+str(sensors)        
        
        # Directory
        time_ = time.strftime("%Y%m%d-%H%M%S")
        directory = os.getcwd() + '/saved_models/' + modeltype+'_'+ time_
        if not os.path.exists(directory):
            os.makedirs(directory)  
        print(directory)

        # Save script
        source_file = os.path.basename(__file__)
        destination_file = os.path.join(directory, source_file)
        print("Saving the code: ", destination_file)
        shutil.copy(__file__, destination_file)

        obs = ''
        
        seed = 30
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        _, _, _, file_full_raws = utils.get_raw_datasets(dataset, magni=False)
        print(file_full_raws)
        df = utils.read_full_raws_without_tv(file_full_raws)

        scores = []
        cms = []
        
        # Get the class names from dataframe
        LABELS = utils.get_class_names(dataset)

        fold = 1
        time_ = time.strftime("%Y%m%d-%H%M%S") 
        subdirectory = directory + '/' + dataset + '_'+str(fold)+'f_'+modeltype + '_'+time_
        print("subdirectory: ", subdirectory)
        print("directory: ", directory)
        if not os.path.exists(subdirectory):
                os.makedirs(subdirectory)

        plots_loss_dir = os.path.join(subdirectory,'plots_loss_dir')
        if not os.path.exists(plots_loss_dir):
            os.makedirs(plots_loss_dir)

        # Get splits in a dictionary
        dict_arrays = utils.get_processed_fold(df, dataset, subdirectory, sensors, fold, seg5, overlap_shift)
        
        # Training step
        table = training(directory, subdirectory, plots_loss_dir, dict_arrays, obs, table, time_, dataset, n_epochs, scores,
                            n_batch, cms, LABELS, ftune=False, modelname=modeltype, modeltype=modeltype, saveModel=True )

        # Set the overlap shift in the table
        table.at[len(table)-1,'ov'] = str(overlap_shift)

        # Compute the average of the scores
        scores = np.array(scores)
        scores = scores.mean(axis=0)
        scores = scores.tolist()
        table = pd.concat([table, table.iloc[[-1]]], ignore_index=True)
        table.reset_index(drop=True, inplace=True)
        table.at[len(table)-1,'test'] = scores[0]
        table.at[len(table  )-1,'test_bal'] = scores[1]
        table.at[len(table)-1,'f1_score_test_we'] = scores[2]
        table.at[len(table)-1,'kappa_test'] = scores[3]
        table.at[len(table)-1,'uuid_val'] = '-'
        table.at[len(table)-1,'uuid_test'] = 'avg'
        table = table.round(3)
    
    # Finish of the training
    endtime_ = time.strftime("%Y%m%d-%H%M%S")
    table_name = directory + "/table_"+dataset+".csv"
    print("Saving table in...", table_name)
    table.to_csv(table_name, sep=',', encoding='utf-8', index=False)
    print(table)
    print("Model name: ", modeltype)
    print("Directory: ", directory)

    print("The program finished at: ", endtime_) 


# MAIN
main()