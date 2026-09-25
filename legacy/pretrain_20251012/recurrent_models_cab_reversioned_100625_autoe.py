import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
#import os

sys.stderr = sys.stdout 

# Define the path you want to add (absolute or relative)
custom_path = os.path.abspath("Recurrence")

# Add it to sys.path if not already added
if custom_path not in sys.path:
    sys.path.insert(0, custom_path)

import shutil
import argparse
import csv
import pandas as pd
import time
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)


#import keras
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, GRU, LayerNormalization, MultiHeadAttention
from tensorflow.keras import layers, backend as K
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Bidirectional
from tensorflow.keras.utils import custom_object_scope

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight, shuffle
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report, cohen_kappa_score
from sklearn.metrics import f1_score, mean_squared_error

#from utils import *
from embedding import *
from models import *
# import utils_cab as utils
import utils as utils
from sensor_attention import SensorAttention

# export HDF5_USE_FILE_LOCKING="FALSE"
# export CUDA_VISIBLE_DEVICES=1
import itertools

seed = 30 # 1, 10 , 15
tf.random.set_seed(seed)
np.random.seed(seed)

def one_hot(y_, n_classes=7):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def plot_loss_acc(epoch, epoch_accuracy, epoch_loss, epoch_validation, 
          epoch_lossval, directory, dataset, time_, plots_loss_dir):
    plt.figure()  
    plt.ylim(0, 2.0)
    plt.plot(epoch_accuracy.values(), 'r--')
    plt.plot(epoch_validation.values(), 'b--')
    plt.title(dataset + '- loss and accuracy')
    # summarize history for loss
    plt.plot(epoch_loss.values(), 'm-')
    plt.plot(epoch_lossval.values(), 'c-')
    plt.ylabel('loss & accuracy (--)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy', 'train loss', 'val loss'], loc='upper right')
    plt.savefig(plots_loss_dir+"/"+str(epoch)+"_loss-acc_"+dataset+"-{}".format(time_)+'.png')
    plt.close()


def build_metrics(_model, dict_arrays):
    
    model = _model

    full_raws_train_fea_np = dict_arrays['x_train']
    full_raws_y_train = dict_arrays['y_train']
    full_raws_val_fea_np = dict_arrays['x_val']
    full_raws_y_val = dict_arrays['y_val']
    full_raws_test_fea_np = dict_arrays['x_test']
    full_raws_y_test = dict_arrays['y_test']

    # TRAIN #
    preds_t = model.predict(full_raws_train_fea_np)
    preds_t_flat = np.argmax(preds_t, axis=1).reshape(-1)
    accuracy_train = float(np.sum(preds_t_flat==full_raws_y_train))/full_raws_y_train.shape[0]
    balanced_accuracy_train = balanced_accuracy_score(full_raws_y_train, preds_t_flat)
    f1_score_train_we = f1_score(full_raws_y_train,  preds_t_flat, average = 'weighted') #macro weighted
    f1_score_train_mic = f1_score(full_raws_y_train,  preds_t_flat, average = 'micro')
    kappa_train = cohen_kappa_score(full_raws_y_train,  preds_t_flat)

    # VAL #
    preds_t_val = model.predict(full_raws_val_fea_np)
    preds_t_flat_val = np.argmax(preds_t_val, axis=1).reshape(-1)
    accuracy_val = float(np.sum(preds_t_flat_val==full_raws_y_val))/full_raws_y_val.shape[0]
    balanced_accuracy_val = balanced_accuracy_score(full_raws_y_val, preds_t_flat_val)
    f1_score_val_we = f1_score(full_raws_y_val,  preds_t_flat_val, average = 'weighted') #macro weighted
    f1_score_val_mic = f1_score(full_raws_y_val,  preds_t_flat_val, average = 'micro')
    kappa_val = cohen_kappa_score(full_raws_y_val,  preds_t_flat_val)

    # TEST #
    preds_t_test = model.predict(full_raws_test_fea_np)
    preds_t_flat_test = np.argmax(preds_t_test, axis=1).reshape(-1)
    accuracy_test = float(np.sum(preds_t_flat_test==full_raws_y_test))/full_raws_y_test.shape[0]
    balanced_accuracy_test = balanced_accuracy_score(full_raws_y_test, preds_t_flat_test)
    f1_score_test_we = f1_score(full_raws_y_test,  preds_t_flat_test, average = 'weighted') #macro weighted
    f1_score_test_mic = f1_score(full_raws_y_test,  preds_t_flat_test, average = 'micro')
    kappa_test = cohen_kappa_score(full_raws_y_test,  preds_t_flat_test)

    return [[accuracy_train, balanced_accuracy_train, f1_score_train_we, f1_score_train_mic, kappa_train], 
            [accuracy_val, balanced_accuracy_val, f1_score_val_we, f1_score_val_mic, kappa_val],
             [accuracy_test, balanced_accuracy_test, f1_score_test_we, f1_score_test_mic, kappa_test]]

""" 
Plot normalized confusion matrix
"""
def plot_confusion_matrix( y_true, y_pred, LABELS,
                          normalize=True,
                          title=None,
                          path_save='cm.png',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    print("\nCreating the confusion matrix ...")
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Choose normalized values or not
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=LABELS, yticklabels=LABELS,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()    
    plt.savefig(path_save)
    plt.close()


# Transformer encoder
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention

    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, activation="relu")(x)
    #x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


# Positional encoding
def get_pos_encoder(shape):
        return PositionalEncoding(shape[1], shape[2], name='PositionalEncoding')


# In[5]:

# Wavenet
def WaveNetResidualConv1D(num_filters, kernel_size, stacked_layer):

    def build_residual_block(l_input):
        resid_input = l_input
        #for dilation_rate in [2**i for i in range(stacked_layer)]:
        for dilation_rate in [2**i for i in range(6, 9)]:
            l_sigmoid_conv1d = layers.Conv1D(num_filters, kernel_size, dilation_rate=dilation_rate, padding='same', activation='sigmoid')(l_input)
            l_tanh_conv1d = layers.Conv1D( num_filters, kernel_size, dilation_rate=dilation_rate, padding='same', activation='tanh')(l_input)
            l_input = layers.Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
            l_input = layers.Conv1D(num_filters, 1, padding='causal')(l_input)
            resid_input = layers.Add()([resid_input ,l_input])
        return resid_input
    return build_residual_block


def clfOnlyTowers(directory, subdirectory, plots_loss_dir, X_train, obs, table, 
                  time_, dataset, n_epochs, scores, n_batch, cms, LABELS, ftune = False, 
                  with_val=False, modelname = '', modeltype = '', saveModel = True):
    

    print()
    print("clfLstm: ", dataset)
    print(X_train.shape)

    # Concatenate the data
    #X_train = np.concatenate([X_train, X_val, X_test], axis=0)

    
    # Training Hyperparameters
    learning_rate = 1e-4
    dropout_rate = 0.5
    n_batch = n_batch
    LSTM_layers = -1
    lstm_hidden_units = 100
    fconn_units = None
    ff_dim = 9
    lstm_reg = 1e-4 #2e-4
    clf_reg = 1e-4 #2e-4
    clipvalue = 5

    n_epochs = n_epochs # n_epochs # Loop 500 times on the dataset
    verbose = 0


    if ftune:
        # Load Model
        #model = load_json_model(modelname)
        print("Fine tunning")
    else:
        print("\nTraining the model from scratch...")

        # Model Definition
        
        raw_inputs = Input(shape=(X_train.shape[1],  X_train.shape[2])) 
        # (X_train.shape[0], X_train.shape[1], X_train.shape[2]) TensorShape([None, 3138, 100, 9]) Error 
        # raw_inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
        
        #xlstm = T2V(100)(raw_inputs)
        
        #xlstm = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(raw_inputs)
        #xlstm = layers.Dropout(dropout_rate)(xlstm)

        # Conv
        
        #si, _ = SensorAttention(n_filters=512, kernel_size=3, dilation_rate=2)(raw_inputs)

        # Convolutions
        #conv_1 = tf.keras.layers.Conv2D(64, kernel_size=3, dilation_rate=2, padding='same', activation='relu')
        #conv_f = tf.keras.layers.Conv2D(1, kernel_size=1, padding='same')
        # si = tf.expand_dims(raw_inputs, axis=3)
        # si = conv_1(si)  #  [None, 100, 9]
        # si = conv_f(si)
        # si = tf.keras.layers.Reshape(raw_inputs.shape[-2:])(si)

        # Model Definition
        num_filters_ = 2
        kernel_size_ = 3
        stacked_layers_ = [12, 8, 4, 1, 0]
        
        print("xlstm_1:")
        print("Type raw_inputs:", type(raw_inputs))

        from tensorflow.keras.optimizers import Adam, RMSprop
        import tensorflow as tf

        if modeltype == 'autoencoder_s':
            autoencoder = build_autoencoder_s(raw_inputs)

        if modeltype == 'autoencoder':
            import tensorflow as tf
            from tensorflow.keras import layers, models
            from tensorflow.keras.regularizers import l2, l1

            #lstm_hidden_units = 64  # Adjust as needed
            n_classes = 3  # Not used in unsupervised model, but you can keep for reference
            ff_dim = 9
            fconn_units = 100

            # First Convolutional Layer
            x = layers.Conv1D(filters=ff_dim, kernel_size=1)(raw_inputs)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dense(fconn_units)(x)
            x = layers.Dropout(dropout_rate)(x)

            # Second Convolutional Layer
            x = layers.Conv1D(filters=ff_dim, kernel_size=1)(x)
            x = layers.MaxPooling1D(pool_size=2)(x)
            x = layers.Dense(fconn_units)(x)
            x = layers.Dropout(dropout_rate)(x)

            # Recurrent Layers (Bidirectional GRUs)
            x = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=True,
                                                kernel_regularizer=l2(lstm_reg)))(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=True,
                                                kernel_regularizer=l2(lstm_reg)))(x)
            x = layers.Dropout(dropout_rate)(x)

            # Attention Layer
            attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
            attention_output = layers.Dropout(dropout_rate)(attention_output)
            x = layers.Add()([x, attention_output])  # Residual connection
            x = layers.LayerNormalization(epsilon=1e-6)(x)

            # Final GRU Layer
            x = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=False,
                                                kernel_regularizer=l2(lstm_reg)))(x)
            x = layers.Dropout(dropout_rate)(x)

            # Decoder: Reverse the process to reconstruct the input
            # You can use dense layers or another GRU/Bidirectional LSTM
            decoder = layers.Dense(fconn_units)(x)  # Decoder step
            decoder = layers.Dense(500 * 9, activation='sigmoid')(decoder)  # Reshape to original shape
            decoder_output = layers.Reshape((500, 9))(decoder)

            # Model definition
            autoencoder = models.Model(inputs=raw_inputs, outputs=decoder_output)

        # Compile the model
        autoencoder.compile(optimizer=RMSprop(learning_rate=1e-4), 
                            loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])



    #print(model.summary()) # summarize layers
    #plot_model(model, to_file=modeltype+'.png') # plot graph
    
    #best_model = "best_model_"+dataset+"_"+time_+".weights.h5"
    best_model = "best_model_"+dataset+"_"+time_+".keras"
    path_best_model = subdirectory+"/"+best_model
    print("path_best_model: ", path_best_model)

    # class CustomEarlyStopping(tf.keras.callbacks.EarlyStopping):
    #     def __init__(self, monitor='val_accuracy', patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False):
    #         super(CustomEarlyStopping, self).__init__(monitor=monitor, patience=patience, verbose=verbose, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)
    #         self.print_frequency = 10  # Print information every 5 epochs (you can adjust this)

    #     def on_epoch_end(self, epoch, logs=None):
    #         if (epoch + 1) % self.print_frequency == 0:
    #             super(CustomEarlyStopping, self).on_epoch_end(epoch, logs)
    

    class CustomModelCheckPoint(tf.keras.callbacks.Callback):
        def __init__(self,**kargs):
            super(CustomModelCheckPoint,self).__init__(**kargs)
            self.epoch_accuracy = {} # loss at given epoch
            self.epoch_loss = {} # accuracy at given epoch
            self.epoch_validation = {}
            self.epoch_lossval = {}
            self.epoch_balanced_accuracy = {}  # balanced accuracy at given epoch

        def on_epoch_begin(self,epoch, logs={}):
            # Things done on beginning of epoch. 
            return

        def on_epoch_end(self, epoch, logs={}):
            # things done on end of the epoch
            self.epoch_accuracy[epoch] = logs.get("accuracy")
            self.epoch_loss[epoch] = logs.get("loss")
            self.epoch_validation[epoch] = logs.get("val_accuracy")
            self.epoch_lossval[epoch] = logs.get("val_loss")
            self.epoch_balanced_accuracy[epoch] = logs.get("val_balanced_accuracy")

            if(dataset.startswith('de_fake_padts_100-9')):
                freq = 20
            else:
                freq = 5

            if ((epoch % freq)==0):
                plot_loss_acc(epoch, self.epoch_accuracy, 
                              self.epoch_loss, self.epoch_validation, 
                              self.epoch_lossval, directory, dataset, 
                              time_, plots_loss_dir) #a random function
            

    chkpoint = CustomModelCheckPoint()

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='max')
    Reducelr_onplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=15, verbose=1, min_delta=1e-4, mode='min')
    # earlyStopping = CustomEarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='max', restore_best_weights=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=subdirectory+'/logs', histogram_freq=1, write_graph=True, write_images=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            path_best_model , save_freq=100, monitor='val_balanced_accuracy'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            path_best_model , save_best_only=True, monitor='val_balanced_accuracy'
        ),
        chkpoint,
        tensorboard_callback,
    ]
    
    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(
    #         path_best_model , save_best_only=True, monitor='val_accuracy'
    #     )
    # ]

    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, verbose=1
    )

    if with_val:
        # Train the model without labels (unsupervised)
        history = autoencoder.fit(X_train, X_train,  # Same for input and output
                                batch_size=n_batch, 
                                epochs=n_epochs,
                                validation_data=(X_val, X_val), 
                                callbacks=callbacks, 
                                verbose=1)

        # Optionally, save the trained model
        autoencoder.save(subdirectory+"/autoencoder_model.h5")

        # Read MSE per epoch
        train_mse = history.history['loss']
        val_mse   = history.history['val_loss']

        for i, (tr, va) in enumerate(zip(train_mse, val_mse), 1):
            print(f"Epoch {i}: train MSE={tr:.6f}, val MSE={va:.6f}")

        # Plot
        plt.plot(train_mse, label='Train MSE')
        plt.plot(val_mse, label='Validation MSE')
    
    else:
        # Train the model without labels (unsupervised)
        history = autoencoder.fit(
                    X_train, X_train,           # input = output for autoencoder
                    batch_size=n_batch,
                    epochs=n_epochs,
                    callbacks=callbacks,
                    verbose=1
                )

        # Optionally, save the trained model
        autoencoder.save(subdirectory+"/autoencoder_model.h5")

        # Read MSE per epoch
        train_mse = history.history['loss']

        for i, tr in enumerate(train_mse, 1):
            print(f"Epoch {i}: train MSE={tr:.6f}")
        plt.plot(train_mse, label='Train MSE')
    
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(subdirectory+"/autoencoder_model_"+dataset+"-{}".format(time_)+'.png') 

    print(directory)
    print(subdirectory)


# Main function
def main():
    # Table of results
    table = pd.DataFrame(columns=['time', 'dataset', 'n_epochs', 'lr','do','batch', 'layers', 
                                        'h_units', 'lstm_reg', 'clf_reg', 'clipvalue', 'train', 'train_bal', 
                                        'f1_score_train_we', 'kappa_train',
                                        'val', 'val_bal', 'f1_score_val_we', 'kappa_val',
                                        'test', 'test_bal', 'f1_score_test_we', 'kappa_test', 'obs'])
    
    # Variables
    #model_type_list = ['wavebigru', 'atteBigru', 'transbigru', 'bigru', 'attnbigru'] 
    model_type_list = ['autoencoder_s'] #['autoencoder', 'transbigru', 'bigru', 'bilstm'] #['attnbigru']
    test_type = ['nusers']   # 'oneuser'
    k_folds = [1]
    norm_method_list = [0]
    n_epochs = [20] # [300] 
    dataset_list = ['de_fake_padts_94u']   # "PCF-GAN/data/DEO10/deo_drinkeat_94u_train_acc_gyr_mag_coord_5secs_100hz.csv"
      #dataset_list = ['deo_drinkeat_94u_train']  # de_fake_padts_94u

    learning_rate = [1e-5]
    dropout_rate = [0.5]
    overlap_shift = [0.5]
    n_batch = [256, 512] #[1024] #[512]
    LSTM_layers = [-1]
    sensors = [3]
    seg5 = [True]

    overlap = True
    normalize = True
    with_val = False
    
    # Hyperparameters
    a = [model_type_list, test_type, k_folds, norm_method_list, dataset_list, 
                       learning_rate, dropout_rate, overlap_shift, n_batch, 
                       n_epochs, LSTM_layers, sensors, seg5]
    combs = list(itertools.product(*a))
    

    # Directory
    time_ = time.strftime("%Y%m%d-%H%M%S")
    directory = os.getcwd() + '/saved_models/' + 'recurrent_models_'+ time_

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save script
    source_file = '/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/'
    source_file = source_file + 'Recurrence/recurrent_models_cab_reversioned_100625_autoe.py'
    destination_folder = directory+"/recurrent_models_cab_reversioned_100625_autoe.py"
    print("Saving the code at: ", destination_folder)
    shutil.copy(source_file, destination_folder)


    dataset = dataset_list[0]

    _, _, _, file_full_raws = utils.get_raw_datasets(dataset, magni=False)
    print(file_full_raws)

    if ( dataset == 'vivabem12_lying' ):
        df = utils.read_full_raws_without_tv(file_full_raws)
    elif ( dataset == 'vivabem12_tv' ):
        df = utils.read_full_raws_without_lying(file_full_raws)
    else:
        df = utils.read_full_raws(file_full_raws, dataset_list[0])

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

        print("Comb: ", comb)

        obs = ''
        
        seed = 30
        tf.random.set_seed(seed)
        np.random.seed(seed)


        # Save df
        # df_name = 'df_'+dataset+'.csv'
        # print("Saving df in...", df_name)
        # df.to_csv(df_name, sep=',', encoding='utf-8', index=False)
        # print("df shape: ", df.shape)

        scores = []
        cms = []
        
        # Get the class names from dataframe
        LABELS = utils.get_class_names(dataset)
#         if (df.iloc[0,4] != 'met' & not (dataset.startswith('eatdrinkanother'))):
#             df_activities = df.groupby([2, 4]).size().reset_index(name='Count')
#             LABELS = np.unique(df_activities[4])
                    
        fold = 1
        time_ = time.strftime("%Y%m%d-%H%M%S") 
        subdirectory = directory + '/' + dataset + '_'+str(fold)+'f_'+modeltype + '_'+time_
        print("subdirectory: ", subdirectory)
        print("directory: ", directory)
        if not os.path.exists(subdirectory):
                os.makedirs(subdirectory)

        # Define file path
        file_path = os.path.join(subdirectory, "hyperparams.txt")

        # Save string to file
        with open(file_path, "w") as f:
            f.write(hyperparams)

        plots_loss_dir = os.path.join(subdirectory,'plots_loss_dir')
        if not os.path.exists(plots_loss_dir):
            os.makedirs(plots_loss_dir)

        # get_processed_fold_valtestuser, get_processed_fold_testuser, get_processed_fold
        train_X = utils.get_processed_fold_unsup(df, dataset, modeltype, subdirectory, sensors, fold,
                                                seg5, normalize, overlap, overlap_shift=overlap_shift)
        print("Model type:", modeltype)
        
        # clfOnlyTowersTest has only GRU model
        table = clfOnlyTowers(directory, subdirectory, plots_loss_dir, train_X, obs, table, 
                              time_, dataset, n_epochs, scores,
                            n_batch, cms, LABELS, ftune=False, with_val= with_val, modelname=modeltype, 
                            modeltype=modeltype, saveModel=True )

        # table.at[len(table)-1,'ov'] = str(overlap_shift)

        # print("End of the fold: "+str(fold))
        # print(table)
        # print("End of the training in dataset: "+dataset)

        # # COMPUTE THE AVERAGE OF ALL THE LISTS OF SCORES
        # scores = np.array(scores)
        # scores = scores.mean(axis=0)
        # scores = scores.tolist()
        # #table = table.append( table.iloc[-1] )
        # table = pd.concat([table, table.iloc[[-1]]], ignore_index=True)
        # table.reset_index(drop=True, inplace=True)
        # table.at[len(table)-1,'test'] = scores[0]
        # table.at[len(table  )-1,'test_bal'] = scores[1]
        # table.at[len(table)-1,'f1_score_test_we'] = scores[2]
        # table.at[len(table)-1,'kappa_test'] = scores[3]
        # table.at[len(table)-1,'uuid_val'] = '-'
        # table.at[len(table)-1,'uuid_test'] = 'avg'
        # table = table.round(3)

        # table_name = directory + "/table_"+dataset+".csv"
        # print("Saving table in...", table_name)
        # table.to_csv(table_name, sep=',', encoding='utf-8', index=False)
        # print(table)
        # print("Model name: ", modeltype)
        # print("Directory: ", directory)    

        # endtime_ = time.strftime("%Y%m%d-%H%M%S")
        # print("The program finished at: ", endtime_) 

    
    # MAIN
main()

# In[9]: