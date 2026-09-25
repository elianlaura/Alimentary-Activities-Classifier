import sys
import os

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

print("GPUs:", tf.config.list_physical_devices('GPU'))  # sanity check

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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

setup_seed(30)

# seed = 30 # 1, 10 , 15
# tf.random.set_seed(seed)
# np.random.seed(seed)

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


def f1_score_weighted(y_true, y_pred):
    # Ensure both y_true and y_pred are float32 to avoid type mismatch
    y_true = K.cast(y_true, 'float32')
    y_pred = K.round(y_pred)  # Round y_pred to get binary values (0 or 1)
    y_pred = K.cast(y_pred, 'float32')

    # Calculate true positives, false positives, and false negatives for each class
    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)

    # Calculate precision and recall for each class
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    # Calculate F1 score for each class
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())

    # Handle NaN values that may result from divisions by zero
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

    # Calculate weights based on the support of each class
    support = K.sum(y_true, axis=0)
    weights = support / K.sum(support)

    # Calculate the weighted F1 score
    weighted_f1 = K.sum(f1 * weights)

    return weighted_f1

# Balanced Accuracy
class BalancedAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.train_bal_acc = []
        self.val_bal_acc = []

    def on_epoch_end(self, epoch, logs=None):
        # Training balanced accuracy
        logs = logs or {}
        y_pred_train = self.model.predict(self.X_train, verbose=0)
        
        y_true_train = np.argmax(self.y_train, axis=1)
        y_pred_train_classes = np.argmax(y_pred_train, axis=1)
        bal_acc_train = metrics.balanced_accuracy_score(y_true_train, y_pred_train_classes)
        self.train_bal_acc.append(bal_acc_train)

        # Validation balanced accuracy
        y_pred_val = self.model.predict(self.X_val, verbose=0)
        y_true_val = np.argmax(self.y_val, axis=1)
        y_pred_val_classes = np.argmax(y_pred_val, axis=1)
        bal_acc_val = metrics.balanced_accuracy_score(y_true_val, y_pred_val_classes)
        self.val_bal_acc.append(bal_acc_val)

        logs['balanced_accuracy'] = bal_acc_train
        logs['val_balanced_accuracy'] = bal_acc_val

        print(f"\nEpoch {epoch+1}: train_bal_acc={bal_acc_train:.4f}, val_bal_acc={bal_acc_val:.4f}")


def clfOnlyTowers(directory, subdirectory, finetune_model, plots_loss_dir, dict_arrays, obs, table, 
                  time_, dataset, n_epochs, scores, n_batch, learning_rate, cms, LABELS, n_dense, mode,
                  activation=None, testing = False, ftune = False, modelname = '', modeltype = '', saveModel = True):
    
    # Load "X" and "y" and "epochs" (the neural network's training and testing inputs) testing inputs

    X_train = dict_arrays['x_train']
    y_train = dict_arrays['y_train']
    X_val = dict_arrays['x_val']
    y_val = dict_arrays['y_val']
    X_test = dict_arrays['x_test']
    y_test = dict_arrays['y_test']

    # Save the numpy elements of the dictionary dict_arrays in .npy files
    # subdirectory = 'npy'
    # np.save(subdirectory+'/x_train.npy', X_train)
    # np.save(subdirectory+'/y_train.npy', y_train)
    # np.save(subdirectory+'/X_val.npy', X_val)
    # np.save(subdirectory+'/y_val.npy', y_val)
    # np.save(subdirectory+'/X_test.npy', X_test)
    # np.save(subdirectory+'/y_test.npy', y_test)

    # Concatenate the data
    #X_train = np.concatenate([X_train, X_val, X_test], axis=0)


    n_classes = int(max(y_train.max(), y_val.max())+1) # Total classes (should go up, or should go down)
    print("\nn_classes = {}".format(n_classes))
    
    # Training Hyperparameters
    learning_rate = learning_rate
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

    acc_type = 'accuracy'
    val_type = 'val_accuracy'

    #best_model = "best_model_"+dataset+"_"+time_+".weights.h5"
    best_model = "best_model_"+dataset+"_"+time_+".keras"
    path_best_model = subdirectory+"/"+best_model
    print("path_best_model: ", path_best_model)

    import tensorflow as tf
    from tensorflow.keras import layers, models
    from collections import Counter
    from tensorflow.keras.losses import CategoricalFocalCrossentropy

    
    # Instantiate the balanced accuracy metric        
    balanced_accuracy_callback = BalancedAccuracyCallback(X_train, 
                                                            one_hot(y_train, n_classes), 
                                                            X_val, 
                                                            one_hot(y_val, n_classes))


    def f1_score_weighted(y_true, y_pred):
        # Ensure both y_true and y_pred are float32 to avoid type mismatch
        y_true = K.cast(y_true, 'float32')
        y_pred = K.round(y_pred)  # Round y_pred to get binary values (0 or 1)
        y_pred = K.cast(y_pred, 'float32')

        # Calculate true positives, false positives, and false negatives for each class
        tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)

        # Calculate precision and recall for each class
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        # Calculate F1 score for each class
        f1 = 2 * precision * recall / (precision + recall + K.epsilon())

        # Handle NaN values that may result from divisions by zero
        f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)

        # Calculate weights based on the support of each class
        support = K.sum(y_true, axis=0)
        weights = support / K.sum(support)

        # Calculate the weighted F1 score
        weighted_f1 = K.sum(f1 * weights)

        return weighted_f1

    class LrChangeLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.prev_lr = None

        def on_epoch_end(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            if self.prev_lr is None or lr != self.prev_lr:
                print(f"\nEpoch {epoch+1}: Learning rate changed to {lr:.6g}")
            self.prev_lr = lr

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
        
        def on_train_begin(self, logs=None):
            # Keras fills self.params when fit() starts
            self.total_epochs_planned = self.params.get("epochs", None)

        def on_epoch_end(self, epoch, logs={}):
            # things done on end of the epoch
            self.epoch_accuracy[epoch] = logs.get("accuracy")
            self.epoch_loss[epoch] = logs.get("loss")
            self.epoch_validation[epoch] = logs.get("val_accuracy")
            self.epoch_lossval[epoch] = logs.get("val_loss")
            self.epoch_balanced_accuracy[epoch] = logs.get("val_balanced_accuracy")

            if ((epoch % 5)==0):
                plot_loss_acc(epoch, self.epoch_accuracy, 
                            self.epoch_loss, self.epoch_validation, 
                            self.epoch_lossval, directory, dataset, 
                            time_, plots_loss_dir) #a random function            
            
            if (epoch + 1) == self.total_epochs_planned:
                plot_loss_acc(epoch, self.epoch_accuracy, 
                              self.epoch_loss, self.epoch_validation, 
                              self.epoch_lossval, directory, dataset, 
                              time_, plots_loss_dir) #a random function

    class WeightsLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\n— Pesos al final de la época {epoch+1} —")
            for layer in self.model.layers:
                w = layer.get_weights()
                if w:  # capa con pesos (conv, dense…)
                    w0 = w[0]  # matriz de pesos (no bias)
                    print(f"{layer.name:20s} mean={w0.mean():.4f}  std={w0.std():.4f}")
            print()

    weights_cb = WeightsLogger()

    chkpoint = CustomModelCheckPoint()

    lrchangelog = LrChangeLogger()

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, verbose=0, mode='max')

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        path_best_model , save_best_only=True, monitor='val_balanced_accuracy'
    )
    if ftune:
        # Load Model
        #model = load_json_model(modelname)
        print("Fine tunning")      

        autoencoder = tf.keras.models.load_model(finetune_model, custom_objects={"T2V": T2V, 
                                                                            "SensorAttention": SensorAttention, 
                                                                            "PositionalEncoding": PositionalEncoding,
                                                                            'WaveNetResidualConv1D': WaveNetResidualConv1D,
                                                                            'BalancedAccuracyCallback': BalancedAccuracyCallback,
                                                                            'balanced_accuracy': utils.balanced_accuracy,
                                                                            'f1_score_weighted':f1_score_weighted})
        
        if modeltype == 'autoencoder_s':
            # Find the last Bidirectional layer
            # Collect all Bidirectional layers (in normal order)
            bidirectional_layers = [layer for layer in autoencoder.layers if isinstance(layer, tf.keras.layers.Bidirectional)]

            # Check if the model has at least 3 Bidirectional layers
            if len(bidirectional_layers) < 3:
                raise ValueError(f"Model has only {len(bidirectional_layers)} Bidirectional layers, need at least 3.")

            # Select the 3rd Bidirectional layer from the end
            target_layer = bidirectional_layers[-3]

            print("Selected Bidirectional layer name:", target_layer.name)

            # encoder = tf.keras.models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('bidirectional_4').output)
        
        elif modeltype == 'autoencoder':
            # Find the last Bidirectional layer in the model
            target_layer = None
            for layer in reversed(autoencoder.layers):  # iterate from last to first
                if isinstance(layer, tf.keras.layers.Bidirectional):
                    target_layer = layer
                    break  # stop at the first one found (the last in the model)

            # Check if it exists
            if not target_layer:
                raise ValueError("No Bidirectional layer found in the model.")

            # Print its name for verification
            print("Last Bidirectional layer name:", target_layer.name)

        # Build the encoder model up to that layer
        encoder = tf.keras.models.Model(inputs=autoencoder.input, outputs=target_layer.output)

        # Build the finetune model
        model = build_finetune_model(encoder, activation, n_dense, mode)

        #model = build_finetune_model_frozen(encoder, n_classes=n_classes)
        #model = build_finetune_model_selective(
                        #     encoder,
                        #     n_classes=3,
                        #     unfreeze_layers=['bidirectional_16', 'bidirectional_17', 'multi_head_attention_3']
                        # )


        
        counts = Counter(y_train)
        total = sum(counts.values())
        n_classes = len(np.unique(y_train))
        print("Number of classes:", n_classes)
        alpha = [ total / (n_classes * counts[i]) for i in range(n_classes) ]

        steps_per_epoch = len(X_train) // n_batch
        decay_steps = steps_per_epoch * 300

        # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        #     initial_learning_rate=1e-3,  # tasa inicial
        #     decay_steps=decay_steps,
        #     alpha=1e-5 / 1e-3            # valor mínimo = 1e-5
        # )

        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, verbose=0, min_lr=1e-6
        ) 
        
        # Compile the model (using sparse categorical crossentropy for class labels)
        model.compile(loss=CategoricalFocalCrossentropy(gamma = 3.0, alpha=alpha),
                            #loss = categorical_focal_loss,
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # learning_rate=lr_schedule
                            metrics=[f1_score_weighted, acc_type]) #sparse_categorical_accuracy

        callbacks = [
        balanced_accuracy_callback,
        checkpoint_cb,  # weights_cb,
        chkpoint,
        earlyStopping,
        lr_callback,
        lrchangelog
        ]
                    
        # One-hot encode y_train and y_val with the correct number of classes
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
        y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes=n_classes)
        
        n_timesteps = 125
        n_classes = 3

        # Train the model (encoder + classifier)
        history = model.fit(X_train, y_train_one_hot, 
                            batch_size=n_batch, 
                            epochs=n_epochs,
                            callbacks=callbacks,
                            validation_data=(X_val, y_val_one_hot),
                            verbose=0)

        # # Train the model (encoder + classifier)
        # history=model.fit(X_train, one_hot(y_train, n_classes), 
        #             batch_size=n_batch, 
        #             epochs=n_epochs,
        #             callbacks = callbacks,
        #             validation_data=(X_val, one_hot(y_val, n_classes)),
        #             verbose = 0)

        # Load the saved model
        model = tf.keras.models.load_model(path_best_model, custom_objects={"T2V": T2V, 
                                                                    "SensorAttention": SensorAttention, 
                                                                    "PositionalEncoding": PositionalEncoding,
                                                                    'WaveNetResidualConv1D': WaveNetResidualConv1D,
                                                                    'BalancedAccuracyCallback': BalancedAccuracyCallback,
                                                                    'balanced_accuracy': utils.balanced_accuracy,
                                                                    'f1_score_weighted':f1_score_weighted})


    else:
        print("\nTraining the model from scratch...")
        print("No code here ...")
        
    print()        
    print(" Directory:", directory)
    print("Subdirectory",subdirectory)
        
       
    # summarize history for accuracy complete
    print("History:")
    print(history.history.keys())
    plt.figure()
    plt.plot(history.history[acc_type], 'r--')
    plt.plot(history.history[val_type], 'b--')
    plt.title(dataset + '- loss and accuracy')
    # summarize history for loss
    plt.plot(history.history['loss'], 'm-')
    plt.plot(history.history['val_loss'], 'c-')
    plt.ylabel('loss & accuracy (--)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy', 'train loss', 'val loss'], loc='upper right')
    plt.savefig(subdirectory+"/loss-acc_comp"+dataset+"-{}".format(time_)+'.png')


    # summarize history for accuracy with limit in axis y 
    plt.figure()
    plt.ylim(0, 1.0)
    # plt.plot(history.history['balanced_accuracy'], 'r--')
    # plt.plot(history.history['val_balanced_accuracy'], 'b--')
    plt.plot(balanced_accuracy_callback.train_bal_acc, 'r--')
    plt.plot(balanced_accuracy_callback.val_bal_acc, 'b--')
    plt.title(dataset + '- balanced accuracy')
    plt.ylabel('balanced accuracy (--)')
    plt.xlabel('epoch')
    plt.legend(['train bal-accuracy', 'val bal-accuracy'], loc='upper right')
    plt.savefig(subdirectory+"/balacc_"+dataset+"-{}".format(time_)+'.png')


    # summarize history for accuracy with limit in axis y 
    plt.figure()
    plt.ylim(0, 2.0)
    # plt.plot(history.history['balanced_accuracy'], 'r--')
    # plt.plot(history.history['val_balanced_accuracy'], 'b--')
    plt.plot(balanced_accuracy_callback.train_bal_acc, 'r--')
    plt.plot(balanced_accuracy_callback.val_bal_acc, 'b--')
    plt.title(dataset + '- loss and balanced accuracy')
    # summarize history for loss
    plt.plot(history.history['loss'], 'm-')
    plt.plot(history.history['val_loss'], 'c-')
    plt.ylabel('loss & balanced accuracy (--)')
    plt.xlabel('epoch')
    plt.legend(['train bal-accuracy', 'val bal-accuracy', 'train loss', 'val loss'], loc='upper right')
    plt.savefig(subdirectory+"/loss-balacc_"+dataset+"-{}".format(time_)+'.png')

    # Plot for history for accuracy
    plt.figure()
    plt.ylim(0, 1)
    plt.title(dataset + ' - accuracy ')
    plt.plot(history.history[acc_type], 'r--')
    plt.plot(history.history[val_type], 'b--')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper right')
    plt.savefig(subdirectory+"/acc_"+dataset+"-{}".format(time_)+'.png')

    # Plot for history for balanced accuracy
    plt.figure()
    plt.ylim(0, 1)
    plt.title(dataset + ' - accuracy ')
    plt.plot(history.history[acc_type], 'r--')
    plt.plot(history.history[val_type], 'b--')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper right')
    plt.savefig(subdirectory+"/acc_"+dataset+"-{}".format(time_)+'.png')


    # Plot for history for loss
    plt.figure()  
    plt.ylim(0, 1.5)
    plt.title(dataset + ' - loss ')
    plt.plot(history.history['loss'], 'm-')
    plt.plot(history.history['val_loss'], 'c-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper right')
    plt.savefig(subdirectory+"/loss_"+dataset+"-{}".format(time_)+'.png')


    # Compute all metrics
    metric_results = [[], [], []]
    print()
    print("Train:")
    predictions_train = model.predict(X_train)
    utils.compute_all_metrics(y_train, predictions_train, dataset, time_, subdirectory, metric_results[2], split='train')
    print()
    print("Val:")
    predictions_val = model.predict(X_val)
    utils.compute_all_metrics(y_val, predictions_val, dataset, time_, subdirectory, metric_results[1], split='val')
    print()
    print("Test:")
    predictions_test = model.predict(X_test)
    utils.compute_all_metrics(y_test, predictions_test, dataset, time_, subdirectory, metric_results[0], split='test')


    # Compute metrics
    metric_results = build_metrics(model, dict_arrays)
    
    new_row = { 'time' : time_,
                'modeltype' : modeltype,
                'dataset': dataset,
                'uuid_val': "",
                'uuid_test': "",
                'n_epochs': n_epochs, #last_epoch_early_stopping, 
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
                'path': ""
            }

	# [acc_test, bal_acc_test, f1_test, k_test]
    scores.append([metric_results[2][0], metric_results[2][1], metric_results[2][2], metric_results[2][4]])
    print(new_row)
    table = pd.concat([table, pd.DataFrame([new_row])], ignore_index=True)  

    table_name = subdirectory+"/table_"+dataset+"_"+time_+".csv"
    print("Saving table in...", table_name)
    table.to_csv(table_name, sep=',', encoding='utf-8', index=False)

    # SAVE THE MODEL ?
    if saveModel:        
        # Saving the model
        modelname = subdirectory+"/model-{}".format(time_)
        # serialize model to JSON
        print("\nSaving the model ...")
        model_json = model.to_json()
        with open(modelname+".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        #model.save_weights(modelname+".weights.h5")
        model.save(modelname+".keras")
        with open(modelname+obs+'-.txt', 'w') as file:
            file.write("Dataset: {} \n".format(dataset))
            file.write("# of Epochs: {} \n".format(n_epochs))
            file.write("Learning Rate: {} \n".format(learning_rate))
            file.write("Clipvalue: {} \n".format(clipvalue))
            file.write("Dropout Rate: {} \n".format(dropout_rate))
            file.write("Batch Size: {} \n".format(n_batch))
            file.write("LSTM_layers: {} \n".format(LSTM_layers))
            file.write("LSTM hidden Units: {} \n".format(lstm_hidden_units))
            file.write("Fully Connected Layer Units: {} \n".format(fconn_units))
            file.write("LSTM Regularization Coefficient: {} \n".format(lstm_reg))
            file.write("Classification Regularization Coefficient : {} \n".format(clf_reg))
            file.write("Verbose : {} \n".format(verbose))         
            file.write("Train Classification Accuracy: {} \n".format(history.history['accuracy'][-1]))
            file.write("Test Classification Accuracy: {} \n".format(history.history['val_accuracy'][-1]))
        print("Model saved !")

    else:
        print("Model not saved.")   
        
    return table


# Main function
# def main():
#     # Table of results
#     table = pd.DataFrame(columns=['time', 'dataset', 'n_epochs', 'lr','do','batch', 'layers', 
#                                         'h_units', 'lstm_reg', 'clf_reg', 'clipvalue', 'train', 'train_bal', 
#                                         'f1_score_train_we', 'kappa_train',
#                                         'val', 'val_bal', 'f1_score_val_we', 'kappa_val',
#                                         'test', 'test_bal', 'f1_score_test_we', 'kappa_test', 'obs'])
    
#     # Variables
#     #model_type_list = ['wavebigru', 'atteBigru', 'transbigru', 'bigru', 'attnbigru'] 
#     model_type_list = ['autoencoder_s'] #['transbigru', 'bigru', 'bilstm'] #['attnbigru']
#     test_type = ['nusers']   # 'oneuser'
#     k_folds = [1]
#     norm_method_list = [0]
#     n_epochs = [50] # [300] 
#     dataset_list = ['eatdrinkanother_94u_10'] #deo_drinkeat_94u_train
#     # fullraws3_vivabem012_drink0eat1another2_aug_acc_gyr_mag__coord__5sec_100hz.csv
#     #dataset_list = ['mhealth-agm-cub']

#     learning_rate = [1e-5]
#     dropout_rate = [0.5]
#     overlap_shift = [0.5]
#     n_batch = [256] #[1024] #[512]
#     LSTM_layers = [-1]  
#     sensors = [3]
#     seg5 = [True]

#     overlap = False
#     normalize = True
#     ftune = True
#     testing = False

#     #finetune_model = "saved_models/recurrent_models_20250823-152320/deo_drinkeat_94u_train_1f_autoencoder_20250823-152350/"
#     #finetune_model = finetune_model + "best_model_deo_drinkeat_94u_train_20250823-152350.keras"

#     # (4)
#     #finetune_model = "saved_models/recurrent_models_20251011-123315/de_fake_padts_94u_1f_autoencoder_s_20251011-123937/"
#     #finetune_model = finetune_model + "best_model_de_fake_padts_94u_20251011-123937.keras"

#     # (13a)
#     finetune_model = "saved_models/recurrent_models_20251012-100533/de_fake_padts_94u_1f_autoencoder_s_20251012-101512/"
#     finetune_model = finetune_model + "best_model_de_fake_padts_94u_20251012-101512.keras"

#     # (14) de_d_e_fake_100-9
#     # finetune_model = 
    
#     print("Pre-trained model:", finetune_model)
#     print()
#     # Hyperparameters
#     a = [model_type_list, test_type, k_folds, norm_method_list, dataset_list, 
#                        learning_rate, dropout_rate, overlap_shift, n_batch, 
#                        n_epochs, LSTM_layers, sensors, seg5]
#     combs = list(itertools.product(*a))
    

#     # Directory
#     time_ = time.strftime("%Y%m%d-%H%M%S")
#     directory = os.getcwd() + '/saved_models/' + 'recurrent_models_'+ time_

#     if not os.path.exists(directory):
#         os.makedirs(directory)
    
#     # Base_path
#     base_path = '/home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/'

#     # Save file
#     source_file = base_path + 'Recurrence/recurrent_models_cab_reversioned_100625_copy.py'
#     destination_folder = directory+"/recurrent_models_cab_reversioned_100625_copy.py"
#     print("Saving the code at: ", destination_folder)
#     shutil.copy(source_file, destination_folder)
    
#     # Save utils    
#     source_file = base_path + 'Recurrence/utils.py'
#     destination_folder = directory+"/utils.py"
#     print("Saving the code at: ", destination_folder)
#     shutil.copy(source_file, destination_folder)

#     # Save models
#     source_file = base_path + 'Recurrence/models.py'
#     destination_folder = directory+"/models.py"
#     print("Saving the code at: ", destination_folder)
#     shutil.copy(source_file, destination_folder)



#     dataset = dataset_list[0]

#     _, _, _, file_full_raws = utils.get_raw_datasets(dataset, magni=False)
#     print(file_full_raws)

#     if ( dataset == 'vivabem12_lying' ):
#         df = utils.read_full_raws_without_tv(file_full_raws)
#     elif ( dataset == 'vivabem12_tv' ):
#         df = utils.read_full_raws_without_lying(file_full_raws)
#     else:
#         df = utils.read_full_raws(file_full_raws, dataset)


#     #with tf.device('/gpu:'+device):
#     for comb in combs:
        
#         modeltype = comb[0]
#         test_type = comb[1]
#         k_folds = comb[2]
#         norm_method = comb[3]
#         dataset = comb[4]
        
#         learning_rate = comb[5]
#         dropout_rate = comb[6]
#         overlap_shift = comb[7]
#         n_batch = comb[8]
#         n_epochs = comb[9]
#         LSTM_layers = comb[10]
#         sensors = comb[11]
#         seg5 = comb[12]

#         hyperparams = modeltype+"_"+test_type+"_"+str(k_folds)+"_"+str(overlap_shift)+"_"+str(norm_method)
#         hyperparams = hyperparams+"_"+ dataset+"_"+str(learning_rate)+"_"+str(dropout_rate)
#         hyperparams = hyperparams+"_"+str(n_batch)+"_"+str(n_epochs)+"_"+str(LSTM_layers)+"_"+str(sensors)

#         print("Comb: ", comb)

#         obs = ''
        
#         seed = 30
#         tf.random.set_seed(seed)
#         np.random.seed(seed)


#         # Save df
#         # df_name = 'df_'+dataset+'.csv'
#         # print("Saving df in...", df_name)
#         # df.to_csv(df_name, sep=',', encoding='utf-8', index=False)
#         # print("df shape: ", df.shape)

#         scores = []
#         cms = []
        
#         # Get the class names from dataframe
#         LABELS = utils.get_class_names(dataset)
# #         if (df.iloc[0,4] != 'met' & not (dataset.startswith('eatdrinkanother'))):
# #             df_activities = df.groupby([2, 4]).size().reset_index(name='Count')
# #             LABELS = np.unique(df_activities[4])
                    
#         fold = 1
#         time_ = time.strftime("%Y%m%d-%H%M%S") 
#         subdirectory = directory + '/' + dataset + '_'+str(fold)+'f_'+modeltype + '_'+time_
#         print("subdirectory: ", subdirectory)
#         print("directory: ", directory)
#         if not os.path.exists(subdirectory):
#                 os.makedirs(subdirectory)


#         # Define file path
#         file_path = os.path.join(subdirectory, "hyperparams.txt")

#         # Save string to file
#         with open(file_path, "w") as f:
#             f.write(hyperparams)

#         plots_loss_dir = os.path.join(subdirectory,'plots_loss_dir')
#         if not os.path.exists(plots_loss_dir):
#             os.makedirs(plots_loss_dir)

#         # get_processed_fold_valtestuser, get_processed_fold_testuser, get_processed_fold
#         dict_arrays = utils.get_processed_fold(df, dataset, modeltype, subdirectory, sensors, fold,
#                                                 seg5, normalize, overlap, overlap_shift=overlap_shift)
        
              
#         # clfOnlyTowersTest has only GRU model
#         table = clfOnlyTowers(directory, subdirectory, finetune_model, plots_loss_dir, dict_arrays, obs, table, 
#                             time_, dataset, n_epochs, scores,
#                             n_batch, cms, LABELS, testing=testing, ftune=ftune, 
#                             modelname=modeltype, modeltype=modeltype, saveModel=True )

#         print("Model type:", modeltype)
#         print(combs)



        

#         # table.at[len(table)-1,'ov'] = str(overlap_shift)

#         # print("End of the fold: "+str(fold))
#         # print(table)
#         # print("End of the training in dataset: "+dataset)

#         # # COMPUTE THE AVERAGE OF ALL THE LISTS OF SCORES
#         # scores = np.array(scores)
#         # scores = scores.mean(axis=0)
#         # scores = scores.tolist()
#         # #table = table.append( table.iloc[-1] )
#         # table = pd.concat([table, table.iloc[[-1]]], ignore_index=True)
#         # table.reset_index(drop=True, inplace=True)
#         # table.at[len(table)-1,'test'] = scores[0]
#         # table.at[len(table  )-1,'test_bal'] = scores[1]
#         # table.at[len(table)-1,'f1_score_test_we'] = scores[2]
#         # table.at[len(table)-1,'kappa_test'] = scores[3]
#         # table.at[len(table)-1,'uuid_val'] = '-'
#         # table.at[len(table)-1,'uuid_test'] = 'avg'
#         # table = table.round(3)

#         # table_name = directory + "/table_"+dataset+".csv"
#         # print("Saving table in...", table_name)
#         # table.to_csv(table_name, sep=',', encoding='utf-8', index=False)
#         # print(table)
#         # print("Model name: ", modeltype)
#         # print("Directory: ", directory)    

#         # endtime_ = time.strftime("%Y%m%d-%H%M%S")
#         # print("The program finished at: ", endtime_) 

    
#     # MAIN
#     time_ = time.strftime("%Y%m%d-%H%M%S")
#     print("End of the training at: ", time_)

#   main()

# In[9]: