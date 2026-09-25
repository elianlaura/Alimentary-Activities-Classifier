import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
import os

def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Custom focal loss function.
    """
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    focal_loss = weight * cross_entropy
    return tf.reduce_sum(focal_loss, axis=-1)

def create_callbacks(path_best_model):
    """
    Create training callbacks.

    Args:
        path_best_model (str): Path to save best model

    Returns:
        list: List of callbacks
    """
    class CustomModelCheckPoint(tf.keras.callbacks.Callback):
        def __init__(self, **kwargs):
            super(CustomModelCheckPoint, self).__init__(**kwargs)
            self.epoch_accuracy = {}
            self.epoch_loss = {}
            self.epoch_validation = {}
            self.epoch_lossval = {}

        def on_epoch_end(self, epoch, logs={}):
            self.epoch_accuracy[epoch] = logs.get("accuracy")
            self.epoch_loss[epoch] = logs.get("loss")
            self.epoch_validation[epoch] = logs.get("val_accuracy")
            self.epoch_lossval[epoch] = logs.get("val_loss")

    chkpoint = CustomModelCheckPoint()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            path_best_model, save_best_only=True, monitor='val_accuracy'
        ),
        chkpoint,
    ]

    return callbacks

def compile_model(model, learning_rate, clipvalue):
    """
    Compile the model with optimizer and loss.

    Args:
        model: Keras model
        learning_rate (float): Learning rate
        clipvalue (float): Gradient clipping value
    """
    model.compile(
        loss=categorical_focal_loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=clipvalue),
        metrics=['accuracy']
    )

def save_model_and_metadata(model, subdirectory, time_, dataset, obs, hyperparameters, history):
    """
    Save model and metadata.

    Args:
        model: Trained model
        subdirectory (str): Save directory
        time_ (str): Timestamp
        dataset (str): Dataset name
        obs (str): Observations
        hyperparameters (dict): Hyperparameters
        history: Training history
    """
    modelname = f"{subdirectory}/model-{time_}"

    # Serialize model to JSON
    model_json = model.to_json()
    with open(f"{modelname}.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    model.save_weights(f"{modelname}.h5")

    # Save hyperparameters and results
    with open(f"{modelname}{obs}-.txt", 'w') as file:
        file.write(f"Dataset: {dataset} \n")
        file.write(f"# of Epochs: {hyperparameters.get('n_epochs', 'N/A')} \n")
        file.write(f"Learning Rate: {hyperparameters.get('learning_rate', 'N/A')} \n")
        file.write(f"Clipvalue: {hyperparameters.get('clipvalue', 'N/A')} \n")
        file.write(f"Dropout Rate: {hyperparameters.get('dropout_rate', 'N/A')} \n")
        file.write(f"Batch Size: {hyperparameters.get('n_batch', 'N/A')} \n")
        file.write(f"LSTM_layers: {hyperparameters.get('LSTM_layers', 'N/A')} \n")
        file.write(f"LSTM hidden Units: {hyperparameters.get('lstm_hidden_units', 'N/A')} \n")
        file.write(f"Fully Connected Layer Units: {hyperparameters.get('fconn_units', 'N/A')} \n")
        file.write(f"LSTM Regularization Coefficient: {hyperparameters.get('lstm_reg', 'N/A')} \n")
        file.write(f"Classification Regularization Coefficient: {hyperparameters.get('clf_reg', 'N/A')} \n")
        file.write(f"Verbose: {hyperparameters.get('verbose', 'N/A')} \n")
        file.write(f"Train Classification Accuracy: {history.history['accuracy'][-1]} \n")
        file.write(f"Test Classification Accuracy: {history.history['val_accuracy'][-1]} \n")

def load_best_model(path_best_model):
    """
    Load the best saved model.

    Args:
        path_best_model (str): Path to best model

    Returns:
        Loaded model
    """
    with custom_object_scope({
        'categorical_focal_loss': categorical_focal_loss,
    }):
        model = tf.keras.models.load_model(path_best_model)
    return model

def save_history(history, subdirectory, dataset, modelname, time_):
    """
    Save training history to CSV.

    Args:
        history: Training history
        subdirectory (str): Save directory
        dataset (str): Dataset name
        modelname (str): Model name
        time_ (str): Timestamp
    """
    hist_file = f"{subdirectory}/history_{dataset}_{modelname}_{time_}.csv"
    hist_df = pd.DataFrame(history.history)
    with open(hist_file, mode='w') as f:
        hist_df.to_csv(f)