import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, GRU, LayerNormalization, MultiHeadAttention
from tensorflow.keras import layers

class ModelBuilder:
    """
    A class for building neural network models for HAR tasks.
    """
    def __init__(self, modeltype, input_shape, n_classes, hyperparameters):
        """
        Initialize the ModelBuilder.

        Args:
            modeltype (str): Type of model ('attnbigru', etc.)
            input_shape (tuple): Shape of input data (timesteps, features)
            n_classes (int): Number of output classes
            hyperparameters (dict): Dictionary of hyperparameters
        """
        self.modeltype = modeltype
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.hyperparameters = hyperparameters

    def build(self):
        """
        Build the neural network model based on the model type.

        Returns:
            tf.keras.Model: The constructed model
        """
        dropout_rate = self.hyperparameters.get('dropout_rate', 0.5)
        lstm_hidden_units = self.hyperparameters.get('lstm_hidden_units', 100)
        lstm_reg = self.hyperparameters.get('lstm_reg', 1e-4)
        clf_reg = self.hyperparameters.get('clf_reg', 1e-4)

        # Raw Input
        raw_inputs = Input(shape=self.input_shape)

        if self.modeltype == 'attnbigru':
            ff_dim = 9
            fconn_units = 100

            # First Convolutional Layer
            xlstm = layers.Conv1D(filters=ff_dim, kernel_size=1)(raw_inputs)
            xlstm = layers.MaxPooling1D(pool_size=2)(xlstm)
            xlstm = Dense(fconn_units)(xlstm)
            xlstm = layers.Dropout(dropout_rate)(xlstm)

            # Second Convolutional Layer
            xlstm = layers.Conv1D(filters=ff_dim, kernel_size=1)(xlstm)
            xlstm = layers.MaxPooling1D(pool_size=2)(xlstm)
            xlstm = Dense(fconn_units)(xlstm)
            xlstm = layers.Dropout(dropout_rate)(xlstm)

            # Recurrent Layers
            xlstm = Bidirectional(GRU(lstm_hidden_units, return_sequences=True,
                            kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            activity_regularizer=tf.keras.regularizers.l1(lstm_reg)))(xlstm)
            xlstm = Dropout(dropout_rate)(xlstm)

            xlstm = Bidirectional(GRU(lstm_hidden_units, return_sequences=True,
                            kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            activity_regularizer=tf.keras.regularizers.l1(lstm_reg)))(xlstm)
            xlstm = Dropout(dropout_rate)(xlstm)

            # Add Self-Attention Layer
            attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(xlstm, xlstm)
            attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
            attention_output = Dropout(dropout_rate)(attention_output)

            # Residual connection
            xlstm = layers.Add()([xlstm, attention_output])
            xlstm = LayerNormalization(epsilon=1e-6)(xlstm)

            # Final GRU Layer
            xlstm = Bidirectional(GRU(lstm_hidden_units, return_sequences=False,
                            kernel_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            recurrent_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            bias_regularizer=tf.keras.regularizers.l2(lstm_reg),
                            activity_regularizer=tf.keras.regularizers.l1(lstm_reg)))(xlstm)
            xlstm = Dropout(dropout_rate)(xlstm)

        # Dense output layer
        class_predictions = Dense(self.n_classes, activation='softmax',
                    kernel_regularizer=tf.keras.regularizers.l2(clf_reg),
                    bias_regularizer=tf.keras.regularizers.l2(clf_reg),
                    activity_regularizer=tf.keras.regularizers.l1(clf_reg),
                    name='class_output')(xlstm)

        # Full model
        model = Model(inputs=raw_inputs, outputs=class_predictions)

        return model