import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2, l1

"""
Autoencoder
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

setup_seed(30)

"""
Fine tuning model
"""
def build_finetune_model(encoder, activation='gelu', n_dense=128, mode='balanced', n_classes=3, dropout_rate=0.3):
    inputs = layers.Input(shape=encoder.input_shape[1:])
    x = encoder(inputs)

    if mode == 'balanced':
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    elif mode == 'light':
        x = layers.Dense(n_dense, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    elif mode == 'deep':
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='silu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='gelu')(x)
        x = layers.Dropout(0.4)(x)
    elif mode == 'classic':
        x = layers.Dense(100, activation=activation)(x)

    x = layers.Dense(n_classes, activation='softmax')(x)

    full_model = models.Model(inputs=inputs, outputs=x)

    # Unfreeze all layers (optional — can freeze encoder later)
    for layer in full_model.layers:
        layer.trainable = True

    print(f"\n Fine-tuning configuration: mode={mode}, activation={activation}, dense={n_dense}")
    #full_model.summary()
    return full_model


"""
Fine tuning model
"""
def build_finetune_model_frozen(encoder, n_classes=3):
    """
    Builds a fine-tuning model that freezes the encoder and
    trains only the Dense classification layers.

    Args:
        encoder: Pre-trained encoder model (e.g., up to the last Bidirectional layer)
        n_classes: Number of output classes (default = 3)

    Returns:
        full_model: Combined encoder + classifier model
    """
    print("\nBuilding fine-tune model with frozen encoder...\n")

    # Freeze all layers in the encoder
    encoder.trainable = False

    # Define input with the same shape as the encoder
    inputs = layers.Input(shape=encoder.input_shape[1:])

    # Pass through the encoder (feature extractor)
    x = encoder(inputs)

    # Add classification head (trainable)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(n_classes, activation='softmax')(x)

    # Build the full model
    full_model = models.Model(inputs=inputs, outputs=x)

    # Compile with a standard optimizer and loss
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print summary (optional)
    full_model.summary()

    print("\n Fine-tuning configuration: Encoder fully frozen.\n")

    return full_model


import tensorflow as tf
from tensorflow.keras import layers, models

def build_finetune_model_selective(encoder, n_classes=3, unfreeze_layers=None):
    """
    Builds a fine-tuning model that allows selective unfreezing
    of specific encoder layers for fine-tuning.

    Args:
        encoder: Pre-trained encoder model
        n_classes: Number of output classes (default = 3)
        unfreeze_layers: list of layer name substrings or indexes to unfreeze
                         (e.g. ['bidirectional_16', 'bidirectional_17'] or [45, 46])

    Returns:
        full_model: Combined encoder + classifier model ready for training
    """
    # --- Step 1: Freeze all encoder layers ---
    for layer in encoder.layers:
        layer.trainable = False

    # --- Step 2: Unfreeze chosen layers (by name or index) ---
    if unfreeze_layers is not None:
        for layer in encoder.layers:
            # Unfreeze if layer name matches any substring in the list
            if isinstance(unfreeze_layers[0], str):
                if any(name in layer.name for name in unfreeze_layers):
                    layer.trainable = True
            # Or if list contains indices
            elif isinstance(unfreeze_layers[0], int):
                idx = encoder.layers.index(layer)
                if idx in unfreeze_layers:
                    layer.trainable = True

    # --- Step 3: Build the classification head ---
    inputs = layers.Input(shape=encoder.input_shape[1:])
    x = encoder(inputs)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(n_classes, activation='softmax')(x)

    # --- Step 4: Create full model ---
    full_model = models.Model(inputs=inputs, outputs=x)

    # --- Step 5: Compile ---
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print summary for confirmation
    print("\n🧩 Fine-tuning configuration:")
    print(f"Total encoder layers: {len(encoder.layers)}")
    unfrozen = [l.name for l in encoder.layers if l.trainable]
    print(f"Unfrozen encoder layers: {unfrozen if unfrozen else 'None (encoder fully frozen)'}\n")

    return full_model



"""
Build symmetric CABIGRU Autoencoder using Keras.
"""
def build_autoencoder_s(raw_inputs,
                        input_shape=(500, 9),
                      dropout_rate=0.3,
                      lstm_hidden_units=64,
                      ff_dim=9,
                      fconn_units=100,
                      lstm_reg=1e-4):

    # ----- Encoder -----
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, padding="same")(raw_inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dense(fconn_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dense(fconn_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=True,
                                        kernel_regularizer=regularizers.l2(lstm_reg)))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=True,
                                        kernel_regularizer=regularizers.l2(lstm_reg)))(x)
    x = layers.Dropout(dropout_rate)(x)

    att = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    att = layers.LayerNormalization(epsilon=1e-6)(att)
    att = layers.Dropout(dropout_rate)(att)
    x = layers.Add()([x, att])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    encoded = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=False,
                                              kernel_regularizer=regularizers.l2(lstm_reg)))(x)
    encoded = layers.Dropout(dropout_rate)(encoded)

    # ----- Decoder -----
    d = layers.RepeatVector(125)(encoded)

    d = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=True))(d)
    d = layers.Dropout(dropout_rate)(d)

    d = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=True))(d)
    d = layers.Dropout(dropout_rate)(d)

    d = layers.Conv1D(ff_dim, kernel_size=1, padding="same")(d)
    d = layers.UpSampling1D(size=2)(d)

    d = layers.Conv1D(ff_dim, kernel_size=1, padding="same")(d)
    d = layers.UpSampling1D(size=2)(d)

    decoder_output = layers.TimeDistributed(layers.Dense(9, activation="sigmoid"))(d)

    # ----- Model -----
    autoencoder = models.Model(inputs=raw_inputs, outputs=decoder_output)
    return autoencoder



"""
CABIGRU with Attention Mechanism. Torch.
"""
class AttnBigRU(nn.Module):
    def __init__(self, input_dim, lstm_hidden_units, lstm_reg, dropout_rate, ff_dim, fconn_units, num_heads=8, num_transformer_blocks=1):
        super(AttnBigRU, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=ff_dim, kernel_size=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(ff_dim, fconn_units)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=fconn_units, out_channels=ff_dim, kernel_size=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc2 = nn.Linear(ff_dim, fconn_units)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Recurrent Layers (Bidirectional GRU)
        self.gru1 = nn.GRU(input_size=fconn_units, hidden_size=lstm_hidden_units, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.gru2 = nn.GRU(input_size=2 * lstm_hidden_units, hidden_size=lstm_hidden_units, batch_first=True, bidirectional=True)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Multi-Head Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_units * 2, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(lstm_hidden_units * 2)
        self.dropout5 = nn.Dropout(dropout_rate)

        # Final GRU Layer (Bidirectional)
        self.gru3 = nn.GRU(input_size=lstm_hidden_units * 2, hidden_size=lstm_hidden_units, batch_first=True, bidirectional=True)
        self.dropout6 = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(2 * lstm_hidden_units, 1)  # Assuming classification with one output, adjust as needed

    def forward(self, x):
        # Convolutional Layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = x.permute(0, 2, 1) 
        x = self.fc1(x)
        x = x.permute(0, 2, 1)  # Back to (batch_size, seq_len, features)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = x.permute(0, 2, 1) 
        x = self.fc2(x)
        x = x.permute(0, 2, 1)   # Back to (batch_size, seq_len, features)
        x = F.relu(x)
        x = self.dropout2(x)

        # Recurrent Layers (GRU)
        x = x.permute(0, 2, 1)
        x, _ = self.gru1(x)
        x = x.permute(0, 2, 1)
        x = self.dropout3(x)

        x, _ = self.gru2(x)
        x = self.dropout4(x)

        # Multi-Head Attention Layer
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.layer_norm(attn_output)
        attn_output = self.dropout5(attn_output)

        # Residual Connection: Add attention output with original GRU output
        x = x + attn_output
        x = self.layer_norm(x)

        # Final GRU Layer
        x, _ = self.gru3(x)
        x = self.dropout6(x)

        # Output Layer (assuming classification)
        out = self.fc_out(x[:, -1, :])  # Only use the last output for classification
        return out
    

"""
CABIGRU Autoencoder. Torch.
"""
class AutoencoderModel(nn.Module):
    def __init__(self, input_shape, ff_dim, fconn_units, lstm_hidden_units, dropout_rate):
        super(AutoencoderModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_shape[1], ff_dim, kernel_size=1)       # (batch_size, channels, length)
        self.pool1 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(ff_dim * (input_shape[0] // 2), fconn_units)  # Adjust for the output size after pooling
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(ff_dim, ff_dim, kernel_size=1)
        self.pool2 = nn.MaxPool1d(2)
        self.fc2 = nn.Linear(ff_dim * (input_shape[0] // 4), fconn_units)  # Adjust for the output size after pooling
        self.dropout2 = nn.Dropout(dropout_rate)

        # Bidirectional GRU layers
        self.gru1 = nn.GRU(fconn_units, lstm_hidden_units, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.gru2 = nn.GRU(lstm_hidden_units * 2, lstm_hidden_units, batch_first=True, bidirectional=True)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=lstm_hidden_units * 2, num_heads=4)
        self.layer_norm1 = nn.LayerNorm(lstm_hidden_units * 2)
        self.dropout5 = nn.Dropout(dropout_rate)
        
        # Decoder
        self.fc3 = nn.Linear(lstm_hidden_units * 2, fconn_units)
        self.fc4 = nn.Linear(fconn_units, input_shape[0] * input_shape[1])
        self.decoder = nn.Sequential(
            nn.Linear(fconn_units, input_shape[0] * input_shape[1]),
            nn.Sigmoid()
        )

    def forward(self, x):
        # First Convolutional Layer
        x = self.pool1(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.dropout1(x)

        x = torch.flatten(x, start_dim=1)
        # Second Convolutional Layer
        x = self.pool2(self.conv2(x.unsqueeze(2)))  # Adding a channel dimension for Conv1d
        x = self.fc2(x)
        x = self.dropout2(x)

        # GRU Layers
        x, _ = self.gru1(x.unsqueeze(1))  # Adding a batch dimension
        x = self.dropout3(x)
        
        x, _ = self.gru2(x)
        x = self.dropout4(x)

        # Attention Layer
        x, _ = self.attention(x, x, x)
        x = self.layer_norm1(x)
        x = self.dropout5(x)

        # Decoder
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(-1, 500, 9)  # Reshape to original input shape (500, 9)

        return x