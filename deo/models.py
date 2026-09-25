"""CABiGRU autoencoder (stage 1) and classifier (stage 2).

Architectures are copied from the code archived with the paper's runs
(saved_models_prev/.../recurrent_models_20251026-024834/models.py):

* build_autoencoder_s  - symmetric Conv1D + BiGRU + multi-head attention autoencoder.
  Stage-1 objective: reconstruct the (normalised) input window with an MSE loss.
  This reconstruction task is the "self-supervised pretraining" of the paper.
* encoder_from_autoencoder - the encoder is the autoencoder truncated at its third
  Bidirectional layer counted from the end (the last encoder BiGRU, 128-d output).
* build_classifier - encoder + classification head (legacy build_finetune_model).
  NOTE: in the 'balanced' and 'deep' heads the activations are hard-coded, so the
  --activation argument only affects the 'light' and 'classic' heads.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

HEADS = ("balanced", "light", "deep", "classic")


def build_autoencoder_s(input_shape=(500, 9), dropout_rate=0.3, lstm_hidden_units=64, ff_dim=9,
                        fconn_units=100, lstm_reg=1e-4):
    raw_inputs = layers.Input(shape=input_shape)
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
    d = layers.RepeatVector(input_shape[0] // 4)(encoded)
    d = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=True))(d)
    d = layers.Dropout(dropout_rate)(d)
    d = layers.Bidirectional(layers.GRU(lstm_hidden_units, return_sequences=True))(d)
    d = layers.Dropout(dropout_rate)(d)
    d = layers.Conv1D(ff_dim, kernel_size=1, padding="same")(d)
    d = layers.UpSampling1D(size=2)(d)
    d = layers.Conv1D(ff_dim, kernel_size=1, padding="same")(d)
    d = layers.UpSampling1D(size=2)(d)
    # Linear output: the corrected protocol reconstructs z-scored windows, which are
    # not bounded to [0, 1]. The legacy sigmoid is available with output="sigmoid".
    out = layers.TimeDistributed(layers.Dense(input_shape[1]))(d)
    return models.Model(inputs=raw_inputs, outputs=out, name="cabigru_autoencoder")


def build_autoencoder_legacy(input_shape=(500, 9)):
    """Exact legacy variant (sigmoid output), for reproducing the archived pretraining."""
    ae = build_autoencoder_s(input_shape)
    x = ae.layers[-2].output
    out = layers.TimeDistributed(layers.Dense(input_shape[1], activation="sigmoid"))(x)
    return models.Model(inputs=ae.input, outputs=out, name="cabigru_autoencoder_legacy")


def encoder_from_autoencoder(autoencoder):
    bidirectional = [l for l in autoencoder.layers if isinstance(l, layers.Bidirectional)]
    if len(bidirectional) < 3:
        raise ValueError("autoencoder has %d Bidirectional layers, need >= 3" % len(bidirectional))
    target = bidirectional[-3]
    return models.Model(inputs=autoencoder.input, outputs=target.output, name="cabigru_encoder")


def build_classifier(encoder, head="deep", activation="gelu", n_dense=200, n_classes=3, dropout_rate=0.3):
    inputs = layers.Input(shape=encoder.input_shape[1:])
    x = encoder(inputs)
    if head == "balanced":
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="gelu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
    elif head == "light":
        x = layers.Dense(n_dense, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    elif head == "deep":
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation="silu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation="gelu")(x)
        x = layers.Dropout(0.4)(x)
    elif head == "classic":
        x = layers.Dense(100, activation=activation)(x)
    else:
        raise ValueError("unknown head %r (expected one of %s)" % (head, HEADS))
    x = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=x, name="cabigru_classifier")
    for layer in model.layers:
        layer.trainable = True
    return model


def build_scratch_classifier(input_shape=(500, 9), **head_kwargs):
    """Same encoder architecture with random initialisation (no pretraining)."""
    return build_classifier(encoder_from_autoencoder(build_autoencoder_s(input_shape)), **head_kwargs)


def load_autoencoder(path):
    return tf.keras.models.load_model(path, compile=False)
