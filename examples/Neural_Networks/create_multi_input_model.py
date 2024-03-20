from tensorflow.keras.layers import (
    concatenate,
    Dense,
    Conv1DTranspose,
    Activation,
    BatchNormalization,
    MaxPooling1D,
    Flatten,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
import tensorflow as tf


def create_multi_input_model(
    cnn_model: Model, mlp_model: Model, filters=(256, 256, 256, 256)
):
    concatenated = concatenate([mlp_model.output, cnn_model.output])
    concatenated = tf.reshape(concatenated, [-1, 1, concatenated.shape[1]])
    chanDim = -1
    x = None
    # loop over the number of filters
    for i, f in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        # CONV => RELU => BN => POOL
        if i == 0:
            x = concatenated
        # x = Dense(f)(x)
        x = Conv1DTranspose(f, kernel_size=3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling1D(pool_size=2, strides=1, padding="same")(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    return Model(inputs=[mlp_model.input, cnn_model.input], outputs=x)
