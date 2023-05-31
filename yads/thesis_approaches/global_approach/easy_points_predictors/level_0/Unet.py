# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import (
    GlobalAveragePooling1D,
    Multiply,
    Add,
    AveragePooling1D,
    Concatenate,
    UpSampling1D,
    Lambda,
)


def cbr(x, out_layer, kernel, stride, dilation):
    x = Conv1D(
        out_layer,
        kernel_size=kernel,
        dilation_rate=dilation,
        strides=stride,
        padding="same",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    return x


def se_block(x_in, layer_n):
    x = GlobalAveragePooling1D()(x_in)
    x = Dense(layer_n // 8, activation="relu")(x)
    x = Dense(layer_n, activation="sigmoid")(x)
    x_out = Multiply()([x_in, x])
    return x_out


def resblock(x_in, layer_n, kernel, dilation, use_se=True):
    x = cbr(x_in, layer_n, kernel, 1, dilation)
    x = cbr(x, layer_n, kernel, 1, dilation)
    if use_se:
        x = se_block(x, layer_n)
    x = Add()([x_in, x])
    return x


def Unet(input_shape=(None, 1)):
    layer_n = 64
    kernel_size = 7
    depth = 2

    input_layer = Input(input_shape)
    input_layer_1 = AveragePooling1D(4)(input_layer)
    input_layer_2 = AveragePooling1D(16)(input_layer)

    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n * 2, kernel_size, 4, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 2, kernel_size, 1)
    out_1 = x

    x = Concatenate()([x, input_layer_1])
    x = cbr(x, layer_n * 3, kernel_size, 4, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 3, kernel_size, 1)
    out_2 = x

    x = Concatenate()([x, input_layer_2])
    x = cbr(x, layer_n * 4, kernel_size, 16, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 4, kernel_size, 1)

    ########### Decoder
    x = UpSampling1D(16)(x)
    x = Concatenate()([x, out_2])
    x = cbr(x, layer_n * 3, kernel_size, 1, 1)

    x = UpSampling1D(4)(x)
    x = Concatenate()([x, out_1])
    x = cbr(x, layer_n * 2, kernel_size, 1, 1)

    x = UpSampling1D(4)(x)
    x = Concatenate()([x, out_0])
    x = cbr(x, layer_n, kernel_size, 1, 1)

    x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
    out = Activation("sigmoid")(x)
    model = Model(input_layer, out)

    return model


def little_UNET(input_shape=(None, 1)):
    layer_n = 64
    kernel_size = 7
    depth = 1

    input_layer = Input(input_shape)

    ########## Encoder
    x = cbr(input_layer, layer_n, kernel_size, 1, 1)
    for i in range(depth):
        x = resblock(x, layer_n, kernel_size, 1)
    out_0 = x

    x = cbr(x, layer_n * 2, kernel_size, 4, 1)
    for i in range(depth):
        x = resblock(x, layer_n * 2, kernel_size, 1)
    out_1 = x

    ########### Decoder

    # x = UpSampling1D(4)(x)
    x = Concatenate()([x, out_1])
    x = cbr(x, layer_n * 2, kernel_size, 1, 1)

    x = UpSampling1D(4)(x)
    x = Concatenate()([x, out_0])
    x = cbr(x, layer_n, kernel_size, 1, 1)

    x = Conv1D(1, kernel_size=kernel_size, strides=1, padding="same")(x)
    out = Activation("sigmoid")(x)
    model = Model(input_layer, out)
    return model
