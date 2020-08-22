import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Dense, Dropout
# from models.pointnet_cls_F import custom_dense,custom_conv
# from custommodel import CustomModel

def custom_conv(x,filters=32,activation='relu',bn_momentum=0.99):
    x = keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = keras.layers.BatchNormalization(momentum=bn_momentum)(x)
    return keras.layers.Activation(activation)(x)

def custom_dense(x,filters=32):
    x = keras.layers.Dense(filters)(x)
    x = keras.layers.BatchNormalization(momentum=0.0)(x)
    return keras.layers.Activation("relu")(x)
