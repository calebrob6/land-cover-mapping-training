import keras
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Model
from keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Concatenate, Cropping2D

from unet import level_block

def jaccard_loss(y_true, y_pred, smooth=0.001, num_classes=7):                                                                              
    intersection = y_true * y_pred                                                                                                          
    sum_ = y_true + y_pred                                                                                                                  
    jac = K.sum(intersection + smooth, axis=(0,1,2)) / K.sum(sum_ - intersection + smooth, axis=(0,1,2))                                    
    return (1.0 - K.sum(jac) / num_classes)    

def baseline_model_landcover(input_shape, num_classes, lr=0.003, loss=None):
    inputs = Input(input_shape)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    #x4 = Conv2D(64, kernel_size=(5,5), strides=(5,5), padding="same", activation="relu")(inputs)
    #x5 = Conv2D(64, kernel_size=(3,3), strides=(2,2), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(32, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model


def extended_model_landcover(input_shape, num_classes, lr=0.003, loss=None):
    inputs = Input(input_shape)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(96, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model


def extended_model_bn_landcover(input_shape, num_classes, lr=0.003, loss=None):
    inputs = Input(input_shape)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(96, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model


def extended2_model_bn_landcover(input_shape, num_classes, lr=0.003, loss=None):
    inputs = Input(input_shape)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(96, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(96, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(64, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model


def unet_landcover(img_shape, out_ch=7, start_ch=64, depth=4, inc_rate=2., activation='relu', 
    dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, lr=0.003, loss="crossentropy"):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(64, 3, activation="relu", padding="same")(o)
    o = Conv2D(out_ch, 1, activation=None)(o)
    o = Activation("softmax")(o)

    model = Model(inputs=i, outputs=o)

    optimizer = Adam(lr=lr)

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model