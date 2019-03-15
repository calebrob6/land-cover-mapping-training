import keras
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from models import jaccard_loss

'''
U-Net: Convolutional Networks for Biomedical Image Segmentation
(https://arxiv.org/abs/1505.04597)
---
img_shape: (height, width, channels)
out_ch: number of output channels
start_ch: number of channels of the first conv
depth: zero indexed depth of the U-structure
inc_rate: rate at which the conv channels will increase
activation: activation function after convolutions
dropout: amount of dropout in the contracting part
batchnorm: adds Batch Normalization if true
maxpool: use strided conv instead of maxpooling if false
upconv: use transposed conv instead of upsamping + conv if false
residual: add residual connections around each conv block if true
'''

def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(img_shape, out_ch=7, start_ch=64, depth=4, inc_rate=2., activation='relu', 
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



def UNet_sgd(img_shape, out_ch=7, start_ch=64, depth=4, inc_rate=2., activation='relu', 
    dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, lr=0.003, loss="crossentropy"):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(64, 3, activation="relu", padding="same")(o)
    o = Conv2D(out_ch, 1, activation=None)(o)
    o = Activation("softmax")(o)

    model = Model(inputs=i, outputs=o)

    optimizer = SGD(lr=lr, momentum=0.9, )

    if loss == "jaccard":
        model.compile(loss=jaccard_loss, metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    elif loss == "crossentropy":
        model.compile(loss="categorical_crossentropy", metrics=["categorical_crossentropy", "accuracy", jaccard_loss], optimizer=optimizer)
    else:
        print("Loss function not specified, model not compiled")
    return model