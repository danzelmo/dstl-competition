from keras.models import Model
import keras.layers as l
from keras.optimizers import Adam

def conv(x, num_filt, k=3, rate=1):
    return l.Conv2D(num_filt, (k, k),activation='relu',dilation_rate=rate, padding='same')(x)

def conv_bn(x, num_filt, k=3, rate=1):
    x = l.Conv2D(num_filt, (k, k), dilation_rate=rate, padding='same')(x)
    x = l.BatchNormalization()(x)
    x = l.core.Activation('relu')(x)
    return x

def conv_bn_dr(net, num_filt, dr, k=3):
    net = conv_bn(net, num_filt, k)
    return l.Dropout(dr)(net)

def mrg(x_sm, x_lrg):
    x_sm = l.UpSampling2D(size=(2,2))(x_sm)
    return l.concatenate([x_sm, x_lrg])

def unet(in_shp_y, in_shp_x, num_chans, n_class):
    inputs = l.Input((in_shp_y, in_shp_x, num_chans))
    x = conv_bn(inputs, 64)
    x0 = conv_bn(x, 64)
    x = l.MaxPooling2D(pool_size=2)(x0)

    x = conv_bn(x, 128)
    x1 = conv_bn(x, 128)
    x = l.MaxPooling2D(pool_size=2)(x1)

    x = conv_bn(x, 256)
    x2 = conv_bn(x, 256)
    x = l.MaxPooling2D(pool_size=2)(x2)

    x = conv_bn(x, 512)
    x = conv_bn(x, 512)
    x = conv_bn(x, 512)

    x = mrg(x, x2)
    x = conv_bn(x, 256)
    x = conv_bn(x, 256)

    x = mrg(x, x1)
    x = conv_bn(x, 128)
    x = conv_bn(x, 128)

    x = mrg(x, x0)
    x = conv_bn_dr(x, 64, 0.5)
    x = conv_bn_dr(x, 64, 0.5)

    x = l.Conv2D(n_class, (1, 1), activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(optimizer=Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
