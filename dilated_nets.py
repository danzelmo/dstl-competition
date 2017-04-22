from keras.models import Model
from keras.optimizers import Adam
from keras import layers as l

def conv(x, num_filt, k=3, rate=1):
    return l.Conv2D(num_filt, (k, k),activation='relu',dilation_rate=rate, padding='valid')(x)


def conv_bn(x, num_filt, k=3, rate=1):
    x = l.Conv2D(num_filt, (k, k),dilation_rate=rate, padding='valid')(x)
    x = l.BatchNormalization()(x)
    x = l.core.Activation('relu')(x)
    return x

def conv_bn_dr(net, num_filt, dr, k=3, rate=1):
    net = conv_bn(net, num_filt, k, rate)
    return l.Dropout(dr)(net)
          
def conv1(x, num_filt):
    return l.Conv2D(num_filt, (1, 1),activation='relu', padding='valid')(x)


def conv1_bn(x, num_filt):
    x = l.Conv2D(num_filt, (1, 1), padding='valid')(x)
    x = l.BatchNormalization()(x)
    x = l.core.Activation('relu')(x)
    return x
         
def conv1_bn_dr(x, num_filt, dr):
    x = conv1_bn(x, num_filt)
    return l.Dropout(dr)(x)

def conv_block(x, filts,  n_conv_lay):
    for i in range(n_conv_lay):
        x = conv_bn(x, filts)
    return x

def crop_conv_block(x, filts, in_crop, n_conv_lay):
    x = crp(x,in_crop - n_conv_lay)
    for i in range(n_conv_lay):
        x = conv_bn(x, filts)
    return x

def dilated_block0(x, filts, rates, in_crop, n_conv_lay):
    for e,rate in enumerate(rates):
        if e == 0:
            x = crp(x,in_crop -sum(rates) - n_conv_lay)
            x = conv_bn(x, filts, rate=rate)
        else:
            x = conv_bn(x, filts, rate=rate)
         
    if n_conv_lay > 0:
        x = conv_block(x, filts, n_conv_lay)
    return x

def dilated_block1(x, filts, rates, in_crop, n_conv_lay):
    for e,rate in enumerate(rates):
        if e == 0:
            x = crp(x,in_crop -sum(rates) -len(rates)*n_conv_lay)
            x = conv_bn(x, filts, rate=rate)
            x = conv_block(x, filts, n_conv_lay)
        else:
            x = conv_bn(x, filts, rate=rate)
            x = conv_block(x, filts, n_conv_lay)
    return x

def slice_rgb(im):
    return im[:,:,:,:3]

def slice_m(im):
    return im[:,:,:,3:]

def crp(x,amount):
    return l.Cropping2D(cropping=((amount,amount),(amount,amount)))(x)

def upm(xm):
    return l.UpSampling2D(size=(4,4))(xm)

def atr_rgb1(rgb, filts, in_crop):
    x1 = dilated_block0(rgb, filts, [7], in_crop,2)
    x2 = dilated_block0(rgb, filts, [11], in_crop,2)
    x3 = dilated_block0(rgb, filts, [3], in_crop,2)
    x4 = dilated_block0(rgb, filts, [5], in_crop,2)

    x = l.concatenate([x1,x2,x3,x4])
    x = conv_bn(x, filts*4)
    x = conv_bn(x, filts*4)
    return x
       
def atr_rgb2(rgb, filts, in_crop):
    x1 = dilated_block0(rgb, filts, [2,2], in_crop,2)
    x2 = dilated_block0(rgb, filts, [4,4], in_crop,2)
    x3 = dilated_block0(rgb, filts, [5,5], in_crop,2)
    x4 = dilated_block0(rgb, filts, [3,3], in_crop,2)
    x = l.concatenate([x1,x2,x3,x4])
    x = conv_bn(x, filts*4)
    x = conv_bn(x, filts*4)
    return x

def atr_m2(m, filts, in_crop):
    x1 = dilated_block0(m, filts, [3,3], in_crop,2)
    x2 = dilated_block0(m, filts, [5,3], in_crop,2)
    x3 = dilated_block0(m, filts, [3,5], in_crop,2)
    x4 = dilated_block0(m, filts, [5,5], in_crop,2)
    x = l.concatenate([x1,x2,x3,x4])
    x = conv_bn(x, filts*4)
    x = conv_bn(x, filts*4)
    return x

def atr_m1(m, filts, in_crop):
    x1 = dilated_block0(m, filts, [3], in_crop, 2)
    x2 = dilated_block0(m, filts, [7], in_crop, 2)
    x3 = dilated_block0(m, filts, [11], in_crop, 2)
    x4 = dilated_block0(m, filts, [15], in_crop, 2)
    x5 = dilated_block0(m, filts, [19], in_crop, 2)
    x = l.concatenate([x1,x2,x3,x4,x5])
    x = conv_bn(x, filts*4)
    x = conv_bn(x, filts*4)
    return x
 
def atr_net(in_shp_y, in_shp_x, num_chans, n_cls):
    filts = 32
      
    out_size1 = 48
    out_size2 = 12
    n_post_atr = 4

    #assumes square input images
    in_crop1 = (in_shp_y - out_size1)//2 - n_post_atr
    in_crop2 = (in_shp_y - out_size2)//2 - n_post_atr
    
    inputs = l.Input((in_shp_y, in_shp_x, num_chans))
    rgb = l.Lambda(slice_rgb)(inputs)
    m = l.Lambda(slice_m)(inputs)

    rgb1 = atr_rgb1(rgb, filts, in_crop1)
    rgb2 = atr_rgb2(rgb, filts, in_crop1)
   
    m1 = atr_m1(m, filts, in_crop2)
    m2 = atr_m2(m, filts, in_crop2)

    m1 = upm(m1)
    m2 = upm(m2)
    
    m1 = crp(m1,1)
    m2 = crp(m2,1)
    
    xall = l.concatenate([m1,m2,rgb1,rgb2])
    xall = conv_bn(xall, filts*8)
    xall = conv_bn(xall, filts*8)

    m1 = crp(m1,3)
    m2 = crp(m2,3)
    rgb1 = crp(rgb1,3)
    rgb2 = crp(rgb2,3)

    x = l.concatenate([xall, m1, m2, rgb1, rgb2])
    x = l.Dropout(0.35)(x)
    x = conv1_bn(x, filts*32)
    x = l.Dropout(0.35)(x)
    x = l.Conv2D(n_cls, (1, 1), activation='sigmoid')(x)

    model = Model(input=inputs, output=x)
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def atr_tiny_top(buff, out_size,num_chans, n_cls):
    filts = 128
    n_post_block = 4

    xconv0, xconv1, x1, x2, x3, inputs = atr_tiny_bot(buff, out_size, num_chans, n_post_block)

    xconv0 = conv_bn(xconv0, filts)
    xconv0 = conv_bn(xconv0, filts)

    xconv1 = conv_bn(xconv1, filts)
    xconv1 = conv_bn(xconv1, filts)

    x1 = conv_bn(x1, filts)
    x1 = conv_bn(x1, filts)

    x2 = conv_bn(x2, filts)
    x2 = conv_bn(x2, filts)

    x3 = conv_bn(x3, filts)
    x3 = conv_bn(x3, filts)

    x = l.concatenate([xconv0, xconv1, x1, x2, x3])
    x = conv_bn(x,filts*2)
    x = conv_bn(x,filts*2)

    x3 = l.Cropping2D(cropping=((2,2),(2,2)))(x3)
    x2 = l.Cropping2D(cropping=((2,2),(2,2)))(x2)
    x1 = l.Cropping2D(cropping=((2,2),(2,2)))(x1)
    xconv0 = l.Cropping2D(cropping=((2,2),(2,2)))(xconv0)
    xconv1 = l.Cropping2D(cropping=((2,2),(2,2)))(xconv1)

    x = l.concatenate([x, xconv0, xconv1, x1, x2, x3])
    x = l.Dropout(0.5)(x)
    x = conv1_bn(x, 8*filts)
    x = l.Dropout(0.5)(x)
    x = l.Conv2D(n_cls, (1,1),activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(optimizer=Adam(lr=0.00001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def atr_tiny_bot(buff, out_size, num_chans, n_post_block):
    filts = 32

    a_r0 = [9,3]
    a_r1 = [5,5]
    a_r2 = [7,7]
    a_r3 = [6,3]

    b_r0 = [11,11]
    b_r1 = [9,9]
    b_r2 = [15,7]
    b_r3 = [19,7]

    c_r0 = [3,3]
    c_r1 = [5,5]
    c_r2 = [7,7]
    c_r3 = [5,7]

    in_shp = 2*buff + out_size
    inputs = l.Input((in_shp,in_shp, num_chans))

    # assumes square input images
    in_crop = (in_shp - out_size)//2 - n_post_block

    # average pooling
    in_crop2 = (in_shp//2 - out_size)//2 - n_post_block

    xconv0 = crop_conv_block(inputs, filts*2,in_crop, 4)

    xa0 = dilated_block1(inputs, filts, a_r0, in_crop,1)
    xa1 = dilated_block1(inputs, filts, a_r1, in_crop,1)
    xa2 = dilated_block1(inputs, filts, a_r2, in_crop,1)
    xa3 = dilated_block1(inputs, filts, a_r3, in_crop,1)

    xb0 = dilated_block1(inputs, filts, b_r0, in_crop,1)
    xb1 = dilated_block1(inputs, filts, b_r1, in_crop,1)
    xb2 = dilated_block1(inputs, filts, b_r2, in_crop,1)
    xb3 = dilated_block1(inputs, filts, b_r3, in_crop,1)

    ave_pool = l.AveragePooling2D(pool_size=(2,2))(inputs)

    xconv1 = crop_conv_block(ave_pool, filts*2, in_crop2, 4)

    xc0 = dilated_block1(ave_pool, filts, c_r0, in_crop2,1)
    xc1 = dilated_block1(ave_pool, filts, c_r1, in_crop2,1)
    xc2 = dilated_block1(ave_pool, filts, c_r2, in_crop2,1)
    xc3 = dilated_block1(ave_pool, filts, c_r3, in_crop2,1)

    xa = l.concatenate([xa0, xa1, xa2,xa3])
    xb = l.concatenate([xb0, xb1, xb2, xb3])
    xc = l.concatenate([xc0, xc1, xc2, xc3])

    return xconv0, xconv1, xa, xb, xc, inputs