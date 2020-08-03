import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Input

class UpSample2D(Layer):
    def __init__(self, scale, method='bicubic', antialias=True, **kwargs):
        super(UpSample2D, self).__init__(**kwargs)
        self.scale = scale
        self.method = method
        self.antialias = antialias
    def call(self, inputs):
        output_size = [inputs.shape[1] * self.scale, inputs.shape[2] * self.scale]
        x = tf.image.resize(inputs, output_size, method=self.method, antialias=self.antialias)
        return x
    
class MeanShift(Layer):
    def __init__(self, sign=-1, **kwargs):
        super(MeanShift, self).__init__(**kwargs)
        self.sign = sign
        self.mean = tf.stack([[[[0.4488, 0.4371, 0.4040]]]])*self.sign
        
    def call(self, inputs):
        return tf.math.add(inputs, self.mean)
    
class SubPixel2D(Layer):
    def __init__(self, scale, **kwargs):
        super(SubPixel2D, self).__init__(**kwargs)
        self.scale = scale
        
    def call(self, inputs):
        x = tf.nn.depth_to_space(inputs, block_size=self.scale)
        return x
    
def head(inputs, nFeats, name):
    return Conv2D(nFeats,(3,3),(1,1),padding='same',name=name+'/Conv')(inputs)

def tail(inputs, nColor, name):
    return Conv2D(nColor,(3,3),(1,1),padding='same',name=name+'/Conv')(inputs)
    
def downblock(inputs, nFeat, name):
    x = Conv2D(nFeat//2,(3,3),(2,2),use_bias=False,padding='same',name=name+'/Conv1')(inputs)
    x = LeakyReLU(alpha=0.2, name=name+'/LeakyReLU')(x)
    x = Conv2D(nFeat,(3,3),(1,1),use_bias=False,padding='same',name=name+'/Conv2')(x)
    return x

def upblock(inputs, nFeat1, nFeat2, scale, name):
    x = Conv2D(nFeat1*scale**2,(3,3),(1,1),padding='same',name=name+'/Conv1')(inputs)
    x = Activation('relu', name=name+'/relu')(x)
    x = SubPixel2D(scale=2, name=name+'/SubPixel2D')(x)
    x = Conv2D(nFeat2,(1,1),(1,1),padding='same',name=name+'/Conv2')(x)
    return x

def calayer(inputs, nFeat, name, reduction=16):
    x = GlobalAveragePooling2D(name=name+'/GAPool')(inputs)
    x = Reshape((1,1,nFeat), name=name+'/Reshape')(x)
    x = Conv2D(nFeat//reduction,(1,1),(1,1),activation='relu',padding='same',name=name+'/Conv1')(x)
    x = Conv2D(nFeat,(1,1),(1,1),activation='sigmoid',padding='same',name=name+'/Conv2')(x)
    x = Multiply(name=name+'/Multiply')([inputs,x])
    return x

def rcab(inputs, nFeat, name):
    x = Conv2D(nFeat,(3,3),(1,1),padding='same',name=name+'/Conv1')(inputs)
    x = Activation(tf.nn.relu, name=name+'/relu')(x)
    x = Conv2D(nFeat,(3,3),(1,1),padding='same',name=name+'/Conv2')(x)
    x = calayer(x,nFeat,name+'/calayer')
    x = Add(name=name+'/Add')([inputs,x])
    return x

def rcabblock(inputs, nFeats, nBlock, name):
    x = inputs
    for b in range(nBlock):
        x = rcab(x, nFeats, name=name+'/nBlock'+str(b)+'/rcab')
    return x

def DRN(input_shape=(128,128,3),model='DRN-S',scale=4,nColor=3,training=True,dual=True):
    if model=='DRN-S' and scale==4:
        #Params:4.8M(Paper)
        #Total params: 4,804,041(This)
        nFeats = 16
        nBlock = 30
        Down_F = [nFeats*2,nFeats*4]
        Up_F1 = [nFeats*4,nFeats*4]
        Up_F2 = [nFeats*2,nFeats*1]
    elif model=='DRN-S' and scale==8:
        #Params:5.4M(Paper)
        #Total params: 5,402,976(This)
        nFeats = 8
        nBlock = 30
        Down_F = [nFeats*2,nFeats*4,nFeats*8]
        Up_F1 = [nFeats*8,nFeats*8,nFeats*4]
        Up_F2 = [nFeats*4,nFeats*2,nFeats*1]
    elif model=='DRN-L' and scale==4:
        #Params:9.8M(Paper)
        #Total params: 9,825,869(This)
        nFeats = 20
        nBlock = 40
        Down_F = [nFeats*2,nFeats*4]
        Up_F1 = [nFeats*4,nFeats*4]
        Up_F2 = [nFeats*2,nFeats*1]
    elif model=='DRN-L' and scale==8:
        #Params:10.0M(Paper)
        #Total params: 10,003,994(This)
        nFeats = 10
        nBlock = 36
        Down_F = [nFeats*2,nFeats*4,nFeats*8]
        Up_F1 = [nFeats*8,nFeats*8,nFeats*4]
        Up_F2 = [nFeats*4,nFeats*2,nFeats*1]

    shortcut = []
    result_nfeat = []
    lr_image = []
    sr2lr_image = []
    outputs = []
    steps = int(np.log2(scale))
    inputs = Input(input_shape, name='inputs')

    x = UpSample2D(scale=scale, name='UpSample2D')(inputs)
    x = MeanShift(sign=-1, name='Mean_Sub')(x)
    x = head(x, nFeats*1, name='head')

    for down_step in range(steps):
        shortcut.append(x)
        x = downblock(x, Down_F[down_step], name='downblock'+str(down_step))
    result_nfeat.append(x)

    for up_step in range(steps):
        x = rcabblock(x, Up_F1[up_step], nBlock, name='rcabblock'+str(up_step))
        x = upblock(x, Up_F1[up_step], Up_F2[up_step], scale=2, name='upblock'+str(up_step))
        x = Concatenate(name='Concat'+str(up_step))([shortcut[steps-up_step-1], x])
        result_nfeat.append(x)
    
    if training:
        if dual:
            x = tail(result_nfeat[-1], nColor, name='tail')
            x = MeanShift(sign=1, name='Mean_Add')(x)
            SR_out = x
            for step in range(steps):
                x = Conv2D(16,(3,3),(2,2),use_bias=False,padding='same',name='dual/Conv1'+str(step))(x)
                x = LeakyReLU(alpha=0.2, name='dual/LeakyReLU'+str(step))(x)
                x = Conv2D(3 ,(3,3),(1,1),use_bias=False,padding='same',name='dual/Conv2'+str(step))(x)
                x = MeanShift(sign=1, name='dual/Mean_Add'+str(step))(x)
                sr2lr_image.append(x)
                y = tail(result_nfeat[-step-2], nColor, name='tail'+str(step))
                y = MeanShift(sign=1, name='Mean_Add'+str(step))(y)
                lr_image.append(y)
                
            outputs.append(SR_out)
            for step in range(steps):
                outputs.append(tf.concat([lr_image[step],sr2lr_image[step]],axis=-1))
            model = Model(inputs, outputs)
        else:
            x = tail(result_nfeat[-1], nColor, name='tail')
            x = MeanShift(sign=1, name='Mean_Add')(x)
            SR_out = x
            model = Model(inputs, SR_out)
        return model
    
    x = tail(result_nfeat[-1], nColor, name='tail')
    x = MeanShift(sign=1, name='Mean_Add')(x)
    SR_out = x
    model = Model(inputs, SR_out)
    return model