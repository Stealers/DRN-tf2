import tensorflow as tf
import os

def data_load(dirs):
    paths=[]
    for root, dirs, files in os.walk(dirs):
        for file in files:
            path = os.path.join(root, file)
            paths.append(path)
    return paths

def data_decode(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image,3,expand_animations=False)
    return image

def data_prepare(image,downscale,input_shape):
    src_h = tf.shape(image)[0]
    src_w = tf.shape(image)[1]
    inp_h = input_shape[0]
    inp_w = input_shape[1]
    lr_h, lr_w = src_h//downscale+1, src_w//downscale+1
    hr_h, hr_w = lr_h*downscale, lr_w*downscale

    hr_image = tf.image.pad_to_bounding_box(image, 0, 0, hr_h, hr_w)
    lr_image = tf.image.resize(hr_image, [lr_h, lr_w], method='bicubic', antialias=True)
    lr_image = tf.cast(tf.clip_by_value(lr_image,0,255), tf.uint8)
    
    return lr_image, hr_image

def data_patch(lr_image,hr_image,downscale,input_shape):

    lr_h = tf.shape(lr_image)[0]
    lr_w = tf.shape(lr_image)[1]

    inp_h = input_shape[0]
    inp_w = input_shape[1]
    out_h = inp_h*downscale
    out_w = inp_w*downscale
    lr_xmin = tf.random.uniform([],minval=0,maxval=lr_h-inp_h+1,dtype=tf.int32,seed=666)
    lr_ymin = tf.random.uniform([],minval=0,maxval=lr_w-inp_w+1,dtype=tf.int32,seed=888)
    hr_xmin = lr_xmin*downscale
    hr_ymin = lr_ymin*downscale

    lr_image = tf.image.crop_to_bounding_box(lr_image, lr_xmin, lr_ymin, inp_h, inp_w)
    hr_image = tf.image.crop_to_bounding_box(hr_image, hr_xmin, hr_ymin, out_h, out_w)
    
    return lr_image, hr_image

def rot90(lr_image, hr_image):
    lr_image = tf.image.rot90(lr_image)
    hr_image = tf.image.rot90(hr_image)
    return lr_image, hr_image

def flip_x(lr_image, hr_image):
    lr_image = tf.image.flip_left_right(lr_image)
    hr_image = tf.image.flip_left_right(hr_image)
    return lr_image, hr_image

def flip_y(lr_image, hr_image):
    lr_image = tf.image.flip_up_down(lr_image)
    hr_image = tf.image.flip_up_down(hr_image)
    return lr_image, hr_image

def data_augment(lr_image, hr_image):
    lr_image, hr_image = tf.cond(tf.random.uniform([1],seed=111) < .5, lambda: rot90(lr_image, hr_image), lambda: (lr_image, hr_image))
    lr_image, hr_image = tf.cond(tf.random.uniform([1],seed=222) < .5, lambda: flip_x(lr_image, hr_image), lambda: (lr_image, hr_image))
    lr_image, hr_image = tf.cond(tf.random.uniform([1],seed=333) < .5, lambda: flip_y(lr_image, hr_image), lambda: (lr_image, hr_image))
    return lr_image, hr_image

def data_normalize(lr_image, hr_image):
    lr_image = tf.math.divide(tf.cast(lr_image,tf.float32),255.)
    hr_image = tf.math.divide(tf.cast(hr_image,tf.float32),255.)
    return lr_image, hr_image
