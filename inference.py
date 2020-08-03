import tensorflow as tf
from model import DRN

def decode_img(path, input_size):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image,3,expand_animations=False)
    h_src = tf.shape(image)[0]
    w_src = tf.shape(image)[1]
    h_parti = h_src//input_size+1
    w_parti = w_src//input_size+1
    h_dst = h_parti*input_size
    w_dst = w_parti*input_size
    image = tf.expand_dims(image, axis=0)
    image = tf.image.pad_to_bounding_box(image, 0, 0, h_dst, w_dst)
    image = tf.math.divide(tf.cast(image,tf.float32),255.)
    return image, h_parti, w_parti, h_src, w_src

def patch_img(image, input_size, h_parti, w_parti):
    image_patchs = []
    for h_step in range(h_parti):
        for w_step in range(w_parti):
            image_patch = tf.image.crop_to_bounding_box(image, h_step*input_size, w_step*input_size, input_size, input_size)
            image_patchs.append(image_patch)
    return image_patchs

def predict(image_patchs_i, model):
    image_patchs_o = []
    for image_patch_i in image_patchs_i:
        image_patch_o = model.predict(image_patch_i)
        image_patchs_o.append(image_patch_o)
    return image_patchs_o

def merge_img(images, h_parti, w_parti, output_h, output_w):
    image=[]
    for h in range(h_parti):
        image_merge_w = tf.concat(images[w_parti*h:w_parti*(h+1)],axis=2)
        image.append(image_merge_w)
    image = tf.concat(image,axis=1)
    image = tf.image.crop_to_bounding_box(image, 0, 0, output_h, output_w)
    return image

def save_img(images,path):
    images = tf.math.multiply(images, 255.)
    images = tf.cast(tf.clip_by_value(images,0,255), tf.uint8)
    for image in images:
        image = tf.image.encode_png(image,3)
        tf.io.write_file(path,image)
    return 'Done!'

def Inference(weights_path, input_size=64, scale=4):
    sr_model = DRN(input_shape=(input_size,input_size,3),model='DRN-S',scale=4,nColor=3,training=False,dual=False)
    sr_model.load_weights(weights_path)
    def inference(lr_path, sr_path):
        image, h_parti, w_parti, h_src, w_src = decode_img(lr_path, input_size)
        image_patchs = patch_img(image, input_size, h_parti, w_parti)
        image_patchs = predict(image_patchs, sr_model)
        images = merge_img(image_patchs, h_parti, w_parti, h_src*scale, w_src*scale)
        images = save_img(images,sr_path)
        return images
    return inference