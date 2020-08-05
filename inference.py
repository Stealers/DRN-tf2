import tensorflow as tf
from model import DRN

def save_img(image,path):
    image = tf.math.multiply(image, 255.)
    image = tf.cast(tf.clip_by_value(image,0,255), tf.uint8)
    image = tf.image.encode_png(image,3)
    tf.io.write_file(path,image)
    return 'Done!'

def Inference(weights_path, scale=4):
    sr_model = DRN(input_shape=(None,None,3),model='DRN-S',scale=scale,nColor=3,training=False,dual=False)
    sr_model.load_weights(weights_path, by_name=True)
    def inference(lr_path, sr_path, sr_model=sr_model):
        image = tf.io.read_file(lr_path)
        image = tf.io.decode_image(image,3,expand_animations=False)
        image = tf.expand_dims(image, axis=0)
        image = tf.math.divide(tf.cast(image,tf.float32),255.)
        image = sr_model.predict(image)
        image = save_img(image[0],sr_path)
        return image
    return inference
