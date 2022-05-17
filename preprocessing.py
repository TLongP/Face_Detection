import tensorflow as tf
import json


def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
    return [label['class']], label['bbox']
def load_image(img_path):
    byte_img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(byte_img)
    return img
def combine_data_frompath(labels_path,images_path,shuffle_buffer=5000,batch_size=32):
    labels = tf.data.Dataset.list_files(labels_path,shuffle=False)
    labels = labels.map(lambda x: tf.py_function(func=load_labels,inp=[x],Tout=[tf.uint8,tf.float32]))
    images = tf.data.Dataset.list_files(images_path,shuffle=False)
    images = images.map(load_image)
    images = images.map(lambda x: tf.image.resize(x,(120,120)))
    images = images.map(lambda x: x/255)
    data = tf.data.Dataset.zip((images,labels))
    data = data.shuffle(shuffle_buffer)
    return data.batch(batch_size=batch_size).cache().prefetch(tf.data.AUTOTUNE)