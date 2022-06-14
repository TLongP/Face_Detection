import tensorflow as tf
import json


def load_labels(label_path):
    """
    loading labels
    label_path is a tensor
    """
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
    return [label['class']], label['bbox']


def load_image(img_path):
    """read image and return a tensor"""
    byte_img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(byte_img)
    return img


def combine_data_frompath(labels_path,images_path,shuffle_buffer=5000,batch_size=32,autotune=False):
    """
    Note that we have  thhe same number of labels files and images files and 
    each label and correspoding image have the same name


    first we read the label and image then we zip them
    do not shuffle the data before zip the data else the labels and images
    do not math each other
    """
    labels = tf.data.Dataset.list_files(labels_path,shuffle=False)
    labels = labels.map(lambda x: tf.py_function(func=load_labels,inp=[x],Tout=[tf.uint8,tf.float32]))

    images = tf.data.Dataset.list_files(images_path,shuffle=False)
    images = images.map(load_image) #read the images
    images = images.map(lambda x: tf.image.resize(x,(120,120))) #reshape the image size
    images = images.map(lambda x: x/255) # normalization

    data = tf.data.Dataset.zip((images,labels))
    data = data.shuffle(shuffle_buffer)
    if autotune:
        return data.batch(batch_size=batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return data.batch(batch_size=batch_size).cache().prefetch(1)