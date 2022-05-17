import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D
from tensorflow.keras import Model

class FaceTracker(Model):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        # Use pre-train VGG model and freeze the layers in VGG
        self.vgg = VGG16(include_top=False)
        #self.vgg.trainable = False
        # for the classification output
        self.max_pooling_1 = GlobalMaxPooling2D()
        self.dense_1 = Dense(2048,activation="relu")
        self.output_layer_1 = Dense(1,activation="sigmoid")
        # for the boundingbox output

        self.dense_2 = Dense(2048,activation="relu")
        self.output_layer_2 = Dense(4,activation="sigmoid")


    def call(self,inputs):

        X = self.vgg(inputs)
        X = self.max_pooling_1(X)

        X1 = self.dense_1(X)
        X1 = self.output_layer_1(X1)

        
        X2 = self.dense_2(X)
        X2 = self.output_layer_2(X2)
        #return tf.concat([X1,X2],axis=-1)
        return X1,X2


    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class ObjectAccuracy(tf.keras.metrics.Metric):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        self.total = self.add_weight("total",initializer="zeros")
    def update_state(self,y_true,y_pred,sample_weight=None):
        pass



def class_loss(class_true, class_predict):
    loss_function = tf.keras.losses.BinaryCrossentropy()
    return loss_function(class_true,class_predict)


def localization_loss(coord_true, coord_predict):
    delta_coord = tf.reduce_sum(tf.square(coord_true[:,:2]-coord_predict[:,:2]))
    height_true = coord_true[:,3] - coord_true[:,1]  
    width_true = coord_true[:,2] - coord_true[:,0]

    height_predict = coord_predict[:,3] - coord_predict[:,1]
    width_predict = coord_predict[:,2] - coord_predict[:,0]

    delta_size =  tf.reduce_sum(tf.square(height_true-height_predict) +
                                tf.square(width_true-width_predict))
    return delta_coord + delta_size


def compute_batch_loss(model,X,Y,training,class_weight=0.5,coord_weight=1):
    Y_ = model(X,training=training)
    batch_class_losses = class_loss(Y[0],Y_[0])
    coord_loss = localization_loss(Y[1],Y_[1])
    return class_weight*batch_class_losses + coord_weight*coord_loss


def gradient(model,X,Y):
    with tf.GradientTape() as tape:
        batch_loss = compute_batch_loss(model,X,Y,training=True)
    return batch_loss, tape.gradient(batch_loss,model.trainable_variables)
    