import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Conv2D
from tensorflow.keras import Model


class ObjectAccuracy(tf.keras.metrics.Metric):
    """ 
    with this class I want a custome accuracy
    So that if IOU (intersection over union) between predicted box and true_box is
    larger than a threshold then this is true else false
    the formula will be:
    A(P\cap T)/A(P\cup T)*O(T)
    where A(P\cap T) is the area of the intersection
    A(P\cup T) is the area of the Union
    O(T) is the true objectness value
    """
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        self.total = self.add_weight("total",initializer="zeros")
    def update_state(self,y_true,y_pred,sample_weight=None):
        pass


@tf.function
def class_loss(class_true, class_predict):
    """Use the binary class loss for the objectness"""
    loss_function = tf.keras.losses.BinaryCrossentropy()
    return loss_function(class_true,class_predict)

@tf.function
def localization_loss(coord_true, coord_predict):
    """ 
    first compute the euclidean distance between the upperleft corners
    then we compute the heights and widths of the true bounding box and predicted box
    return is the sum of the euclidean distance of the corners and square distance of the 
    difference of the widths and heighs

    """
    delta_coord = tf.reduce_sum(
                                tf.square(coord_true[:,:2] -
                                coord_predict[:,:2])
                                )

    height_true = coord_true[:,3] - coord_true[:,1]  
    width_true = coord_true[:,2] - coord_true[:,0]

    height_predict = coord_predict[:,3] - coord_predict[:,1]
    width_predict = coord_predict[:,2] - coord_predict[:,0]
    delta_size =  tf.reduce_sum(
                                tf.square(height_true - height_predict) +
                                tf.square(width_true - width_predict)
                                )
    return delta_coord + delta_size



class FaceTracker(Model):
    def __init__(self,filter_1=2048,filter_2=2048,freezing=False,**kwargs) -> None:
        super().__init__(**kwargs)
        # Use pre-train VGG model and freeze the layers in VGG
        self.vgg = VGG16(include_top=False)
        self.vgg.trainable = freezing
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        # for the classification output
        self.conv_1 = Conv2D(
                            filters=self.filter_1,
                            kernel_size=(3,3),
                            padding="valid",
                            activation="relu"
                            )
        self.conv_output_1 = Conv2D(
                                    filters = 1,
                                    kernel_size=(1,1),
                                    activation="sigmoid")
        # for the boundingbox output

        self.conv_2 = Conv2D(
                            filters=self.filter_2,
                            kernel_size=(3,3),
                            padding="valid",
                            activation="relu"
                            )
        self.conv_output_2 = Conv2D(
                                    filters = 4,
                                    kernel_size=(1,1),
                                    activation="sigmoid"
                                    )


    def call(self,inputs,**kwargs):
        """
        X1 has shape (
                        batch_size,
                        number of box in vertical,
                        number of box in horizontal,
                        1)
        X2 has shape (
                        batch_size,
                        number of box in vertical,
                        number of box in horizontal,
                        4
                    )
        Since we want to dectect only 1 bounding box ie input has shape (120,120,3)
        the return value is the same if we use dense layer instead of convolution.

        for example if the input has shape (240,120,3)
        than X1 has shape (5,1,1), 
        X2: (5,1,4)
        so here the "box" jumps 30 pixel verticaly every time
        """
        X = self.vgg(inputs)

        X1 = self.conv_1(X)
        X1 = self.conv_output_1(X1)

        
        X2 = self.conv_2(X)
        X2 = self.conv_output_2(X2)


        return X1[:,0,0],X2[:,0,0]
        

    def compile(self,optimizer,classweight=1,coordweight=1,**kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.classweight = classweight
        self.coordweight = coordweight



    def train_step(self,inputs,training=True,**kwargs):
        X,Y = inputs
        with tf.GradientTape() as tape:
            Y_ = self(X,training=training)
            batch_classloss = class_loss(Y[0],Y_[0])
            batch_coordloss = localization_loss(Y[1],Y_[1]) 
            total_loss = (
                        self.coordweight * batch_coordloss +
                        self.classweight * batch_classloss
                        )
        grad = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad,self.trainable_variables))
        return {
                "total_loss":total_loss,
                "classloss":batch_classloss,
                "coordloss":batch_coordloss
                }

    def test_step(self,inputs,training=False,**kwargs):
        X,Y = inputs
        Y_ = self(X,training=training)
        batch_classloss = class_loss(Y[0],Y_[0])
        batch_coordloss = localization_loss(Y[1],Y_[1]) 
        total_loss = (
                        self.coordweight * batch_coordloss +
                        self.classweight * batch_classloss
                    )
        return {
                "total_loss":total_loss,
                "classloss":batch_classloss,
                "coordloss":batch_coordloss
                }

    def get_config(self):
        """ Use this to save custome class
            I will fix this
        """
        base_config = super().get_config()
        return {
                **base_config,
                "classweight" : self.classweight,
                "coordweight" : self.coordweight,
                "filter_conv_1" : self.filter_1,
                "filter_conv_2" : self.filter_2
                }


