import numpy as np
import glob
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.keras.backend as K

class UNetTrainer:
    def __init__(self, filter_num, img_height=256, img_width=256, img_channels=1, epochs=100):
        self.filter_num = filter_num

        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.epochs = epochs

    def conv_block(self, x, filter_size, size, dropout, batch_norm=False):
        """_summary_

        Args:
            x (_type_): _description_
            filter_size (_type_): _description_
            size (_type_): _description_
            dropout (_type_): _description_
            batch_norm (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=3)(conv)
        conv = layers.Activation("relu")(conv)
        conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
        if batch_norm is True:
            conv = layers.BatchNormalization(axis=3)(conv)
        conv = layers.Activation("relu")(conv)
        if dropout > 0:
            conv = layers.Dropout(dropout)(conv)

        return conv

    def create_unet_model(self, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=False, filter_num=32, filter_size=3, up_sample_size=2):
        """Creates an instance of keras.models.Model class that follows the UNet architecture

        Args:
            NUM_CLASSES (int, optional): Defines how many different classes (items) to segment. Defaults to 1.
            dropout_rate (float, optional): _description_. Defaults to 0.0.
            batch_norm (bool, optional): _description_. Defaults to False.
            filter_num (int, optional): _description_. Defaults to 32.
            filter_size (int, optional): _description_. Defaults to 3.
            up_sample_size (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        # Network Structure
        FILTER_NUM = filter_num # number of filters for the first layer
        FILTER_SIZE = filter_size # size of the convolutional filter
        UP_SAMP_SIZE = up_sample_size # size of upsampling filters
        
        inputs = layers.Input((self.img_width, self.img_height, self.img_channels))

        # Downsampling layers
        # DownRes 1, convolution + pooling
        conv_128 = self.conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
        pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
        # DownRes 2
        conv_64 = self.conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
        # DownRes 3
        conv_32 = self.conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
        pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
        # DownRes 4
        conv_16 = self.conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
        pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
        # DownRes 5, convolution only
        conv_8 = self.conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

        # Upsampling layers
    
        up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
        up_16 = layers.concatenate([up_16, conv_16], axis=-1)
        up_conv_16 = self.conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 7
        
        up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
        up_32 = layers.concatenate([up_32, conv_32], axis=-1)
        up_conv_32 = self.conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 8
        
        up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
        up_64 = layers.concatenate([up_64, conv_64], axis=-1)
        up_conv_64 = self.conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
        # UpRes 9
    
        up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
        up_128 = layers.concatenate([up_128, conv_128], axis=-1)
        up_conv_128 = self.conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

        # 1*1 convolutional layers
        outputs = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
        outputs = layers.Activation('sigmoid')(outputs)  #Change to softmax for multichannel

        # Model 
        model = models.Model(inputs, outputs, name="UNet")

        return model
    
    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


    def iou(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1)

    def dice_coef_loss(self, y_true, y_pred):
        return 1-self.dice_coef(y_true, y_pred)
        
    def precision(self, y_true, y_pred):
        '''Calculates the precision, a metric for multi-label classification of
        how many selected items are relevant.
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        
    def recall(self, y_true, y_pred):
        '''Calculates the recall, a metric for multi-label classification of
        how many relevant items are selected.
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
        
    def accuracy(self, y_true, y_pred):
        '''Calculates the mean accuracy rate across all predictions for binary
        classification problems.
        '''
        return K.mean(K.equal(y_true, K.round(y_pred)))
    

    # #Implement args - remove hardcoding
    # def reduce_lr(self):
    #     return tf.keras.callbacks.ReduceLROnPlateau(
    #             monitor='val_loss',  # Metric to monitor
    #             factor=0.1,  # Factor by which the learning rate will be reduced
    #             patience=3,  # Number of epochs with no improvement after which learning rate will be reduced
    #             verbose=1)  # Print a message when learning rate is reduced
    
    # def early_stopper(self):
    #     return tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    # def checkpoint(self):
    #     return tf.keras.callbacks.ModelCheckpoint(
    #             filepath='best_unet2D.h5',
    #             monitor='val_loss',
    #             save_best_only=True,
    #             verbose=1)