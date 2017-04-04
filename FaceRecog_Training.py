#Necessary Import Statements
import os
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF
tf.python.control_flow_ops = tf

from keras.preprocessing.image import ImageDataGenerator
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.utils import np_utils
from keras.optimizers import adam, SGD

from keras.callbacks import ModelCheckpoint
from keras.initializations import lecun_uniform, glorot_normal, glorot_uniform

import matplotlib.pyplot as plt

import os 
from PIL import Image
import h5py

#Confirming the Version of TensorFlow and Keras
# This code should work well with Keras -- 1.1.0 and TensorFlow 0.10.0
print ("Tensorflow Version", tf.__version__)
print ("Keras Version", keras.__version__)

#DEFINING MODEL ARCHITECTURE
def facereg_model():

	model1 = models.Sequential()

	#FIRST CONVOLUTION-POOLING LAYER
	model1.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=(100,100,1), init=glorot_uniform))
	model1.add(Activation('relu'))
	model1.add(MaxPooling2D(pool_size=(2, 2)))

	#SECOND CONVOLUTION ONLY LAYER (NO POOLING) 
	model1.add(Convolution2D(64, 3, 3, border_mode='same', init=glorot_uniform))
	model1.add(Activation('relu'))

	#THIRD CONVOLUTION-POOLING LAYER
	model1.add(Convolution2D(64, 3, 3, border_mode='same', init=glorot_uniform))
	model1.add(Activation('relu'))
	model1.add(MaxPooling2D(pool_size=(2, 2)))

	#FIRST FULLY CONNECTED LAYER
	model1.add(Flatten())  
	model1.add(Dense(128))
	model1.add(Activation('relu'))

	#DROPOUT - Helps Reduce Overfitting
	model1.add(Dropout(0.5))

	#SECOND FULLY CONNECTED LAYER- TEN NUMBER OF CLASSES
	model1.add(Dense(10))
	model1.add(Activation('softmax'))

	#OPTIMIZER USED- STOCHASTIC GRADIENT DESCENT
	sgd = SGD(lr=0.0001, momentum=0.9, nesterov=False)

	#GENERATE THE MODEL GRAPH
	model1.compile(loss='categorical_crossentropy',
		  optimizer=sgd,
		  metrics=['accuracy'])

	return model1


def Train(trainfiles_path, validationfiles_path, weights_path):
    
	#Setting The BATCH SIZE for Image Generators
	batch_size = 32

	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator()

	validation_datagen = ImageDataGenerator()

	train_generator = train_datagen.flow_from_directory(
	trainfiles_path,  # this is the target directory
	target_size=(100, 100),  
	batch_size=batch_size,
	class_mode='categorical',
	color_mode ='grayscale') 

	# this is a similar generator, for validation data
	validation_generator = validation_datagen.flow_from_directory(
	validationfiles_path,
	target_size=(100, 100),
	batch_size=batch_size,
	class_mode='categorical',
	color_mode ='grayscale')
	
	#Fetch the Model
	model1 = facereg_model()
	
	#START THE TRAINING PROCESS
	model1.fit_generator(
        #Provide name of trainimg image generator
        train_generator,
        #Total Number of Train Images
        samples_per_epoch=3000, 
        #Number of Epochs
        nb_epoch=20,  
        #Provide name of Validation image generator
        validation_data=validation_generator,
        #Total Number of Validation Images
        nb_val_samples=40)

	model1.save(weights_path)

	print ("Training Process Successful")

if __name__ == "__main__":

	Train_path = '/home/viswa/face_recognition/10_Class_Train'
	Validation_path = '/home/viswa/face_recognition/10_Class_Valid/'
	Weights_path = '/home/viswa/face_recognition/10_Class_FaceRecog_Trial2.hdf5'

	model = facereg_model()
	print (model.summary())
	
	print ("Start Training Process")
	Train(Train_path, Validation_path, Weights_path)

