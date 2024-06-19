import tensorflow as tf


def generate model():
model tf.keras.Sequential([
i first convolutional iayer
tf.keras layers Conv2D(32, filter size 3, activation 'relu'),tf.keras.layers.MaxPool2D(pool size 2,strides 2),
tf keras.layers Conv2D(64,filter size 3, activation 'relu'),
tf.keras.layers.MaxPool2D(pool size-2,strides 2),
 fu11vconnected classifier
tf.keras.layers.Flatten(),
tf,keras.layers.Dense(1024,activation 'relu')，
tf,keras.layers,Dense(10,activation 'softmax')# 10 outputs
1)
return model
