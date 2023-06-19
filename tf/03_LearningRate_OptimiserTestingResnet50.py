# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 08:34:31 2022

@author: noton
"""
#Some elements used from the following tutorial: 
#https://github.com/r-sajal/DeepLearning-/blob/master/ComputerVision/k-fold-accuracy-comparison-blog.ipynb
##References :: https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594
##References ::https://www.kaggle.com/code/suniliitb96/tutorial-keras-transfer-learning-with-resnet50/notebook
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os
from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras.applications import ResNet50,ResNet101
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# LR = 0.0001

epochs =200
print(os.getcwd())
base_dir =  "/home/mii/.keras/datasets/flower_photos/"
folders = os.listdir(base_dir)
print(folders)
# img = 150
batch = 20

train = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    subset = "training",
    validation_split=0.2,
    image_size = (128, 128),
    batch_size=batch,
    seed=123
)

val = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    subset = "validation",
    validation_split=0.2,
    image_size = (128, 128),
    batch_size=batch,
    seed=123
)

# train_datagen = ImageDataGenerator(rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.1, rotation_range=45,
#     width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	horizontal_flip=True,
#     validation_split=0.2)
#
# train = train_datagen.flow_from_directory(
#     base_dir,
#     target_size=(128 , 128),
#     batch_size= batch,
#     class_mode='categorical',
#     shuffle=True,
#     subset='training')
#
# val = train_datagen.flow_from_directory(
#     base_dir, # same directory as training data
#     target_size=(128, 128),
#     batch_size= batch,
#     class_mode='categorical',
#     shuffle=True,
#     subset='validation')

reg_value = 0.1
# labels = (train.class_indices)
# labels = dict((v , k) for k , v in labels.items())
labels = dict(enumerate(train.class_names))
print(labels)
#References https://github.com/sukilau/Ziff-deep-learning/blob/master/3-CIFAR10-lrate/CIFAR10-lrate.ipynb
num_classes = 2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
def cnn_model() : 
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    add_model = Sequential()
    add_model.add(Flatten())
    add_model.add(Dropout(0.3))
    add_model.add(Dense(128, activation='relu',kernel_regularizer=l2(reg_value)))
    add_model.add(Dropout(0.3))
    add_model.add(Dense(64, activation='relu',kernel_regularizer=l2(reg_value)))
    add_model.add(Dropout(0.5))
    # add_model.add(Dense(2, activation='sigmoid'))
    add_model.add(Dense(1, activation='sigmoid'))
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    # print(model.summary())
    return(model)

# model5 = cnn_model()
# model5.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
#               metrics=['accuracy'])

# history5 = model5.fit(X_train, y_train, 
#                      validation_data=(X_test, y_test), 
#                      epochs=epochs, 
#                      batch_size=batch_size,
#                      verbose=2)

# # fit CNN model using Adadelta optimizer
# model6 = cnn_model()
# model6.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
#               metrics=['accuracy'])
# history6 = model6.fit(X_train, y_train, 
#                      validation_data=(X_test, y_test), 
#                      epochs=epochs, 
#                      batch_size=batch_size,
#                      verbose=2)

# # fit CNN model using RMSprop optimizer
# model7 = cnn_model()
# model7.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
#               metrics=['accuracy'])
# history7 = model7.fit(X_train, y_train, 
#                      validation_data=(X_test, y_test), 
#                      epochs=epochs, 
#                      batch_size=batch_size,
#                      verbose=2)

# # fit CNN model using Adam optimizer
# model8 = cnn_model()
# model8.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
#               metrics=['accuracy'])
# history8 = model8.fit(X_train, y_train, 
#                      validation_data=(val), 
#                      epochs=epochs, 
#                      batch_size=batch_size,
#                      verbose=2)




model1 = cnn_model()
model1.compile(loss = 'binary_crossentropy' , 
               optimizer = tf.keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08)
               ,metrics = 'accuracy')
model1.summary()
# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', verbose=1)
history = model1.fit(train, epochs = epochs, validation_data = val)
model1.save("my_h6_model.h5")
# # # #Prints out test loss and accuracy
# # results_test = model.evaluate(test)
# # # print(results_test)

# import sklearn.metrics as metrics
# # y_preds = model.predict_classes(X_test_data).flatten()
# # metrics.classification_report(y_test_labels, y_pred_labels)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test_dir =  "C:/MSC_Workspace/03_Project_ML_GIS_RR/3_DeepLearning/DL/Test"
test_generator = ImageDataGenerator(rescale=1. / 255)
testdata_generator = test_generator.flow_from_directory(test_dir,
    target_size=(128,128),
    shuffle=False)

predictions = model1.predict(testdata_generator)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = testdata_generator.classes
class_labels = list(testdata_generator.class_indices.keys())  

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)


import pylab as pl
import seaborn as sns
cm = confusion_matrix(true_classes, predicted_classes)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);

