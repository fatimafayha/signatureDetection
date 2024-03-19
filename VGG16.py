#Loading the libraries
import numpy as np
import pandas as pd

#Mounting drive in colab
from google.colab import drive

drive.mount('/content/drive')

#unzip dataset
!unzip '/content/drive/My Drive/Colab Notebooks/CVproj/archiveData.zip'

#Load train and test data
train_dir="/content/sign_data/sign_data/train/"
test_dir="/content/sign_data/sign_data/test/"

#visualise a train data sample
import matplotlib.pyplot as plt
img = plt.imread('/content/sign_data/sign_data/train/001/001_01.PNG')
plt.imshow(img)

#Visualise another train data sample
img1 = plt.imread('/content/sign_data/sign_data/train/001_forg/0119001_01.png')
plt.imshow(img1)

#Define barch size
SIZE = 224

# Loading and preparing the data
import cv2
import os
import glob

train_data = []
train_labels = []

for per in os.listdir('/content/sign_data/sign_data/train/'):
    for data in glob.glob('/content/sign_data/sign_data/train/'+per+'/*.*'):
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        train_data.append([img])
        if per[-1]=='g':
            train_labels.append(np.array(1))
        else:
            train_labels.append(np.array(0))

train_data = np.array(train_data)/255.0
train_labels = np.array(train_labels)

#Test Data

test_data = []
test_labels = []

for per in os.listdir('/content/sign_data/sign_data/test/'):
    for data in glob.glob('/content/sign_data/sign_data/test/'+per+'/*.*'):
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE,SIZE))
        test_data.append([img])
        if per[-1]=='g':
            test_labels.append(np.array(1))
        else:
            test_labels.append(np.array(0))

test_data = np.array(test_data)/255.0
test_labels = np.array(test_labels)

#converting train data to categorical data
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)

#Checking train data size
train_data.shape

#Reshaping train/test data
train_data = train_data.reshape(-1, SIZE,SIZE, 3)
test_data = test_data.reshape(-1, SIZE,SIZE, 3)

#Checking new shape of train data
train_data.shape

#Train data labels shape - Original or Forged labels
train_labels.shape

#shuffle train and test data
from sklearn.utils import shuffle
train_data,train_labels = shuffle(train_data,train_labels)
test_data,test_labels = shuffle(test_data,test_labels)

#Transfer learning using VGG16 pre-trained on ImageNet dataset and adding new layers on top for Signature-Verification Dataset
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.summary()

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dropout(0.5))  # Add a Dropout layer for regularization
add_model.add(Dense(128, activation='relu'))  # Add another Dense layer
add_model.add(Dense(64, activation='relu'))  # Add one more Dense layer
add_model.add(Dense(2, activation='softmax'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.summary()

#Define stopping condition to stop model training
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
earlyStopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=1)

early_stop=[earlyStopping]


EPOCHS = 20
BS = 64       #Batch Size

# Train the model for the new layers added

#progess = model.fit(train_data,train_labels, batch_size=BS,epochs=EPOCHS, callbacks=early_stop,validation_split=.3)

#model.save("HC_SignForg3.h5")

#Save model

from keras.models import model_from_json

# Load the model architecture from JSON
json_file_path = '/content/HC_SignForg3.json'
with open(json_file_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Create the model from the loaded JSON
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights('/content/HC_SignForg3_weights.h5')

import tensorflow as tf

# Load the saved model
model = loaded_model

#Save the training/Validation loss values for each epoch in pickle file
import pickle

#history = {
 #   'accuracy': progess.history['accuracy'],
  #  'val_accuracy': progess.history['val_accuracy'],
    #'loss': progess.history['loss'],
 #   'val_loss': progess.history['val_loss']
#}

#with open('training_history.pkl', 'wb') as file:
   # pickle.dump(history, file)

# Plotting the training history
#acc = history['accuracy']
#val_acc = history['val_accuracy']
#loss = history['loss']
#val_loss = history['val_loss']

epochs = range(1,14)



# Load the saved history for train/val loss values
with open('/content/training_history.pkl', 'rb') as file:
    loaded_history = pickle.load(file)

#Plot Train/Validation Loss and Accuracies for each epoch
import matplotlib.pyplot as plt


acc = loaded_history['accuracy']
val_acc = loaded_history['val_accuracy']
loss = loaded_history['loss']
val_loss = loaded_history['val_loss']

plt.plot(epochs, acc, 'b', label='Training Acc')
plt.plot(epochs, val_acc, 'r', label='Validation Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
#plt.savefig('accuracy_vgg2.jpg')

plt.show()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


#plt.savefig('loss_vgg.jpg')
plt.show()

#Make predictions using the trained model
pred = model.predict(test_data)

#Evaluate Test Accuracy
from sklearn.metrics import accuracy_score
print('Test Accuracy : ',accuracy_score(np.argmax(pred,axis=1), test_labels)*100, '%')

# Make predictions on the test set
import matplotlib.pyplot as plt
import numpy as np

test_predictions = model.predict(test_data)

#Predict the class of Test images using the trained model, and visualise the results
# Define class names
class_names = ["Original", "Forged"]

# Choose some random samples to display
num_samples_to_display = 5
random_indices = np.random.choice(len(test_data), num_samples_to_display, replace=False)

# Display the images along with their predictions
plt.figure(figsize=(15, 5))

for i, index in enumerate(random_indices, 1):
    plt.subplot(1, num_samples_to_display, i)
    plt.imshow(test_data[index])

    actual_class = class_names[int(test_labels[index])]
    predicted_class = class_names[np.argmax(test_predictions[index])]

    plt.title(f'Actual: {actual_class}\n Predicted: {predicted_class}')
    plt.axis('off')

plt.show()
