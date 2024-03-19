


import os
import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt

from tensorflow import keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from google.colab import drive

"""Defining the parameters"""

# -------------- Model ----------------
# backbone_model = 'custom_cnn'
## CNN models (pretrained on ImageNet)
# backbone_model = 'Xception'
# backbone_model = 'InceptionV3'
# backbone_model = 'ResNet50'
backbone_model = 'MobileNetV2'

# freeze convolutional layers:
freeze_conv_layers = True

# data:
img_size = 224        # image size

# training parameters:
batch_size = 64       # batch size
learning_rate = 1e-2  # learning rate
num_epoches = 5      # maximum number of epoches
steps_per_epoch = 100 # itration steps in each epoch

"""Loading the dataset"""

data_path="/content/drive/MyDrive/sign_data"

"""Image Preprocessing function and Data Loader Class"""

def img_norm(x):
  # a simple image preprocessing function
  return (x - x.mean(axis=(0,1,2), keepdims=True)) / x.std(axis=(0,1,2), keepdims=True)

class DataLoader:
  #constructor
  def __init__(self, dataset, batch_size=32, img_size=112, dir='./'):
    self.dataset = dataset
    self.batch_size = batch_size
    self.dir = dir
    self.img_size = img_size
  #shuffler
  def shuffle(self):
    return self.dataset.sample(frac=1)
  #generator
  def datagen(self, repeat_flag=True):
    num_samples = len(self.dataset)
    while True:
        # shuffling the samples
        self.dataset = self.shuffle()
        for batch in range(1, num_samples, self.batch_size):
            image1_batch_samples = self.dir + "/" + self.dataset.iloc[:, 0][batch:batch + self.batch_size]
            image2_batch_samples = self.dir + "/" + self.dataset.iloc[:, 1][batch:batch + self.batch_size]
            label_batch_samples = self.dataset.iloc[:, 2][batch:batch + self.batch_size]
            Image1, Image2, Label = [], [], []
            for image1, image2, label in zip(image1_batch_samples, image2_batch_samples, label_batch_samples):
                # append them to Images directly
                image1_data = Image.open(image1)
                image2_data = Image.open(image2)
                # resizing the images
                image1_data = image1_data.resize((self.img_size, self.img_size))
                image2_data = image2_data.resize((self.img_size, self.img_size))
                # converting to array
                image1_data = img_to_array(image1_data)
                image2_data = img_to_array(image2_data)

                # image1_data = preprocess_input(image1_data)
                # image2_data = preprocess_input(image2_data)
                image1_data = img_norm(image1_data)
                image2_data = img_norm(image2_data)

                Image1.append(image1_data)
                Image2.append(image2_data)
                Label.append(label)
            # convert each list to numpy arrays to ensure that they get processed by fit function
            Image1 = np.asarray(Image1).astype(np.float32)
            Image2 = np.asarray(Image2).astype(np.float32)

            Label = np.asarray(Label).astype(np.float32)
            yield [Image1, Image2], Label
        if not repeat_flag:
          break

"""Generators"""

train_set_file = "/content/drive/MyDrive/sign_data/train_data.csv"
test_set_file = "/content/drive/MyDrive/sign_data/test_data.csv"

train_val_set = pd.read_csv(train_set_file)
train_set, val_set = train_test_split(train_val_set, test_size=0.2)
test_set = pd.read_csv(test_set_file)

train_gen= DataLoader(train_set, batch_size, img_size, "/content/drive/MyDrive/sign_data/train")
test_gen = DataLoader(test_set, batch_size, img_size, "/content/drive/MyDrive/sign_data/test")
val_gen= DataLoader(val_set, batch_size, img_size, "/content/drive/MyDrive/sign_data/train")

"""Testing the train generator"""

train_batch = next(train_gen.datagen())
print("Train batch images shape:", train_batch[0][0].shape, train_batch[0][1].shape)
print("Train batch labels shape:", train_batch[1].shape)

"""Model

Base model
"""

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


def custom_cnn():
  #Convolution Layer 1
  model = Sequential()
  model.add(Conv2D(4, (3,3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(2,2))
  model.add(Dropout(0.25))
  #Convolution Layer 2
  model.add(Conv2D(16, (3,3), activation='relu'))
  model.add(MaxPooling2D(5,5))
  model.add(Dropout(0.25))
  # Additional Convolutional Layer
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(2, 2))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256, activation='relu'))

  return model


def def_base_model(backbone='xception', freeze_conv_layers=True):
  print('backbone model: ' + backbone)
  if backbone == 'Xception':
    base_model = Xception(weights='imagenet', include_top=False)
  elif backbone == 'InceptionV3':
    base_model = InceptionV3(weights='imagenet', include_top=False)
  elif backbone == 'ResNet50':
    base_model = ResNet50(weights='imagenet', include_top=False)
  elif backbone == 'MobileNetV2':
    base_model = MobileNetV2(weights='imagenet', include_top=False)
  else:
    raise("unexpected backbone model. Backbone model can be choosen from: "
    "'Xception', 'InceptionV3', 'MobileNetV2', and 'ResNet50'")

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Flatten()(x)
  x = Dense(128, activation='relu')(x)

  # first: train only the top layers (which were randomly initialized)
  # i.e. freeze all convolutional layers
  if freeze_conv_layers:
    print('freeze convolutional layers ...')
    for layer in base_model.layers:
        layer.trainable = False
  model = Model(inputs=base_model.input, outputs=x)
  return model

"""Siamese Model"""

def siamese_model(input_shape, backbone_model='custom_cnn',
                  freeze_conv_layers=True):
    input1 = Input(input_shape)
    input2 = Input(input_shape)

    if backbone_model=='custom_cnn':
      base_model = custom_cnn()
    else:
      base_model = def_base_model(backbone_model, freeze_conv_layers)

    # Call the model with the inputs:
    embedding1 = base_model(input1)
    embedding2 = base_model(input2)

    # custom loss layer:
    loss_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    manhattan_distance = loss_layer([embedding1, embedding2])

    # add a dense layer for 2-class classification (genuine and fraud):
    output = Dense(1, activation='sigmoid')(manhattan_distance)

    network = Model(inputs=[input1, input2], outputs=output)
    return network

model = siamese_model((img_size, img_size, 3), backbone_model, freeze_conv_layers)
model.summary()

"""Train and Test the data"""

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

"""Compiling the model"""

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=5*steps_per_epoch,
    decay_rate=0.5)

optimizer = Adam(learning_rate=lr_schedule, weight_decay=0.2)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy', f1_score])

early_stopper =  EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1)
custom_callback = [early_stopper]

"""Train"""

print("Training the data")
checkpoint_filepath = data_path + '/best_model.hdf5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(
    train_gen.datagen(),
    verbose=1,
    steps_per_epoch=steps_per_epoch,  # set appropriate steps_per_epoch
    epochs=num_epoches,
    validation_data=val_gen.datagen(),
    validation_steps=1,  # set appropriate validation_steps
    callbacks=[model_checkpoint_callback]
)

"""Plotting the train curves"""

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for f1_score
plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('model f1_score')
plt.ylabel('f1_score')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

"""Saving the model"""

keras.saving.save_model(model, backbone_model + '.h5', overwrite=True)

"""Testing the model"""

# loaded_model = keras.saving.load_model(backbone_model + '.h5')
loaded_model = keras.saving.load_model(checkpoint_filepath,custom_objects={"f1_score": f1_score})
result = loaded_model.evaluate(test_gen.datagen(repeat_flag=False), batch_size=None,
                               verbose=1, sample_weight=None, steps=None,
                               callbacks=None, max_queue_size=10, workers=1,
                               use_multiprocessing=False, return_dict=False)

"""Confusion matrix for the classes

"""

y_gt = []
y_pr = []
for data in test_gen.datagen(repeat_flag=False):
  labels = data[1]
  predictions = loaded_model.predict(data[0], verbose=0)
  for i, label in enumerate(labels):
    y_gt.append(label)
    y_pr.append(predictions[i])

y_pr = np.round(y_pr)
cm = confusion_matrix(y_gt, y_pr, normalize='true')
print(cm)

"""Calculating metrics

"""

from sklearn.metrics import classification_report
print(classification_report(y_gt, y_pr))

predictions = loaded_model.predict(test_gen.datagen(repeat_flag=False), verbose=1)

# Assuming 'predictions' is a numpy array with predicted probabilities
# If you want the binary predictions, you can round the probabilities
binary_predictions = np.round(predictions)

# Assuming 'predictions' is a numpy array with predicted probabilities
# If you want to classify whether the signature is original or forged based on a threshold, you can set a threshold
threshold = 0.5  # Adjust the threshold as needed
classification_predictions = (predictions >= threshold).astype(int)

# Print the classification predictions
print("Classification Predictions:")
print(classification_predictions)

"""Predicting the data"""

def visualize_all_predictions(generator, model, threshold=0.5):
    """
    Visualize predictions on all samples from the generator.

    Args:
    - generator: Data generator providing images and labels.
    - model: Trained model for predictions.
    - threshold: Classification threshold for binary predictions.

    Returns:
    None
    """
    # Initialize lists to store true labels, predicted labels, and images
    true_labels = []
    predicted_labels = []
    images = []

    # Loop through the generator to get all samples
    for samples in generator.datagen(repeat_flag=False):
        # Extract images and true labels
        image1_batch, image2_batch = samples[0]
        true_labels_batch = samples[1]

        # Make predictions
        predictions = model.predict([image1_batch, image2_batch], verbose=0)

        # Apply threshold for binary predictions
        binary_predictions = (predictions >= threshold).astype(int)

        # Extend the lists with batch information
        true_labels.extend(true_labels_batch)
        predicted_labels.extend(binary_predictions)
        images.extend(image1_batch)  # Assuming image1_batch corresponds to genuine signatures

    # Visualize the predictions for all images
    for i in range(len(true_labels)):
        plt.figure(figsize=(6, 3))

        # Display the original image
        plt.imshow(images[i])
        actual_label = "original" if true_labels[i] == 1 else "forged"
        predicted_label = "original" if predicted_labels[i] == 1 else "forged"

        plt.title(f'Actual: {actual_label}, Predicted: {predicted_label}')

        plt.show()

# Visualize predictions on all test samples
visualize_all_predictions(test_gen, loaded_model, threshold=0.5)

