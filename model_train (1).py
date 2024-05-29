import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

tf.compat.v1.set_random_seed(2019)

#Load training and testing data
with open("dataset\\train\\x_train.pickle", "rb") as f: #training data features
    xTrain = pickle.load(f)

with open("dataset\\train\\y_train.pickle", "rb") as g: #training data labels
    yTrain = pickle.load(g)

with open("dataset\\test\\x_test.pickle", "rb") as h: #testing data features
    x_test = pickle.load(h)

with open("dataset\\test\\y_test.pickle", "rb") as i: #testing data labels
    y_test = pickle.load(i)

#Dataset split (testing and validation)
x_train, x_val, y_train, y_val = train_test_split(xTrain, yTrain, test_size=0.2, random_state=2019)

class_count = len(np.unique(y_train, axis=0))

#Preprocess the training, validation, and testing data
x_train = x_train.reshape(-1, 45,45, 1)
x_val = x_val.reshape(-1, 45,45, 1)
x_test = x_test.reshape(-1, 45,45, 1)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
#x_train = x_train / 255.   pixel normalization is skipped since the pixel values
#x_val = x_val / 255.       of the images in the pickle file are already normalized.
#x_test = x_test / 255.

#y_train = to_categorical(y_train)  converting of labels to one-hot format is also skipped
#y_val = to_categorical(y_val)      since the labels in the pickle file are already converted.
#y_test = to_categorical(y_test)

#The shape of the training and validation data features should be (...,45,45,1)
#print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

#Create the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation = "relu", input_shape = (45,45,1)),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.8, seed = 2019),  #Dropout layer
    tf.keras.layers.Conv2D(64,(3,3), activation = "relu"),  
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.8, seed = 2019), #Dropout layer
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(650, activation="relu"),      #Adding the Hidden layer
    tf.keras.layers.Dropout(0.5, seed = 2019),
    tf.keras.layers.Dense(class_count, activation = "softmax")   #Adding the Output Layer
])

model.summary()

#Specify optimizers
from tensorflow.keras.optimizers import RMSprop,SGD,Adam
adam=Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

#Fit model
history = model.fit(x_train, y_train, batch_size=256, epochs=35, validation_data=(x_val, y_val))

#Evalute the model using the testing data
results = model.evaluate(x_test, y_test)
print("Loss: ", results[0], "Accuracy: ", results[1])

#Save the model
model.save('model2.h5')
print("Model saved as model.h5")
