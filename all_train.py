import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model, load_model
from keras.models import model_from_json
import pickle
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if os.path.exists("model/all_X.txt.npy"):
    X = np.load("model/all_X.txt.npy")
    Y = np.load("model/all_Y.txt.npy")
else:
    face_X = np.load("model/faceX.txt.npy")
    face_Y = np.load("model/faceY.txt.npy")
    face_X = face_X.astype('float32')
    face_X = face_X/255
    indices = np.arange(face_X.shape[0])
    np.random.shuffle(indices)
    face_X = face_X[indices]
    face_Y = face_Y[indices]
    finger_X = np.load("model/fingerX.txt.npy")
    finger_Y = np.load("model/fingerY.txt.npy")
    finger_X = finger_X.astype('float32')
    finger_X = finger_X/255
    indices = np.arange(finger_X.shape[0])
    np.random.shuffle(indices)
    finger_X = finger_X[indices]
    finger_Y = finger_Y[indices]
    iris_X = np.load("model/irisX.txt.npy")
    iris_Y = np.load("model/irisY.txt.npy")
    iris_X = iris_X.astype('float32')
    iris_X = iris_X/255
    indices = np.arange(iris_X.shape[0])
    np.random.shuffle(indices)
    iris_X = iris_X[indices]
    iris_Y = iris_Y[indices]
    face_model = load_model("model/face_weights.hdf5")
    finger_model = load_model("model/finger_weights.hdf5")
    iris_model = load_model("model/iris_weights.hdf5")   
    face_features = Model(face_model.inputs, face_model.layers[-2].output)#create face  model
    face_features = face_features.predict(face_X)  #extracting face features from vgg16
    finger_features = Model(finger_model.inputs, finger_model.layers[-2].output)#create finger  model
    finger_features = finger_features.predict(finger_X)  #extracting finger features from vgg16
    iris_features = Model(iris_model.inputs, iris_model.layers[-2].output)#create iris  model
    iris_features = iris_features.predict(iris_X)  #extracting iris features from vgg16
    X = np.hstack((face_features, finger_features[0:100], iris_features[0:100]))
    Y = face_Y
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], 16, 16, 3))
    np.save("model/all_X.txt", X)
    np.save("model/all_Y.txt", Y)
    print("Extraction Completed")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
fusion_model = Sequential()
fusion_model.add(Convolution2D(32, (3, 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
fusion_model.add(MaxPooling2D(pool_size = (2, 2)))
fusion_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
fusion_model.add(MaxPooling2D(pool_size = (2, 2)))
fusion_model.add(Flatten())
fusion_model.add(Dense(units = 256, activation = 'relu'))
fusion_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
fusion_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/fusion_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/fusion_weights.hdf5', verbose = 1, save_best_only = True)
    hist = fusion_model.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/fusion_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    fusion_model.load_weights("model/fusion_weights.hdf5")   
predict = fusion_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)







            
