import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
'''
path = 'Dataset/Face'
face_X = []
face_Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32,32))
            face_X.append(img)
            face_Y.append(int(name)-1)
            print(root+"/"+name)

face_X = np.asarray(face_X)
face_Y = np.asarray(face_Y)

np.save("model/faceX.txt",face_X)
np.save("model/faceY.txt",face_Y)
'''
face_X = np.load("model/faceX.txt.npy")
face_Y = np.load("model/faceY.txt.npy")

face_X = face_X.astype('float32')
face_X = face_X/255

indices = np.arange(face_X.shape[0])
np.random.shuffle(indices)
face_X = face_X[indices]
face_Y = face_Y[indices]
face_Y = to_categorical(face_Y)

face_X_train, face_X_test, face_y_train, face_y_test = train_test_split(face_X, face_Y, test_size=0.2)
vgg16 = VGG16(input_shape=(face_X_train.shape[1], face_X_train.shape[2], face_X_train.shape[3]), include_top=False, weights='imagenet')
for layer in vgg16.layers:
    layer.trainable = False
face_model = Sequential()
face_model.add(vgg16)
face_model.add(Convolution2D(32, (1 , 1), input_shape = (face_X_train.shape[1], face_X_train.shape[2], face_X_train.shape[3]), activation = 'relu'))
face_model.add(MaxPooling2D(pool_size = (1, 1)))
face_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
face_model.add(MaxPooling2D(pool_size = (1, 1)))
face_model.add(Flatten())
face_model.add(Dense(units = 256, activation = 'relu'))
face_model.add(Dense(units = face_y_train.shape[1], activation = 'softmax'))
face_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/face_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/face_weights.hdf5', verbose = 1, save_best_only = True)
    hist = face_model.fit(face_X_train, face_y_train, batch_size = 32, epochs = 20, validation_data=(face_X_test, face_y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/face_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    face_model.load_weights("model/face_weights.hdf5")   
predict = face_model.predict(face_X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(face_y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)

#finger==============================================================
'''
path = 'Dataset/FingerVein'
finger_X = []
finger_Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32,32))
            finger_X.append(img)
            finger_Y.append(int(name)-1)
            print(root+"/"+name)

finger_X = np.asarray(finger_X)
finger_Y = np.asarray(finger_Y)

np.save("model/fingerX.txt",finger_X)
np.save("model/fingerY.txt",finger_Y)
'''
finger_X = np.load("model/fingerX.txt.npy")
finger_Y = np.load("model/fingerY.txt.npy")

finger_X = finger_X.astype('float32')
finger_X = finger_X/255

indices = np.arange(finger_X.shape[0])
np.random.shuffle(indices)
finger_X = finger_X[indices]
finger_Y = finger_Y[indices]
finger_Y = to_categorical(finger_Y)

finger_X_train, finger_X_test, finger_y_train, finger_y_test = train_test_split(finger_X, finger_Y, test_size=0.2)
vgg16 = VGG16(input_shape=(finger_X_train.shape[1], finger_X_train.shape[2], finger_X_train.shape[3]), include_top=False, weights='imagenet')
for layer in vgg16.layers:
    layer.trainable = False
finger_model = Sequential()
finger_model.add(vgg16)
finger_model.add(Convolution2D(32, (1 , 1), input_shape = (finger_X_train.shape[1], finger_X_train.shape[2], finger_X_train.shape[3]), activation = 'relu'))
finger_model.add(MaxPooling2D(pool_size = (1, 1)))
finger_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
finger_model.add(MaxPooling2D(pool_size = (1, 1)))
finger_model.add(Flatten())
finger_model.add(Dense(units = 256, activation = 'relu'))
finger_model.add(Dense(units = finger_y_train.shape[1], activation = 'softmax'))
finger_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/finger_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/finger_weights.hdf5', verbose = 1, save_best_only = True)
    hist = finger_model.fit(finger_X_train, finger_y_train, batch_size = 32, epochs = 30, validation_data=(finger_X_test, finger_y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/finger_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    finger_model.load_weights("model/finger_weights.hdf5")   
predict = finger_model.predict(finger_X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(finger_y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)
            
#iris================================================
'''
path = 'Dataset/Iris'
iris_X = []
iris_Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32,32))
            iris_X.append(img)
            iris_Y.append(int(name)-1)
            print(root+" == "+name)

iris_X = np.asarray(iris_X)
iris_Y = np.asarray(iris_Y)

np.save("model/irisX.txt",iris_X)
np.save("model/irisY.txt",iris_Y)
'''
iris_X = np.load("model/irisX.txt.npy")
iris_Y = np.load("model/irisY.txt.npy")

iris_X = iris_X.astype('float32')
iris_X = iris_X/255

indices = np.arange(iris_X.shape[0])
np.random.shuffle(indices)
iris_X = iris_X[indices]
iris_Y = iris_Y[indices]
iris_Y = to_categorical(iris_Y)

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_Y, test_size=0.2)
vgg16 = VGG16(input_shape=(iris_X_train.shape[1], iris_X_train.shape[2], iris_X_train.shape[3]), include_top=False, weights='imagenet')
for layer in vgg16.layers:
    layer.trainable = False
iris_model = Sequential()
iris_model.add(vgg16)
iris_model.add(Convolution2D(32, (1 , 1), input_shape = (iris_X_train.shape[1], iris_X_train.shape[2], iris_X_train.shape[3]), activation = 'relu'))
iris_model.add(MaxPooling2D(pool_size = (1, 1)))
iris_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
iris_model.add(MaxPooling2D(pool_size = (1, 1)))
iris_model.add(Flatten())
iris_model.add(Dense(units = 256, activation = 'relu'))
iris_model.add(Dense(units = iris_y_train.shape[1], activation = 'softmax'))
iris_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/iris_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/iris_weights.hdf5', verbose = 1, save_best_only = True)
    hist = iris_model.fit(iris_X_train, iris_y_train, batch_size = 32, epochs = 20, validation_data=(iris_X_test, iris_y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/iris_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    iris_model.load_weights("model/iris_weights.hdf5")   
predict = iris_model.predict(iris_X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(iris_y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)
