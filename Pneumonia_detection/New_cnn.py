import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
classifier=Sequential()
#convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3) , activation='relu') )
#pooling
classifier.add(MaxPooling2D(pool_size=(2,2))) #reduce the size of feauture map
classifer.add(Dropout(0.25))

#second layer

classifier.add(Convolution2D(32,3,3, activation='relu') )
#pooling
classifier.add(MaxPooling2D(pool_size=(2,2))) #reduce the size of feauture map
classifer.add(Dropout(0.25))

#flattening
classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu')) #Ann hidden layer

classifier.add(Dense(output_dim=1,activation='sigmoid'))  #if not binary then softmax activation function

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Pneumonia set/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'Pneumonia set/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=457,
        nb_epoch=10,
        validation_data=test_set,
        nb_val_samples=96
        )





# serialize model to JSON
model_json = classifier.to_json()
with open("Pneumonia_model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
classifier.save_weights("Pneumonia_model.h5")
print("Saved model to disk")
from keras.models import model_from_json   
import numpy
import cv2
from keras.preprocessing import image
path = "IM-0147-0001.jpeg"
json_file = open('Pneumonia_model.json', 'r')
model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("Pneumonia_model.h5")
img_width = 64
img_height = 64
img = cv2.imread(path)
img = cv2.resize(img,(64,64),3)
img = img.reshape((-1, 64, 64, 3))

#test_image= image.load_img(picturePath, target_size = (img_width, img_height)) 
#test_image = image.img_to_array(test_image)
#test_image = numpy.expand_dims(test_image, axis = 0)
#test_image = test_image.reshape(img_width, img_height)
result = model.predict(img)
if (result[0][0]==1.0):
    print("Pneumonia")
else:
    print("Normal")
    
