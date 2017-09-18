import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
import os
    
# Création du réseau de neurones convolutionel
classifier = Sequential()

# Etape 1 - Convolution
classifier.add(Convolution2D(8, 3, 3, input_shape = (100, 15, 3), activation = 'relu'))

# Etape 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2eme couche de convolution
classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Etaoe 3 - Flattening
classifier.add(Flatten())

# Etape 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 4, activation = 'softmax'))

# Compilation
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_dir = '/home/julien/data_science/ng2/training_set'
test_dir = '/home/julien/data_science/ng2/test_set'
valid_dir = '/home/julien/data_science/ng2/valid_set'


training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (100, 15),
                                                 batch_size = 8,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = (100, 15),
                                            batch_size = 8,
                                            class_mode = 'sparse')

classifier.fit_generator(training_set,
                         samples_per_epoch = 170,
                         nb_epoch = 100,
                         validation_data = test_set,
                         nb_val_samples = 10)

#classifier.save('model_ng2.h5')

list_0_to_25 = sorted(os.listdir(valid_dir+"/0_to_25"))
list_26_to_50 = sorted(os.listdir(valid_dir+"/26_to_50"))
list_51_to_75 = sorted(os.listdir(valid_dir+"/51_to_75"))
list_76_to_100 = sorted(os.listdir(valid_dir+"/76_to_100"))

training_set.class_indices
good_predictions = []

predictions_0_to_25 = []
for elt in list_0_to_25: 
    test_image = image_utils.load_img(valid_dir+"/0_to_25/"+elt, target_size=(100, 15))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict_on_batch(test_image)
    predictions_0_to_25.append(result)

predictions_0_to_25 = np.array(predictions_0_to_25).reshape((len(predictions_0_to_25),4))  
predictions_0_to_25 = np.argmax(predictions_0_to_25, axis=1)
good_predictions.append(len(predictions_0_to_25[predictions_0_to_25 == 0])/len(list_0_to_25))

predictions_26_to_50 = []
for elt in list_26_to_50: 
    test_image = image_utils.load_img(valid_dir+"/26_to_50/"+elt, target_size=(100, 15))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict_on_batch(test_image)
    predictions_26_to_50.append(result)

predictions_26_to_50 = np.array(predictions_26_to_50).reshape((len(predictions_26_to_50),4))  
predictions_26_to_50 = np.argmax(predictions_26_to_50, axis=1)
good_predictions.append(len(predictions_26_to_50[predictions_26_to_50 == 1])/len(list_26_to_50))

predictions_51_to_75 = []
for elt in list_51_to_75: 
    test_image = image_utils.load_img(valid_dir+"/51_to_75/"+elt, target_size=(100, 15))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict_on_batch(test_image)
    predictions_51_to_75.append(result)

predictions_51_to_75 = np.array(predictions_51_to_75).reshape((len(predictions_51_to_75),4))  
predictions_51_to_75 = np.argmax(predictions_51_to_75, axis=1)
good_predictions.append(len(predictions_51_to_75[predictions_51_to_75 == 2])/len(list_51_to_75))

predictions_76_to_100 = []
for elt in list_76_to_100: 
    test_image = image_utils.load_img(valid_dir+"/76_to_100/"+elt, target_size=(100, 15))
    test_image = image_utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict_on_batch(test_image)
    predictions_76_to_100.append(result)

predictions_76_to_100 = np.array(predictions_76_to_100).reshape((len(predictions_76_to_100),4))  
predictions_76_to_100 = np.argmax(predictions_76_to_100, axis=1)
good_predictions.append(len(predictions_76_to_100[predictions_76_to_100 == 3])/len(list_76_to_100))

print(good_predictions)
