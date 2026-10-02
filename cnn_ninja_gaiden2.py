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

# Les images font 1000x25 (largeur x hauteur). Keras attend (hauteur, largeur) :
# on garde la barre horizontale avec assez de colonnes pour distinguer les seuils 25/50/75%
img_size = (25, 200)

# Etape 1 - Convolution
classifier.add(Convolution2D(8, 3, 3, input_shape = img_size + (3,), activation = 'relu'))

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

# Pas de flip, zoom ni shear : ils changent le sens ou la longueur apparente
# de la barre, donc le pourcentage de vie, sans changer l'étiquette
train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_dir = '/home/julien/data_science/ng2/training_set'
test_dir = '/home/julien/data_science/ng2/test_set'
valid_dir = '/home/julien/data_science/ng2/valid_set'


training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = img_size,
                                                 batch_size = 8,
                                                 class_mode = 'sparse')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = img_size,
                                            batch_size = 8,
                                            class_mode = 'sparse')

classifier.fit_generator(training_set,
                         samples_per_epoch = 170,
                         nb_epoch = 100,
                         validation_data = test_set,
                         nb_val_samples = len(test_set.filenames))

#classifier.save('model_ng2.h5')

# Evaluation sur le valid set, avec le même prétraitement qu'à l'entraînement (rescale 1/255)
classes = sorted(training_set.class_indices, key = training_set.class_indices.get)
confusion = np.zeros((len(classes), len(classes)), dtype = int)

for true_class in classes:
    for elt in sorted(os.listdir(valid_dir+"/"+true_class)):
        test_image = image_utils.load_img(valid_dir+"/"+true_class+"/"+elt, target_size = img_size)
        test_image = image_utils.img_to_array(test_image) / 255.
        test_image = np.expand_dims(test_image, axis = 0)
        result = classifier.predict_on_batch(test_image)
        confusion[training_set.class_indices[true_class], np.argmax(result)] += 1

good_predictions = confusion.diagonal() / confusion.sum(axis = 1)

for name, rate in zip(classes, good_predictions):
    print(name, rate)

# Lignes : vraie classe, colonnes : classe prédite
print(classes)
print(confusion)
