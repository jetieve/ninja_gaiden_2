#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 22:58:05 2017

@author: julien
"""
import numpy as np
from keras.preprocessing import image as image_utils
import os

# Removing not in-game pictures
#movie_dir = '/home/julien/data_science/ng2/health_bar/movie_test/movie/'
not_movie_dir = '/home/julien/data_science/ng2/movie_clean/'
full_dataset_dir = '/home/julien/data_science/ng2/health_bar/'
#movie_img_list = []
not_movie_img_list = []
moy_list = [[],[],[]]

# Algorithme s'exécutant dans un répertoire avec 20 images ingame
# La partie gauche de l'image est toujours la même ingame, et variable en 
# non-ingame : on va calculer des seuils pour la moyenne des 3 composants
# de chaque pixel afin de savoir si une image est ingame ou non
not_movie_list = sorted(os.listdir(not_movie_dir))
for mv in not_movie_list: 
    test_image = image_utils.load_img(not_movie_dir+mv, target_size=(25, 1000))
    test_image = image_utils.img_to_array(test_image)
    moyenne_1 = 0
    moyenne_2 = 0
    moyenne_3 = 0
    for i in range(25):
        for j in range(7):
                moyenne_1 += test_image[i][j][0]
                moyenne_2 += test_image[i][j][1]
                moyenne_3 += test_image[i][j][2]
    moyenne_1 /= 135
    moyenne_2 /= 135
    moyenne_3 /= 135
    moy_list[0].append(moyenne_1)
    moy_list[1].append(moyenne_2)
    moy_list[2].append(moyenne_3)
    test_image = np.expand_dims(test_image, axis=0)
    not_movie_img_list.append(test_image)
    
seuil_min_1 = 2*min(moy_list[0]) - max(moy_list[0])
seuil_max_1 = 2*max(moy_list[0]) - min(moy_list[0])
seuil_min_2 = 2*min(moy_list[1]) - max(moy_list[1])
seuil_max_2 = 2*max(moy_list[1]) - min(moy_list[1])
seuil_min_3 = 2*min(moy_list[2]) - max(moy_list[2])
seuil_max_3 = 2*max(moy_list[2]) - min(moy_list[2])

# Maintenant que les seuils sont calculés, on peut supprimer les images
# non-ingame dans le dossier contenant toutes les images
moy_list = [[],[],[]]

full_dataset_list = sorted(os.listdir(full_dataset_dir))
for i in range(4):
    full_dataset_list.pop(0) # ne pas prendre en compte les répertoires de classification

for pic in full_dataset_list: 
    test_image = image_utils.load_img(full_dataset_dir+pic, target_size=(25, 1000))
    test_image = image_utils.img_to_array(test_image)
    moyenne_1 = 0
    moyenne_2 = 0
    moyenne_3 = 0
    for i in range(25):
        for j in range(7):
                moyenne_1 += test_image[i][j][0]
                moyenne_2 += test_image[i][j][1]
                moyenne_3 += test_image[i][j][2]
    moyenne_1 /= 135
    moyenne_2 /= 135
    moyenne_3 /= 135
    moy_list[0].append(moyenne_1)
    moy_list[1].append(moyenne_2)
    moy_list[2].append(moyenne_3)
    if moyenne_1 < seuil_min_1 or moyenne_1 > seuil_max_1 or moyenne_2 < seuil_min_2 or moyenne_2 > seuil_max_2 or moyenne_3 < seuil_min_3 or moyenne_3 > seuil_max_3:
        os.remove(full_dataset_dir+pic)
    