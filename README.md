# Ninja Gaiden 2 : analysis of gameplay through machine learning

The goal of this project is to analyze gameplay videos from video games and retrieve valuable information from it. For example, it could be to analyze a player's performance variations throughout the game which could help detect which parts are too unbalanced in difficulty, or to understand the best player's strategies in the form of statistics in order to reproduce them. For this project, I chose to work on Ninja Gaiden 2 for Xbox 360, as it's one of my favorite games and I would know what relevant features to extract from it. This project is done with Python.

## Data acquisition and preprocessing

I used videos from dangavsterGAMES's Youtube channel : https://www.youtube.com/playlist?list=PLpx5oke1hzifxaSuZPuaKbiSrcaAB79nw , I downloaded videos for chapter 2, 6 and 11 (there are 14 chapters in this game).

These videos require different steps of preprocessing. First, they have to be split into images. For that I used the software 'Free Video to JPG Converter' and I extracted a picture for every 100 frames, which gave me a dataset of around 3000 pictures. As a first step to this project, I wanted to do an analysis of the healthbar, which is located at the top-left corner of the screen :

![NG2 screenshot](http://image.noelshack.com/fichiers/2017/38/1/1505748875-ng2.png)

The healthbar can be divided into 3 parts :
 - Blue : health left
 - Black : health that will be regenerated after an area is cleared of enemies
 - Red : health that can only be refilled with a savepoint or a health item
 
It can be increased through the use of some items so its size isn't constant throughout the game.
 
Since the healthbar is always located in the same portion of the screen, I cropped the images with the software 'IrfanView' to make the computations easier by keeping only the relevant part of the screen :

![Health bar](http://image.noelshack.com/fichiers/2017/38/1/1505749444-ninja-gaiden-2-ch2-dangavster-1970.jpg)

These videos have movies for the launching of the console, Youtube's channel intro, ingame movies... which are parts that are not relevant to the gameplay, it would be bad to train a model on them or to make predictions on such images. That's what ```clean_movie_images.py``` is for : it reads images through the Keras module, then it identifies which part of the screenshots is always there in the gameplay videos, and absent elsewhere. In this case, this would be the tiny yellow part at the left of the healthbar which can be seen in the picture above. Here's how it works :

- The program imports a batch of 20 images extracted from gameplay parts. It is necessary to import many images, since the values are not strictly equal from one picture to another, because of the quality of the video encoding and the screenshots.
- Since an image can be seen as a 3D matrix (length, width, 3 color components), for the tiny yellow part that we want to analyze, the mean of each pixel for each color component is computed. The mean is computed for each image and we get 3 lists of means (one for each color component). Then we define upper and lower boundaries : 

```python
lower = 2*min(list_means) - max(list_means)
upper = 2*max(list_means) - min(list_means)
```
For an image to be classified as ingame screenshot (and not movie screenshot), the mean for each color component in the specified area will have to fall between those boundaries. Otherwise, the image is deleted from the folder.

## Building a neural network

Convolutional neural networks are one of the most efficients techniques to detect features in images, so I chose to build one to solve this problem. To do so, I worked with the library Keras, using a TensorFlow backend. 
