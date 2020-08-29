# facial_landmark_detection

Creating a CNN model to detect facial landmark in an image.

Requirements

1. Tensorflow 1.8.0
2. Tkinter
3. Numpy
4. OpenCv 3

Description
Use Convolutional Neural Network to detect facial landmark in an image.

## We will use the following steps to create our text recognition model.

1. Collecting Dataset
2. Preprocessing Data
3. Creating Network Architecture
4. Defining Loss function
5. Training model
6. detect facial landmark in real-time

Dataset
We will use data provided by Large-scale CelebFaces Attributes (CelebA) Dataset. This is a huge dataset total of 2 GB images. Here I have used 150000 images for the training set with 0.08% images for validation dataset. This data contains text image segments which look like images shown below:

![intro](https://user-images.githubusercontent.com/58425689/91628517-2de2b600-e9e0-11ea-99f8-7e02775f3fda.png)

To download the dataset either you can directly download from this [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

# Preprocessing
Now we are having our dataset, to make it acceptable for our model we need to use some preprocessing. We need to preprocess both the input image and output labels. To preprocess our input image we will use followings:

1. Read .txt LandmarkData and cropdata file and convert into .csv file 
2. Read the image and convert into a gray-scale image
3. Crop only faces from image using cropdata and also subract LandmarkData with cropdata
4. Make each image of size (96,96)
5. make each landamrk point into (96,96) size
6. Expand image dimension as (96,96,1) to make it compatible with the input shape of architecture
7. Normalize the image pixel values by dividing it with 255.
8. Normalize the landamrk point values by subracting with 48 and then dividing it with 48

# Model Architecture
we will create our model architecture and train it with the preprocessed data.

# Model = CNN + Fully Connected Layer + mean square errors loss

Our model consists of three parts:

## 1. Convolution Neural Network Layer
A convolutional neural network (CNN, or ConvNet) is a class of deep learning, feed artificial neural networks that has successfully been applied to analyzing visual imagery. CNN compares any image piece by piece and the pieces that it looks for in an while detection is called as features. The convolutional neural network to extract features from the image

There are five main operations in the CNN:
    a) Convolution.
    b) ReLU.
    c) Pooling or Sub

### a) Convolution layer

The primary purpose of Convolution in case of a CNN is to extract features from the input image. Each convolution layer takes image as a batch input of four dimension N x Color-Channel x width x height. Kernels or filters are also four dimensional (Number of feature maps in, number of feature maps out, filter width and filter height) which are set of learnable parameters (weights and biases). In each convolution layer, four dimensional convolution is calculate between image batch and feature maps by dot p between them. After convolution only parameter that changes are image width and height.

### b) Rectified Linear Unit

An additional operation called ReLU has been used after
every Convolution operation. A Rectified Linear Unit (ReLU)
is a cell of a neural network which uses the following
activation function to calculate its output given x:
R(x) = Max(0,x) 

### c) Pooling Layer
In this layer, the dimensionality of the feature map
reduces to get shrink maps that would reduce the parameters
and computations. Pooling can be Max, Average or Sum.
Number of filters in convolution layer is same as the number
of output maps from pooling. Pooling takes input from
rectified feature maps and then downsized it according to
algorithm. 

## 2. Fully Connected Layer
This is the final layer where the actual classification
occurs where this layer takes downsized or shrink feature
maps obtained after convolution, ReLU and pooling layer
and flatten. It is a traditional Multi
uses a softmax activation function. Convol
layers generate high-level features. The purpose of the fully
connected layer is to use these features to classify into
various classes based on labels.

### This network architecture.
Let’s see the steps that we used to create the architecture:

![1](https://user-images.githubusercontent.com/58425689/91628799-296bcc80-e9e3-11ea-9e5b-ba80f6ad9ab4.png)
![2](https://user-images.githubusercontent.com/58425689/91628801-2a9cf980-e9e3-11ea-8b52-34b878407ec9.png)

### Loss Function
we use mean square errors loss function.

![download (2)](https://user-images.githubusercontent.com/58425689/91628871-a1d28d80-e9e3-11ea-8bfc-027a6e0cd0dc.png)

Test the model
Our model is now trained with images. Now its time to test the model. We can use our training model. 


![1](https://user-images.githubusercontent.com/58425689/91628913-173e5e00-e9e4-11ea-900b-f0ab5603df71.gif)
