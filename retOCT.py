# This neural network will identify a retinal OCT class
    # Either Normal, CNV, DME, or Multiple Drusen
# Based on CNN built in Udemy course by Eremenko and de Ponteves
# Dataset taken from Kaggle: Retinal OCT Images (optical coherence tomography)
# URL on 21/02/2020: https://www.kaggle.com/paultimothymooney/kermany2018

##########################################
# Step 1: Import necessary Keras modules #
##########################################

# keras.models Sequential defines the model as a sequence of layers,
#   ie from the input layer to the convolution layer to the pooling layer...
#   in a sequence
from keras.models import Sequential

# keras.layers Convolution2D is the convolution step, and in 2d as images are...
#   in 2 dimensions. This package will represent each convolutional layer
from keras.layers import Convolution2D

# keras.layers MaxPooling2D is the pooling step in 2 dimensions, used to...
#   continue on to the pooling step from the convolution step
from keras.layers import MaxPooling2D

# keras.layers Flatten used to turn all the pooled feature maps into one...
#   large feature vector
from keras.layers import Flatten

# keras.layers Dense used to create a fully connected layer just like a...
#   classic ANN
from keras.layers import Dense

# keras.layers Dropout removes a fraction of all weights, prevents overfitting
from keras.layers import Dropout

# The following 2 lines might be needed if running on Mac
    # uncomment below if duplicate library error occurs
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#############################
# Step 2: Convolution layer #
#############################
# Initialise the Convolutional Neural Network
#   called classifier as we are classifying images, as a Sequential object
classifier = Sequential()

# Add a Convolution2D layer to the NN
classifier.add(
    Convolution2D(
        # :filters: The number of filters we want to create
            # 32 is common practice to start with, then add more conv layers
        filters = 32, # so 32 filters -> 32 feature maps

        # :kernel_size: the size of the filter/feature detector/kernel
        kernel_size = (3,3),

        # :input_shape: The shape of each image;
            # For Theano backend:
                # 3 channels for RGB or 1 for B&W,
                # number of rows of pixels, number of columns of pixels
                # e.g. input_shape = (3, 256, 256)
            # For TensorFlow backend:
                # number of rows of pixels, number of columns of pixels
                # 3 channels for RGB or 1 for B&W,
                # e.g. input_shape = (256, 256, 3)
        input_shape = (256, 256, 3), # colour images all 256x256 as specified by
                                    # the dataset, using TensorFlow backend

        # :activation: Enter str of the activation function desired to ensure
            # non-linearity, in case of negative values from conv operation
        activation = "relu"
    )
)

######################################
# Step 3: Pooling and Dropout Layers #
######################################
# Add a pooling layer to the CNN
classifier.add(
    MaxPooling2D(
        # :pool_size: The size of the pooling filter applied to the feature map
            # 2x2 is very common for a pooling filter
        pool_size = (2, 2)
        # :strides: if not input, will default to pool_size
    )
)

# Dropout layer with rate of 0.2, so 20% of parameters set to 0.
classifier.add(Dropout(rate = 0.2))

#########################################################################
# Step 4: Additional Convolutional and Pooling Layers, Flattening Layer #
#########################################################################

classifier.add(
    Convolution2D(
        filters = 64,
        kernel_size = (3,3),
        activation = 'relu'
    )
)
classifier.add(
    MaxPooling2D(
        pool_size = (2,2)
    )
)
classifier.add(Dropout(rate = 0.3))

classifier.add(
    Convolution2D(
        filters = 128,
        kernel_size = (3,3),
        activation = 'relu'
    )
)
classifier.add(
    MaxPooling2D(
        pool_size = (2,2)
    )
)
classifier.add(Dropout(rate = 0.3))

# Add a flattening layer to the CNN
classifier.add(Flatten())

##################################
# Step 5: Fully Connected Layers #
##################################
# Add the first fully connected layer
classifier.add(
    Dense(
        # :units: The number of nodes in the Dense layer
            # Common practice to pick power of 2
        units = 128,

        # :activation: The desired activation function applied to the nodes
        activation = 'relu'
    )
)
classifier.add(Dropout(rate = 0.5))

# Add another fully connected layer
classifier.add(Dense(
    units = 64,
    activation = 'relu'
    )
)
classifier.add(Dropout(rate = 0.2))

# Add third and final fully connected layer for the output
classifier.add(
    Dense(
        # :units: Now we will have 4 output nodes, as there are 4 classes
        units = 4,

        # :activation: The softmax function will be used to return 4 values...
            # ... between 0 and 1, probability for each class
        activation = 'softmax'
    )
)

###########################
# Step 6: Compile the CNN #
###########################

classifier.compile(
    # :optimizer: Adjustment of the weights based on the Stochastic Gradient...
        # ... Descent algorithm, of which 'Adam' is an efficient version
        # 'Adam'/SGD is dependent on a loss function, which is defined next...
    optimizer = 'adam',

    # :loss: The loss function that we are looking to minimise
        # In this case, we use categorical cross entropy
            # Categorical because of the 4 classes of desired output
            # Cross entropy because of its efficacy
    loss = 'categorical_crossentropy',

    # :metrics: Used to evaluate the performance of the model based on parameter
        # 'accuracy' compares true to predicted values
    metrics = ['accuracy']
)

##############################
# Step 7: Image Augmentation #
##############################
# Image augmentation prevents overfitting to the training set, which would...
    # ... result in high accuracy on the training set but lower accuracy on...
    # ... the test set
# Overfitting can occur when there are only a few training values
# Augmentation applies random transformations to batches within the dataset...
    # ... resulting in theoretically more images to train with
    # Enriching the dataset without adding any more images

from keras.preprocessing.image import ImageDataGenerator

# :ImageDataGenerator: intercepts the images being fed to the NN, applies...
    # ... random transformations to the images, and feeds ONLY the...
    # ... transformed images to the NN.
    # As a result, the NN is only ever seeing new images each epoch, as the...
    # ... images it is being fed have been transformed differently than the last
train_datagen = ImageDataGenerator(
    # :rescale: Alters the values of each pixel by the scale factor
        # In this case, each pixel will contain a value from 0 -> 1 as max...
        # ... pixel value is 255
    rescale = 1./255,

    # :shear_range: Shear angle in counterclockwise direction in degrees
        # Shear transformation of the images
    shear_range = 0.1,

    # :zoom_range: If float, sets [lower, upper] zoom values to...
        # [1-zoom_range, 1+zoom_range]. Or just input [lower, upper]
    zoom_range = 0.2,

    # :horizontal_flip: Randomly flip inputs horizontally
    horizontal_flip = True
)

# In the testing/validation dataset, we will just rescale the images so pixels
    # have a value between 0 - 1
valid_datagen = ImageDataGenerator(rescale = 1./255)

###############################
# Step 8: Apply preprocessing #
###############################

# Define the training set as flow_from_directory
training_set = train_datagen.flow_from_directory(
    # Path to the training dataset
    'retOCTdata/train',

    # :target_size: size of the input images
    target_size = (256,256),

    # :batch_size: size of batches of random samples of images to be fed to NN,
        # after which the weights will be updated
    batch_size = 32,

    # :class_mode: 'binary' if 2 classes, else 'categorical'
    class_mode = 'categorical'
)

# Clarify which index each class is aligned to
print('CLASS INDICES: ' + str(training_set.class_indices))

# Define the test set as flow_from_directory
valid_set = valid_datagen.flow_from_directory(
    # Path to the test dataset, contains more samples than "valid"
    'retOCTdata/test',

    # :target_size: size of the input images
    target_size = (256,256),

    # :batch_size: size of batches of random samples of images to be fed to NN,
        # after which the weights will be updated
    batch_size = 32,

    # :class_mode: 'binary' if 2 classes, else 'categorical'
    class_mode = 'categorical'
)

###########################################
# Step 9: Apply NN to Preprocessed Images #
###########################################

# Import the checkpoint to run after each epoch
from keras.callbacks import ModelCheckpoint

# Specify checkpoint
classifierCheckpoint = ModelCheckpoint(
    filepath = 'retOCTmodels/1retOCTmodel.h5', # saving model to this path
    save_best_only = True,  # only saving the best model ie if loss < best loss
    monitor = 'val_loss')   # save dependent on validation loss value

callbacksList = [classifierCheckpoint]


classifier.fit_generator(
    # :generator: The generated dataset used to train the classifier NN
    generator = training_set,

    # :samples_per_epoch: Number of images per epoch
        # 83484 images in training set -> 2608.875 batches -> 2609 steps/batches
    steps_per_epoch = 2609,

    # :epochs: number of epochs
    epochs = 25,

    # :validation_data: The dataset with the validation data
    validation_data = valid_set,

    # :nb_val_samples: number of samples in the validation dataset
        # 968 images in the "test" directory -> 30.25 -> 31 steps/batches
    validation_steps = 31,

    # :callbacks: list of the callbacks desired for this model fitting
    callbacks = callbacksList
)
