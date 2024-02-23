#!python/bin

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.random.set_seed(1)
import random
random.seed(1)
from pickle import FALSE
import json
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from split_w_stratification import *

############################


def eul2quat(ax, ay, az, atol=1e-8):
    # Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    # Args:
    # ax: X rotation angle in radians.
    # ay: Y rotation angle in radians.
    # az: Z rotation angle in radians.
    # atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    # Return:
    # Numpy array with three entries representing the vectorial component of the quaternion.

    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r = np.zeros((3, 3))
    r[0, 0] = cz * cy
    r[0, 1] = cz * sy * sx - sz * cx
    r[0, 2] = cz * sy * cx + sz * sx

    r[1, 0] = sz * cy
    r[1, 1] = sz * sy * sx + cz * cx
    r[1, 2] = sz * sy * cx - cz * sx

    r[2, 0] = -sy
    r[2, 1] = cy * sx
    r[2, 2] = cy * cx

    # Compute quaternion:
    qs = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs, 0.0, atol):
        i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3
        w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
        qv[i] = 0.5 * w
        qv[j] = (r[i, j] + r[j, i]) / (2 * w)
        qv[k] = (r[i, k] + r[k, i]) / (2 * w)
    else:
        denom = 4 * qs
        qv[0] = (r[2, 1] - r[1, 2]) / denom;
        qv[1] = (r[0, 2] - r[2, 0]) / denom;
        qv[2] = (r[1, 0] - r[0, 1]) / denom;
    return qv


def similarity3D_parameter_space_random_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale, n):
    # Create a list representing a random (uniform) sampling of the 3D similarity transformation parameter space. As the
    # SimpleITK rotation parametrization uses the vector portion of a versor we don't have an
    # intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    # versor.
    # Args:
    # thetaX, thetaY, thetaZ: Ranges of Euler angle values to use, in radians.
    # tx, ty, tz: Ranges of translation values to use in mm.
    # scale: Range of scale values to use.
    # n: Number of samples.
    # Return:
    # List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).

    theta_x_vals = (thetaX[1] - thetaX[0]) * np.random.random(n) + thetaX[0]
    theta_y_vals = (thetaY[1] - thetaY[0]) * np.random.random(n) + thetaY[0]
    theta_z_vals = (thetaZ[1] - thetaZ[0]) * np.random.random(n) + thetaZ[0]
    tx_vals = (tx[1] - tx[0]) * np.random.random(n) + tx[0]
    ty_vals = (ty[1] - ty[0]) * np.random.random(n) + ty[0]
    tz_vals = (tz[1] - tz[0]) * np.random.random(n) + tz[0]
    s_vals = (scale[1] - scale[0]) * np.random.random(n) + scale[0]
    res = list(zip(theta_x_vals, theta_y_vals, theta_z_vals, tx_vals, ty_vals, tz_vals, s_vals))
    return [list(eul2quat(*(p[0:3]))) + list(p[3:7]) for p in res]


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size, dim, metadata, shuffle, imageDirectory, covariates, aug,
                 extention_data,zero_or_one=0):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.metadata = metadata
        self.imageDirectory = imageDirectory
        self.covariates = covariates
        self.aug = aug
        self.extention_data = extention_data
        self.zero_or_one = zero_or_one
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, w = self.__data_generation(list_IDs_temp)

        return X, y, w

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, len(self.extention_data)+2))
        clinical_data = np.empty((self.batch_size, self.metadata), dtype='float32')
        y = np.empty((self.batch_size), dtype=int)  # contains covariate
        w = np.empty((self.batch_size), dtype='float32')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #start augmenting variable to 0
            augmenting=0
            # Store sample
            imgs={} #declare dictionary to save metric maps
            aff_img = sitk.ReadImage('/home/milton.camachocamach/scratch/second_project/affine/' + str(ID[self.zero_or_one]) + '.nii.gz')
            # load jacobians and augment jacobinas
            jbn_img = sitk.ReadImage('/home/milton.camachocamach/scratch/second_project/jacobians/' + str(ID[self.zero_or_one]) + '.nii.gz')
            # for every extention data read the data
            for n, ext in enumerate(self.extention_data):
                if n == 0:
                    #if we want to augment
                    if (self.aug == "True"):
                        #test for agumentation randomness
                        if np.random.uniform() >= .5:
                            #set augmenting variable to 1
                            augmenting=1
                            #save image into the dictionary
                            imgs[str(ext)] = sitk.ReadImage(self.imageDirectory +str(ext)+'/'+ str(ID[self.zero_or_one]) + '__' + str(ext) + '.nii.gz')
                            #create augmentation
                            center = np.array(
                                imgs[str(ext)].TransformContinuousIndexToPhysicalPoint(np.array(imgs[str(ext)].GetSize()) / 2.0))
                            aug_transform = sitk.Similarity3DTransform()
                            aug_transform.SetCenter(center)
                            aug_parameters = similarity3D_parameter_space_random_sampling(thetaX=(-np.pi / 40.0, np.pi / 40.0),
                                                                                        thetaY=(-np.pi / 40.0, np.pi / 40.0),
                                                                                        thetaZ=(-np.pi / 40.0, np.pi / 40.0),
                                                                                        tx=(-5.0, 5.0), ty=(-5.0, 5.0),
                                                                                        tz=(-5.0, 5.0), scale=(0.9, 1),
                                                                                        n=1)
                            aug_transform.SetParameters(aug_parameters[0])
                            #augment image
                            aug_image = sitk.Resample(imgs[str(ext)], imgs[str(ext)], aug_transform)
                            imgs[str(ext)] = sitk.GetArrayFromImage(aug_image)
                            #augment intensities image
                            aug_aff_image = sitk.Resample(aff_img, aff_img, aug_transform)
                            aff = sitk.GetArrayFromImage(aug_aff_image)                            
                            #augment jacobian image
                            aug_jbn_image = sitk.Resample(jbn_img, jbn_img, aug_transform)
                            jbn = sitk.GetArrayFromImage(aug_jbn_image)
                            #augment clinical data
                            clinical_data_temp = self.covariates[np.where(self.covariates[:, 0] == ID[self.zero_or_one])][0,2:4]
                            clinical_data[i,] = [clinical_data_temp[0], np.random.uniform(clinical_data_temp[0]-clinical_data_temp[0]*0.1, clinical_data_temp[0]+clinical_data_temp[0]*0.1)]
                        else:
                            #get original image
                            imgs[str(ext)] = sitk.ReadImage(self.imageDirectory +str(ext)+'/'+ str(ID[self.zero_or_one]) + '__' + str(ext) + '.nii.gz')
                            imgs[str(ext)] = sitk.GetArrayFromImage(imgs[str(ext)])
                            #get original intensities
                            aff = sitk.GetArrayFromImage(aff_img)
                            #get original jacobian
                            jbn = sitk.GetArrayFromImage(jbn_img)
                            #get original clinical data
                            clinical_data[i,] = self.covariates[np.where(self.covariates[:, 0] == ID[self.zero_or_one])][0,2:4]
                    else:
                        #get original image
                        imgs[str(ext)] = sitk.ReadImage(self.imageDirectory +str(ext)+'/'+ str(ID[self.zero_or_one]) + '__' + str(ext) + '.nii.gz')
                        imgs[str(ext)] = sitk.GetArrayFromImage(imgs[str(ext)])
                        #get original intensities
                        aff = sitk.GetArrayFromImage(aff_img)
                        #get original jacobian
                        jbn = sitk.GetArrayFromImage(jbn_img)
                        #get original clinical data
                        clinical_data[i,] = self.covariates[np.where(self.covariates[:, 0] == ID[self.zero_or_one])][0,2:4]
                #if is not the first map
                else:
                    #if augmenting variable was set to 1 for the first image
                    if augmenting==1:
                        #augment current map
                        imgs[str(ext)] = sitk.ReadImage(self.imageDirectory +str(ext)+'/'+ str(ID[self.zero_or_one]) + '__' + str(ext) + '.nii.gz')
                        aug_image = sitk.Resample(imgs[str(ext)], imgs[str(ext)], aug_transform)
                        imgs[str(ext)] = sitk.GetArrayFromImage(aug_image)
                    else:
                        #get original image
                        imgs[str(ext)] = sitk.ReadImage(self.imageDirectory +str(ext)+'/'+ str(ID[self.zero_or_one]) + '__' + str(ext) + '.nii.gz')
                        imgs[str(ext)] = sitk.GetArrayFromImage(imgs[str(ext)])
                #get the images into a single 4D image
                if n == 0:
                    Xn = np.float32(imgs[str(ext)].reshape(self.dim[0], self.dim[1], self.dim[2], 1))
                else:
                    Xn = np.concatenate((Xn,np.float32(imgs[str(ext)].reshape(self.dim[0], self.dim[1], self.dim[2], 1))), axis=3)
            #store 4D image
            X[i,] = np.concatenate((Xn,np.float32(jbn.reshape(self.dim[0], self.dim[1], self.dim[2], 1)),np.float32(aff.reshape(self.dim[0], self.dim[1], self.dim[2], 1))), axis=3)
            #store class
            y[i,] = self.covariates[np.where(self.covariates[:, 0] == ID[self.zero_or_one])][0, 1]  # gets the covariate value for a given ID.
            w[i,] = self.covariates[np.where(self.covariates[:, 0] == ID[self.zero_or_one])][0, 4]

        return [X, clinical_data], y, w  # This line will take care of outputing the inputs for training and the labels

def sfcn_simpler(inputLayer):
    initializer = tf.keras.initializers.HeUniform(seed=0)
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)

    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)

    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)

    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)
 
    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)

    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=ReLU()(x)

    #block 7, different from paper
    x=AveragePooling3D(padding='same')(x)
    x=Dropout(.2)(x)
    x = Flatten(name="flat1")(x)
    x = Dense(units=16)(x)
    z=Dense(units=1, activation='sigmoid',name="dense1")(x)

    return z

def sfcn_simpler_as(inputLayer):
    initializer = tf.keras.initializers.HeUniform(seed=0)
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)

    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)

    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)

    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)

    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)

    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1), kernel_initializer=initializer, bias_initializer='zeros', padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=AveragePooling3D(padding='same')(x)
    x=ReLU()(x)

    #block 7
    x=Dropout(.2)(x)
    x = Flatten(name="flat1")(x)
    clinical = Concatenate()([x, inputLayer[1]])
    clinical = Dense(units=16)(clinical)
    z=Dense(units=1, activation='sigmoid',name="dense1")(clinical)

    return z

def sfcn_simpler_regu(inputLayer,regu):
    initializer = tf.keras.initializers.HeUniform(seed=0)
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)

    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)

    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)

    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)

    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)

    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=AveragePooling3D(padding='same')(x)
    x=ReLU()(x)

    #block 7
    x=Dropout(.2)(x)
    x = Flatten(name="flat1")(x)
    x = Dense(units=16)(x)
    z=Dense(units=1, activation='sigmoid',name="dense1")(x)

    return z

def sfcn_simpler_regu_as(inputLayer,regu):
    initializer = tf.keras.initializers.HeUniform(seed=0)
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)

    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)

    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)

    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)

    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)

    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1), kernel_initializer=initializer, kernel_regularizer=l1(regu), bias_initializer='zeros', padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=AveragePooling3D(padding='same')(x)
    x=ReLU()(x)

    #block 7
    x=Dropout(.2)(x)
    x = Flatten(name="flat1")(x)
    clinical = Concatenate()([x, inputLayer[1]])
    clinical = Dense(units=16)(clinical)
    z=Dense(units=1, activation='sigmoid',name="dense1")(clinical)

    return z

######################### CONFIG PARAM #########################
# For multi gpu use - always provide: nbBatch mod nbGPUs = 0
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

covariatesFile = 'metadata.csv'
imageDirectory = 'path to directory with the images' 

###
#   PARAMETERS
###

def pmts(modelname, first_train, model_loaded, epochs, split, lr, ld, aug, extensions, use_age_sex, regu, batchsize):
    return {
    'covName': 'group_dummy',
    'metadata1': 'sex_dummy',
    'metadata2': 'age_znorm',
    'modelname': modelname,
    'aug': aug,
    'batch_size': batchsize,
    'num_epochs': epochs,
    'learning_rate': lr,
    'learning_decay': ld,
    'regu': float(regu) if regu != 'False' else 'False',
    'first_train': first_train,
    'num_metadata': 2,
    'age_and_sex' : use_age_sex,
    'ext_data': extensions.split(','),
    'model_loaded': model_loaded,
    'explainability': False,
    'ran_split': split,
    'imagex': 160,
    'imagey': 192,
    'imagez': 160
    }
#                                                                                 
parameters = pmts(sys.argv[1], #pmts(modelname (str),
                  sys.argv[2], #first_train (str)
                  sys.argv[3], #model_loaded (str)
                  int(sys.argv[4]), #epochs (int)
                  int(sys.argv[5]), #split (int)
                  float(sys.argv[6]), #lr (float)
                  float(sys.argv[7]), #ld (float)
                  sys.argv[8], #aug (str)
                  sys.argv[9], #extensions (str)
                  sys.argv[10], #use_age_sex (str)
                  sys.argv[11], #regu (str-float)
                  int(sys.argv[12])) #batchsize (int)

# define working directory
working_dir = 'path to where the models are saved'
loading_dir = 'path to where the models are loaded from'

aug = parameters['aug']
first_train = parameters['first_train']
n_e = parameters['num_epochs']

num_metadata = 2
multigpu = False
if multigpu:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# define optimizer
Optimizer = Adam(parameters['learning_rate'], beta_1=0.9, epsilon=1e-01, amsgrad=True)

# define Loss
Loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, name='binary_crossentropy')

# define metrics
Metrics = [ 
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
            ]

# Get all images index
cov = pd.read_csv(covariatesFile)

# lets create binned age based on the mean age
conditions_age = [(cov['age'] <= cov['age'].mean()),
                  (cov['age'] > cov['age'].mean())
                  ]

v_namse = ['younger', 'older']

# create new column 'binned_age' to stratify with
cov['binned_age'] = np.select(conditions_age, v_namse)
IDs_list = cov['Subject'].to_numpy()
IDs = IDs_list

####
#crete a weight for the diferent studies
studies = cov.study.unique()
w_of_studies = {}
w_of_pd_studies = {}
w_of_hc_studies = {}

for i in studies:
    ith_study = cov[cov.study == i]
    ith_study_pd = cov.loc[(cov['study'] == i) & (cov['group_dummy'] == 1)]
    ith_study_hc = cov.loc[(cov['study'] == i) & (cov['group_dummy'] == 0)]
    w_of_studies[i] = 1 - len(ith_study)/len(cov)
    w_of_pd_studies[str(i)+'_PD'] = 1 - len(ith_study_pd)/len(cov[cov['group_dummy'] == 1])
    w_of_hc_studies[str(i)+'_HC'] = 1 - len(ith_study_hc)/len(cov[cov['group_dummy'] == 0])

#add weights to cov
cov['sample_weights'] = cov[['study','group_dummy']].apply(lambda x : w_of_pd_studies[x[0]+'_PD'] if (x[1] == 1) else w_of_hc_studies[x[0]+'_HC'], axis=1)

# Extract covariates (creates an array with patient ID in column 0, covariate (e.g. PD or HC) in column 1, and other covariates to adjust for in next columns. to pass to data generator)
covs = pd.concat([cov.loc[0:, "Subject"], cov.loc[0:, parameters['covName']], cov.loc[0:, parameters['metadata1']],
                  cov.loc[0:, parameters['metadata2']], cov.loc[0:,"sample_weights"], cov.loc[0:, "age"], 
                  cov.loc[0:, "study"], cov.loc[0:, "group"]],
                 axis=1)
covs = covs.to_numpy()

# create the splits with stratification
train, validation, test = split_stratified_into_train_val_test(cov,
                                                                stratify_colnames=['group_dummy','binned_age','sex_dummy'], 
                                                                frac_train=0.75,
                                                                frac_val=0.1,
                                                                frac_test=0.15,
                                                                random_state=parameters['ran_split'])

# create some folders
try:
    os.mkdir(working_dir)
except:
    print('already created')
try:
    os.mkdir(working_dir + parameters['modelname'])
except:
    print('already created')
try:
    os.mkdir(working_dir + parameters['modelname'] + '/internal_splits')
except:
    print('already created')

# save the new splits as npy
np.save(working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_TestIds.npy',
        test)
np.save(working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_TrainingIds.npy',
        train)
np.save(working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_ValidationIds.npy',
    validation)

# save the new splits as csv
train.to_csv(working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_train.csv')
validation.to_csv(working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_validate.csv')
test.to_csv(working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_test.csv')

# load npys
IDs_test = np.load(
    working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_TestIds.npy',
    allow_pickle=True)
IDs_train = np.load(
    working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_TrainingIds.npy',
    allow_pickle=True)
IDs_valid = np.load(
    working_dir + parameters['modelname'] + '/internal_splits/' + parameters['modelname'] + '_ValidationIds.npy',
    allow_pickle=True)


# save the hyperparamenters
pd.DataFrame.from_dict(parameters, orient='index').to_csv(working_dir + parameters['modelname'] + '/' + 'parameters_' + parameters['modelname'] + '.csv')

######################### Main #########################
savedModelName = working_dir + parameters['modelname'] + '/' + parameters['modelname'] + '.hdf5'

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(savedModelName, monitor='val_loss', verbose=2,
                                                         save_best_only=True, include_optimizer=True,
                                                         save_weights_only=False, mode='auto',
                                                         save_freq='epoch')


# Generates training and validation batches
training_generator = DataGenerator(IDs_train, parameters['batch_size'],
                                   (parameters['imagex'], parameters['imagey'], parameters['imagez']),
                                   parameters['num_metadata'], True,
                                   imageDirectory,
                                   covs, parameters['aug'], parameters['ext_data'], 0)
valid_generator = DataGenerator(IDs_valid, parameters['batch_size'],
                                (parameters['imagex'], parameters['imagey'], parameters['imagez']),
                                parameters['num_metadata'], True,
                                imageDirectory,
                                covs, False, parameters['ext_data'], 0)

##
#   Declaration
##

if parameters['first_train'] == 'True':
    # with strategy.scope():
    print('starting from scratch')
    inputA = Input(shape=(parameters['imagex'], parameters['imagey'], parameters['imagez'], len(parameters['ext_data'])+2), name="InputA")
    inputB = Input(shape=(parameters['num_metadata']), name="InputB")
    if parameters['age_and_sex'] == 'True':
        if parameters['regu'] == 'False':
            z = sfcn_simpler_as([inputA,inputB])
        else:
            z = sfcn_simpler_regu_as([inputA,inputB],parameters['regu'])
    else:
        if parameters['regu'] == 'False':
            z = sfcn_simpler([inputA,inputB])
        else:
            z = sfcn_simpler_regu([inputA,inputB],parameters['regu'])
    model = Model(inputs=[inputA,inputB], outputs=[z])
    model.summary()
    model.compile(loss=Loss, optimizer=Optimizer, metrics=Metrics)
else:
    model = tf.keras.models.load_model(loading_dir+'/'+parameters['model_loaded']+'/'+parameters['model_loaded']+'.hdf5')

print('will start training')

history = model.fit(training_generator, 
                    epochs=parameters['num_epochs'], 
                    validation_data=valid_generator,
                    callbacks=[checkpoint_callback, 
                                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
                               ],
                    verbose=2)

# Get the dictionary containing each metric and the loss for each epoch
history_dict = history.history

# Save it under the form of a json file
try:
    json.dump(history_dict, open(working_dir + parameters['modelname'] + '/' + parameters['modelname'] + '.json', 'w'))
except:
    pd.DataFrame(history_dict).to_csv(working_dir + parameters['modelname'] + '/' + parameters['modelname'] + '.csv')

# plot and save the metric graph
def plot_metrics(metric_numpy, ylabel, x_label, title, save=True, file_path='resultados/plots/accuracy_',
                 file_name='metrics_plot'):
    '''This functions plots and could save the necessary plots'''
    plt.figure()
    for i in metric_numpy:
        plt.plot(i)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x_label)
    plt.legend(['Training', 'Validation'], loc='upper left')
    if save:
        plt.savefig(file_path + '/' + file_name + '.png')

# create a plot
plot_metrics([history_dict['accuracy'], history_dict['val_accuracy']], 'Accuracy', 'Epochs', title='Accuracy',
             save=True, file_path=working_dir + parameters['modelname'], file_name=parameters['modelname'] + '_acc')
plot_metrics([history_dict['loss'], history_dict['val_loss']], 'Loss', 'Epochs', title='Loss', save=True,
             file_path=working_dir + parameters['modelname'], file_name=parameters['modelname'] + '_loss')
plot_metrics([history_dict['precision'], history_dict['val_precision']], 'Precision', 'Epochs', title='Precision', save=True,
             file_path=working_dir + parameters['modelname'],
             file_name=parameters['modelname'] + '_precision')
plot_metrics([history_dict['recall'], history_dict['val_recall']], 'Recall', 'Epochs', title='Recall', save=True,
             file_path=working_dir + parameters['modelname'],
             file_name=parameters['modelname'] + '_recall')

rest = True
if rest:
    # reload the best performing model for the evaluation
    model.load_weights(working_dir + parameters['modelname'] + '/' + str(parameters['modelname']) + '.hdf5')
    model.compile(loss=Loss, 
                optimizer=Optimizer, 
                metrics=Metrics)
    model.trainable = False


    def evaluation(model, data_generation, name_of_split):
        loss, acc, precision, recall = model.evaluate(data_generation, verbose=2)
        print(name_of_split + " loss: {:5.2f}%".format(loss))
        print(name_of_split + ", accuracy: {:5.2f}%".format(100 * acc))
        print(name_of_split + ", precision: {:5.2f}%".format(precision))
        print(name_of_split + ", recall: {:5.2f}%".format(recall))
        try:
            pd.DataFrame.from_dict({'loss': loss, 'accuracy': acc, 'precision':precision, 'recal':recall},
                                orient='index').to_csv(
                working_dir + parameters['nfold']+ '/' + parameters['modelname'] + '/' + name_of_split + '_' + parameters[
                    'modelname'] + '.csv')
        except:
            print('donothing')


    # evaluate training
    print(parameters['modelname'] + " training")
    evaluation(model, training_generator, 'Training')

    # evaluate validation
    print(parameters['modelname'] + " validation")
    evaluation(model, valid_generator, 'Validation')

    # Generate testing data
    testing_generator = DataGenerator(IDs_test, 1,
                                      (parameters['imagex'], parameters['imagey'], parameters['imagez']), num_metadata,
                                      False,
                                      imageDirectory, covs, False, parameters['ext_data'], 0)

    # Evaluate the model test model
    print(parameters['modelname'] + " Testing")
    evaluation(model, testing_generator, 'Testing')