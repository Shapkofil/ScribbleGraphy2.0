import sys
import numpy as np
import pandas as pd
import pickle
import os

import cv2
import time
import gc

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from sklearn.utils import shuffle

import numpy.random as rng
import matplotlib.pyplot as plt

def initialize_weights(shape, name=None, dtype = None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None, dtype = None):
	return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = layers.Input(input_shape)
    right_input = layers.Input(input_shape)
    
    # Convolutional Neural Network
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(256, (4,4), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='sigmoid',
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias,
                          kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = layers.Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = tf.keras.models.Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net

def get_batch(batch_size,s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=False)
    
    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]
    
    # initialize vector for the targets
    targets=np.zeros((batch_size,))
    
    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)
        
        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category  
        else: 
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1,n_classes)) % n_classes
        
        pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,1)
    
    return pairs, targets

def generate(batch_size, s="train"):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)

def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, w, h = X.shape
    
    indices = rng.randint(0, n_examples,size=(N,))
    if language is not None: # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low,high),size=(N,),replace=False)

    else: # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes),size=(N,),replace=False)            
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
    test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,1)
    support_set = X[categories,indices,:,:]
    support_set[0,:,:] = X[true_category,ex2]
    support_set = support_set.reshape(N, w, h,1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image,support_set]

    return pairs, targets

def test_oneshot(model, N, k, s = "val", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N,s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct

if __name__ == '__main__':

	#Paths
	train_folder = "./images_background/"
	val_folder = './images_evaluation/'
	save_path = '/media/shapkofil/HDD/Machine_Learning/SNN'

	# Hyper parameters
	evaluate_every = 2000 # interval for evaluating on one-shot tasks
	batch_size = 32
	n_iter = 20000 # No. of training iterations
	N_way = 20 # how many classes for testing one-shot tasks
	n_val = 250 # how many one-shot tasks to validate on
	best = -1

	#Create a model
	model = get_siamese_model((105, 105, 1))
	model.summary()

	optimizer = tf.keras.optimizers.Adam(lr = 0.00006)
	model.compile(loss="binary_crossentropy",optimizer=optimizer)

	with open("pickles/train.pickle", "rb") as f:
	    (Xtrain, train_classes) = pickle.load(f)

	with open("pickles/val.pickle", "rb") as f:
	    (Xval, val_classes) = pickle.load(f)

	Xtrain = Xtrain.astype('float16')
	Xval = Xval.astype('float16')

	#Training Loop
	print("Starting training process!")
	print("-------------------------------------")
	t_start = time.time()
	for i in range(1, n_iter+1):
	    (inputs,targets) = get_batch(batch_size)
	    loss = model.train_on_batch(inputs, targets)
	    if i % evaluate_every == 0:
	        print("\n ------------- \n")
	        print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
	        print("Train Loss: {0}".format(loss)) 
	        val_acc = test_oneshot(model, N_way, n_val, verbose=True)
	        model.save_weights(os.path.join(save_path, 'weights.{}.h5'.format(i)))
	        if val_acc >= best:
	            print("Current best: {0}, previous best: {1}".format(val_acc, best))
	            best = val_acc
	    #Colect the garbage
	    gc.collect()


	
	#Saving architecture         
	with open('./finalModel/oneshot_architecture.json', 'w') as f:
	    f.write(model.to_json())

	#Save optimal weights
	model.save_weights('./finalModel/oneshot_weights.h5')

	#Save the whole model(weight included) as h5
	model.save('./finalModel/SCNN.h5')
	print("Model Saved!")