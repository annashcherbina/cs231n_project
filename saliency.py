from Params import * 
from load_data import *
from preprocess import *
from vgg_like_convnet import *
from alexnet import * 
from vgg16_keras import * 
from pretrained import * 
import h5py 

import keras
import theano
import theano.tensor as T 

#for visualization:
from keras.utils.visualize_util import plot
from keras.optimizers import SGD, Adagrad, Adadelta 

import pickle
import numpy as np 

import skimage 
from skimage.io import imread 

import matplotlib.pyplot as plt 

def compile_saliency_function(model):
    """
    Compiles a function to compute the saliency maps and predicted classes
    for a given minibatch of input images.
    """
    inp = model.layers[0].get_input()
    outp = model.layers[-1].get_output()
    max_outp = T.max(outp, axis=1)
    saliency = theano.grad(max_outp.sum(), wrt=inp)
    max_class = T.argmax(outp, axis=1)
    return theano.function([inp], [saliency, max_class])

def show_images(img_original,img, saliency, max_class, title):
    classes = [str(x) for x in range(200)]
    # get out the first map and class from the mini-batch
    saliency = saliency[0]
    saliency = saliency[::-1].transpose(1, 2, 0)
    max_class = max_class[0]
    # plot the original image and the three saliency map variants
    plt.figure(figsize=(10, 10), facecolor='w')
    plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
    plt.subplot(2, 3, 1)
    plt.title('raw input')
    plt.imshow(img_original)
    plt.subplot(2,3,2) 
    plt.title('preprocessed input') 
    plt.imshow(img) 
    plt.subplot(2, 3, 4)
    plt.title('abs. saliency')
    plt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    plt.subplot(2, 3, 5)
    plt.title('pos. saliency')
    x = (np.maximum(0, saliency) / saliency.max())[:,:,0]
    plt.imshow(x)
    plt.subplot(2, 3, 6)
    plt.title('neg. saliency')
    x = (np.maximum(0, -saliency) / -saliency.min())[:,:,0]
    plt.imshow(x)
    plt.savefig('augmentation_and_dropout_and_regularization_pomagranite.png') 
    # plt.show()


def main(): 
    #load the test data 
    #load the model 
    data=h5py.File('imagenet.hdf5','r')
    X_train=np.asarray(data['X_train']) 
    Y_train=np.asarray(data['Y_train']) 
    X_valid=np.asarray(data['X_valid']) 
    Y_valid=np.asarray(data['Y_valid']) 
    X_test=np.asarray(data['X_test']) 
    #good image
    #img_original=imread('test_6628.JPEG').astype(np.uint8)
    #img=np.expand_dims(X_test[2113],axis=0)

    img_original=imread('test_8715.JPEG').astype(np.uint8)
    img=np.expand_dims(X_test[1862],axis=0)
    #training 
    #img_original=imread('n01910747_2.JPEG').astype(np.uint8) 
    #img=np.expand_dims(X_train[96435],axis=0) 
    print str(img.shape) 
    #pretrained_model=vgg_train(X_train,Y_train,batch_size=128,nb_epoch=200) 
    #pretrained_model=pretrained_finetune('assignment3_freeze_and_stack_weights.hdf5',True)
    pretrained_model=pretrained_finetune('assignment3_weights_learnslow.hdf5',False) 
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    pretrained_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    saliency_function=compile_saliency_function(pretrained_model) 
    saliency,max_class=saliency_function(img) 
    print "got saliency function!!!" 
    show_images(img_original,img,saliency,max_class,"Jellyfish (VGG-Like ConvNet)")
    

if __name__=="__main__": 
    main() 
