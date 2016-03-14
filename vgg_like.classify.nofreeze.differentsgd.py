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
#for visualization:
from keras.utils.visualize_util import plot
from keras.optimizers import SGD, Adagrad, Adadelta 

import pickle
import numpy as np 
import sys

def main():
    #or load pre-processed data from a previously saved hdf5 file:
    
    data=h5py.File('imagenet.transpose.individually.augment.contrast.tint.hdf5','r')
    X_train=np.asarray(data['X_train']) 
    Y_train=np.asarray(data['Y_train']) 
    X_valid=np.asarray(data['X_valid']) 
    Y_valid=np.asarray(data['Y_valid']) 
    X_test=np.asarray(data['X_test']) 
    
    pretrained_model=vgg_train(weights="vgg_like_weights_dropout_regularization_augmenteddataTintContrast.stupidsgd.hdf5")
    sgd = SGD(lr=1e-1)#, decay=1e-6, momentum=0.9, nesterov=True)
    pretrained_model.compile(optimizer=sgd, loss='categorical_crossentropy',trainLayersIndividually=0)    
    print "compilation finished, fitting model" 
    print "pretrained_model.trainLayersIndividually:"+str(pretrained_model.trainLayersIndividually) 
    if pretrained_model.trainLayersIndividually==1: 
        train_epochs=5 
    else: 
        train_epochs=10     
    history=pretrained_model.fit(X_train, Y_train, 128,train_epochs,validation_data=tuple([X_valid,Y_valid]),verbose=1,show_accuracy=True)
    pretrained_model.save_weights("vgg_like_weights_dropout_regularization_augmenteddataTintContrast.stupidsgd.hdf5",overwrite=True) 
    class_predictions=pretrained_model.predict_classes(X_test) 
    np.savetxt('vgg_like_class_predictions_dropout_regularization_augmenteddataTintContrast.stupidsgd.txt',class_predictions,fmt='%i',delimiter='\t') 
    train_scores=pretrained_evaluate(pretrained_model,X_train,Y_train)
    print "pretrained model training scores:"+str(train_scores) 
    valid_scores=pretrained_evaluate(pretrained_model,X_valid,Y_valid)
    print "pretrained validation scores:"+str(valid_scores)


    print "writing out the predictions file" 
    predictions=open('vgg_like_class_predictions_dropout_regularization_augmenteddataTintContrast.stupidsgd.txt','r').read().split('\n') 
    while '' in predictions: 
        predictions.remove('') 
        
    wnids=open(labels,'r').read().split('\n') 
    while '' in wnids: 
        wnids.remove('') 

    cur_dir=test_dir+"images/"
    onlyfiles = [f for f in listdir(cur_dir) if isfile(join(cur_dir, f))]
    entries=10000
    outf=open('vgg_like_class_predictions_dropout_regularization_augmenteddataTintContrast.formatted.stupidsgd.txt','w') 

    for i in range(entries): 
        image_name=onlyfiles[i] 
        predict_index=int(predictions[i])
        wnid1=wnids[predict_index]
        outf.write(image_name+'\t'+str(wnid1)+'\n')

if __name__=="__main__":
    main() 
