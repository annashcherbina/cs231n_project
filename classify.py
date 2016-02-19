from Params import * 
from load_data import *
from preprocess import *
from vgg_like_convnet import *
from alexnet import * 
from vgg16_keras import * 

import keras
import theano
#for visualization:
from keras.utils.visualize_util import plot

import pickle

def main():
    #load data 
    X_train,Y_train,X_valid,Y_valid,X_test=load_data(training_dir,valid_dir,test_dir,labels,sample)
    #preprocess data by mean subtraction and normalization 
    X_train,X_valid,X_test=preprocess(X_train,X_valid,X_test)
    #del X_train
    del X_test

    #or load pre-processed data from a previously saved pickle:
    '''
    print "loading data from pickle..." 
    f=open("imagenetdata.pkl",'r')
    X_train=pickle.load(f)
    X_valid=pickle.load(f)
    X_test=pickle.load(f)
    Y_train=pickle.load(f)
    Y_valid=pickle.load(f)
    f.close()
    '''
    #print "loaded data from pickle" 
    #OPTIONAL: save loaded/pre-processed data to a pickle to save time in the future
    '''
    print "pickling preprocessed data!" 
    outf=open("imagenetdata_resized224.pkl","w")
    pickle.dump(X_train,outf)
    pickle.dump(X_valid,outf)
    pickle.dump(X_test,outf)
    pickle.dump(Y_train,outf)
    pickle.dump(Y_valid,outf)
    outf.close()
    print "done pickling preprocessed data!"
    '''
    #train a VGG-like convent
    '''
    vgg_model=vgg_train(X_train,Y_train)    
    train_scores=vgg_evaluate(vgg_model,X_train,Y_train)
    print "VGG-like net training scores:"+str(train_scores) 
    valid_scores=vgg_evaluate(vgg_model,X_valid,Y_valid)
    print "VGG-like net validation scores:"+str(valid_scores)
    #Visualize the pretty model
    plot(vgg_model,to_file="vgg_like_convnet.png") 
    '''
    #train a Keras version of the ConvNet implemented in Assignment#2 in class
    #TODO

    #train AlexNet
    '''
    alexnet_model=alexnet_train(X_train,Y_train)    
    train_scores=alexnet_evaluate(alexnet_model,X_train,Y_train)
    print "AlexNet training scores:"+str(train_scores) 
    valid_scores=alexnet_evaluate(alexnet_model,X_valid,Y_valid)
    print "AlexNet validation scores:"+str(valid_scores)
    #Visualize the pretty model
    plot(alexnet_model,to_file="alexnet_like_convnet.png") 
    '''
    
    #VGG-16 with pretrained weights
    vgg16_model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    vgg16_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    train_scores=vgg16_evaluate(vgg16_model,X_train,Y_train)
    print "vgg16 training scores:"+str(train_scores) 
    valid_scores=vgg16_evaluate(vgg16_model,X_valid,Y_valid)
    print "vgg16 validation scores:"+str(valid_scores)
    #Visualize the pretty model
    plot(vgg16_model,to_file="vgg16_convnet.png") 
    

if __name__=="__main__":
    main() 
