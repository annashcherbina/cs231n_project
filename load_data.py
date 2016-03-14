#This script loads the training, test, and validation data into a format that is compatible with keras.
import skimage
from skimage.io import ImageCollection,concatenate_images,imread
from skimage.transform import resize
from skimage.color import gray2rgb
import numpy as np
from os import listdir
from os.path import isfile, join
from Params import * 
from augment_data import * 

#convert a list of y_values to one-hot encoded version 
def one_hot_encode(y):
    N=len(y)
    #D=1000
    D=200 
    y_encoded=np.zeros((N,D))
    for i in range(len(y)):
        val=y[i]
        y_encoded[i,val]=1
    #print str(y_encoded[0])
    return y_encoded



#reads an image and converts grayscale to RGB so that all images have 3 color channels 
def imreadconvert(Xname):
    X=imread(Xname).astype(np.float32)
    if len(X.shape)==3:
        X=X.transpose(2,0,1)
        return X
    else:
        return gray2rgb(X).transpose(2,0,1)  
    
def load_validation_data(valid_dir,label_dict,sample):
    print "loading validation data!" 
    image_to_id=open(valid_dir+"val_annotations.txt",'r').read().split('\n')
    while '' in image_to_id:
        image_to_id.remove('') 
    image_names=[]
    image_labels=[]
    for entry in image_to_id:
        tokens=entry.split('\t')
        image_names.append(valid_dir+'images/'+tokens[0])
        image_labels.append(label_dict[tokens[1]])
    num_entries=int(round(len(image_names)*sample))
    image_names=image_names[0:num_entries]
    images=concatenate_images(ImageCollection(image_names,load_func=imreadconvert))
    print "image val shape:"+str(images.shape) 
    print "loaded validation data:"
    image_labels=np.asarray(image_labels[0:num_entries])
    image_labels=np.reshape(image_labels,(len(image_labels),))
    image_labels=one_hot_encode(image_labels) 
    return images, image_labels

    

def load_test_data(test_dir,sample):
    print "loading test data!"
    cur_dir=test_dir+"images/"
    onlyfiles = [f for f in listdir(cur_dir) if isfile(join(cur_dir, f))]
    onlyfiles=[cur_dir+f for f in onlyfiles] 
    numfiles=int(round(len(onlyfiles)*sample))
    onlyfiles=onlyfiles[0:numfiles] 
    images=concatenate_images(ImageCollection(onlyfiles,load_func=imreadconvert))
    print "loaded test data:"
    print str(images.shape) 
    return images 

def load_train_data(train_dir,label_dict,sample):
   print "loading training data!" 
   nsamples=int(round(sample*500))#500 images for each of 200 labels
   file_names=[]
   labels=[] 
   for label in label_dict:
       #print str(label) 
       cur_dir=train_dir+label+"/images" 
       onlyfiles = [f for f in listdir(cur_dir) if isfile(join(cur_dir, f))][0:nsamples]
       onlyfiles=[cur_dir+'/'+f for f in onlyfiles]
       file_names=file_names+onlyfiles
       cur_labels=nsamples*[label_dict[label]]
       labels=labels+cur_labels
   X_train=concatenate_images(ImageCollection(file_names,load_func=imreadconvert))
   print "loaded training data"
   print str(X_train.shape)
   Y_train=np.asarray(labels)
   Y_train=np.reshape(Y_train,(len(Y_train),))
   Y_train=one_hot_encode(Y_train) 
   print str(Y_train.shape) 
   return X_train,Y_train


#augment the training data by flipping horizontally, flipping vertically, scaling up crops, tinting, contrasting 
def augment_train_data(X_train,Y_train):
    N=X_train.shape[0] 
    toaugment=N/3 #use 20% of the data for each type of augmentation; overall our training data size will double 

    tinted_indices=np.random.randint(0,N,toaugment) 
    tinted=random_tint(X_train[tinted_indices]) 
    #print "tinted!" 
    contrast_indices=np.random.randint(0,N,toaugment) 
    contrasted=random_contrast(X_train[contrast_indices])
    #print "contrasted!" 
    cropped_indices=np.random.randint(0,N,toaugment) 
    cropped=random_crops(X_train[cropped_indices],tuple([55,55])) 
    print "cropped!" 
    horiz_flip_indices=np.random.randint(0,N,toaugment) 
    horiz_flipped=flip_horizontally(X_train[horiz_flip_indices]) 
    print "flipped horizontally!" 
    
    vert_flip_indices=np.random.randint(0,N,toaugment) 
    vert_flipped=flip_vertically(X_train[vert_flip_indices])
    print "flipped vertically!" 

    X_train=np.concatenate((X_train,tinted,contrasted,cropped,horiz_flipped,vert_flipped),axis=0) 
    Y_train=np.concatenate((Y_train,Y_train[tinted_indices],Y_train[contrast_indices],Y_train[cropped_indices],Y_train[horiz_flip_indices],Y_train[vert_flip_indices]),axis=0) 

    #X_train=np.concatenate((X_train,cropped,horiz_flipped,vert_flipped),axis=0) 
    #Y_train=np.concatenate((Y_train,Y_train[cropped_indices],Y_train[horiz_flip_indices],Y_train[vert_flip_indices]),axis=0) 
    return X_train,Y_train 

def load_data(train_dir,valid_dir,test_dir,labels,sample):
    #sample indicates the fraction of data to be used, between 0 and 1 
    if sample>1:
        sample=1
    if sample <0.01:
        sample=0.01
    label_dict=dict()
    labels=open(labels,'r').read().split('\n')
    while '' in labels:
        labels.remove('')
    for i in range(len(labels)):
        label_dict[labels[i]]=i
    print "built dictionary of labels (id --> number) "
    X_valid,Y_valid=load_validation_data(valid_dir,label_dict,sample)
    X_test=load_test_data(test_dir,sample)
    X_train,Y_train=load_train_data(train_dir,label_dict,sample) 
    print "augmenting training data!" 
    X_train,Y_train=augment_train_data(X_train,Y_train) 
    print "augmented X_train:"+str(X_train.shape) + ','+"augmented Y_train:"+str(Y_train.shape) 
    return X_train,Y_train,X_valid,Y_valid,X_test
