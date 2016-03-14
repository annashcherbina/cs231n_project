from Params import * 
import h5py 
from vis_utils import * 
import numpy as np 
import pylab as pl

weights=h5py.File('assignment3_weights_learnslow.hdf5','r') 
conv1=np.asarray(weights['layer_1']['param_0'])
print str(conv1.shape) 
grid=visualize_grid(conv1.transpose(0,2,3,1)) 
#print str(grid.shape) 
pl.imshow(grid.astype('uint8'))
pl.axis('off') 
pl.savefig('conv1_augment_dropout_regularization_weights.png') 


weights=h5py.File('assignment3_weights.hdf5','r') 
conv1=np.asarray(weights['layer_1']['param_0'])
print str(conv1.shape) 
grid=visualize_grid(conv1.transpose(0,2,3,1)) 
#print str(grid.shape) 
pl.imshow(grid.astype('uint8'))
pl.axis('off') 
pl.savefig('conv1_weights.png') 


weights=h5py.File('assignment3_freeze_and_stack_weights.hdf5','r') 
conv1=np.asarray(weights['layer_1']['param_0'])
print str(conv1.shape) 
grid=visualize_grid(conv1.transpose(0,2,3,1))  
#print str(grid.shape) 
pl.imshow(grid.astype('uint8'))
pl.axis('off') 
pl.savefig('conv1_freeze_and_stack_weights.png') 



weights=h5py.File('vgg_model_weights.hdf5','r') 
conv1=np.asarray(weights['layer_0']['param_0'])
print str(conv1.shape) 
grid=visualize_grid(conv1.transpose(0,2,3,1))  
#print str(grid.shape) 
pl.imshow(grid.astype('uint8'))
pl.axis('off') 
pl.savefig('vgg_weights.png') 


