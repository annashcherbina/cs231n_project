from Params import *
from load_data import *
import h5py  
from augment_data import * 
import matplotlib.pyplot as plt 

'''
X_train,Y_train,X_valid,Y_valid,X_test=load_data(training_dir,valid_dir,test_dir,labels,sample)
del X_train 
del Y_train 
del X_valid 
del Y_valid 
X_test=X_test[0:10] 
f=h5py.File('augment_qc.hdf5','w')
dset_xtest=f.create_dataset("X_test",data=X_test[0:10]) 
f.flush() 
f.close() 
'''
data=h5py.File('augment_qc.hdf5','r')
X_test=np.asarray(data['X_test']) 
#print "data is loaded and ready for augmentation" 

#try all the augmentations 
tinted=random_tint(X_test) 
contrasted=random_contrast(X_test)
cropped=random_crops(X_test,tuple([55,55])) 
horiz_flipped=flip_horizontally(X_test) 
vert_flipped=flip_vertically(X_test) 
'''
print str(X_test[0].shape) 
print str(horiz_flipped[0].shape) 
print str(vert_flipped[0].shape) 
print str(cropped[0].shape) 
print str(contrasted[0].shape) 
print str(tinted[0].shape) 
'''
#visualize the results 
plt.figure(figsize=(10, 10), facecolor='w')
#plt.suptitle("Class: " + classes[max_class] + ". Saliency: " + title)
for i in range(4): 
    plt.subplot(2, 3, 1)
    plt.title('Original')
    plt.imshow(X_test[i].transpose(1,2,0))
    plt.subplot(2, 3, 2)
    plt.title('X-flipped') 
    plt.imshow(horiz_flipped[i].transpose(1,2,0))
    plt.subplot(2, 3, 3)
    plt.title('Y-flipped') 
    plt.imshow(vert_flipped[i].transpose(1,2,0))
    plt.subplot(2, 3, 4)
    plt.title('Cropped and Scaled') 
    plt.imshow(cropped[i].transpose(1,2,0))
    plt.subplot(2,3,5) 
    plt.title('Contrasted') 
    plt.imshow(contrasted[i].transpose(1,2,0))
    plt.subplot(2,3,6) 
    plt.title('Tinted') 
    plt.imshow(tinted[i].transpose(1,2,0))
    plt.savefig('data_augmentation_qc_'+str(i)+'.png') 
    #print str(X_test[i]) 
