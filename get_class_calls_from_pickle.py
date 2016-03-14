#saves model predictions to a text file by pulling them out of the pickle that was generated when the model was trained 
import pickle
import sys  
import numpy 

pickle_file=open(sys.argv[1],'rb') 
data1=pickle.load(pickle_file) 
data2=pickle.load(pickle_file) 
data3=pickle.load(pickle_file) 
data4=pickle.load(pickle_file) 
data5=pickle.load(pickle_file) 
#numpy.savetxt('validation_weights.txt',data1,fmt='%i',delimiter="\t")
#numpy.savetxt('test_weights_raw.txt',data2,fmt='%i',delimiter="\t")
#numpy.savetxt('test_weights_classes.txt',data3,fmt='%i',delimiter="\t")
numpy.savetxt('test_one_epoch.txt',data5,fmt='%i',delimiter='\t') 

