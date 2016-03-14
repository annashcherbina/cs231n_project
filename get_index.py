from Params import * 
import sys 
from os import listdir
from os.path import isfile, join

label_dict=dict()
labels=open(labels,'r').read().split('\n')
while '' in labels:
    labels.remove('')
for i in range(len(labels)):
    label_dict[labels[i]]=i
print "built dictionary of labels (id --> number) "
file_names=[]
labels=[] 
for label in label_dict:
    #print str(label) 
    cur_dir=training_dir+label+"/images" 
    onlyfiles = [f for f in listdir(cur_dir) if isfile(join(cur_dir, f))]
    onlyfiles=[cur_dir+'/'+f for f in onlyfiles]
    file_names=file_names+onlyfiles
    #print str(len(file_names)) 
    #cur_labels=nsamples*[label_dict[label]]
    #labels=labels+cur_labels
    #print str(len(labels))
#print str(file_names) 
#our_index=file_names.index(cur_dir+'/'+sys.argv[1]) 
#print str(our_index) 
outf=open('train_data_names.txt','w') 
outf.write('\n'.join(file_names))
