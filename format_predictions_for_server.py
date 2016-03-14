#formats predictions in a way that can be uploaded to evaluation server 
from os import listdir
from os.path import isfile, join
import sys
from statistics import mode 
from Params import * 

#predictions=open('ensemble.csv','r').read().split('\n') 
predictions=open(sys.argv[1],'r').read().split('\n') 
while '' in predictions: 
    predictions.remove('') 

wnids=open(labels,'r').read().split('\n') 
while '' in wnids: 
    wnids.remove('') 



#training data 
file_names=[]
#labels=[] 

label_dict=dict()
labels=open(labels,'r').read().split('\n')
while '' in labels:
    labels.remove('')
for i in range(len(labels)):
    label_dict[labels[i]]=i

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
onlyfiles=file_names 

#cur_dir=test_dir+"images/"
#onlyfiles = [f for f in listdir(cur_dir) if isfile(join(cur_dir, f))]

entries=100000
#outf=open('ensemble_formatted.tsv','w') 
outf=open(sys.argv[2],'w') 
#outf.write('Image\tPretrained\tPretrainedFreezeAndStack\tVGG_Like\tRegularizationAndDropout\tEnsemble\n')
for i in range(entries): 
    image_name=onlyfiles[i] 
    predict_indices=predictions[i].split('\t') 
    predict_indices=[int(i) for i in predict_indices] 
    wnid1=wnids[predict_indices[0]]
    #wnid2=wnids[predict_indices[1]] 
    #wnid3=wnids[predict_indices[2]] 
    #wnid4=wnids[predict_indices[3]] 
    #try: 
    #    vote=mode([wnid1,wnid2,wnid3,wnid4])
    #except: 
    #    vote=wnid4
    #if len(set(predict_indices))==1: 
    #    #all 3 agree!!! 
    #    print image_name+ '\t'+str(vote) +'\t' + str(predict_indices)
    outf.write(image_name+'\t'+str(wnid1)+'\n')#+'\t'+str(wnid2)+'\t'+str(wnid3)+'\t'+str(wnid4)+'\t'+str(vote)+'\n')

