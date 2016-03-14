from statistics import mode 
data=open('ensemble.csv','r').read().split('\n')
while '' in data: 
    data.remove('') 

outf=open('predictions.csv','w') 
for line in data: 
    tokens=line.split('\t') 
    ids=[tokens[1],tokens[3],tokens[5]] 
    try: 
        vote=mode(ids) 
    except: 
        vote=ids[1] 
    outf.write(tokens[0]+'\t'+str(vote)+'\n')

