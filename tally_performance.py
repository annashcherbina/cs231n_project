data=open('train.check1','r').read().split('\n') 
while '' in data: 
    data.remove('') 
matches=0 
mismatches=0 
for line in data: 
    tokens=line.split('\t') 
    if tokens[0]==tokens[1]: 
        matches+=1 
    else: 
        mismatches+=1

total=matches+mismatches 
accuracy=float(matches)/total 
print "accuracy:"+str(accuracy) 
