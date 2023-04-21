import csv
import numpy as np
from collections import Counter
filename = './test.csv'
with open(filename) as f:
    reader = csv.reader(f)
    data = list(reader)
new_data = []
f.close()
# f=open("train_s1.txt","w")
for i in data:
    # if i[1] == 'claim':
    #     line = ['label','text']
    #     f.writelines(' '.join(line)+'\n')
    if i[1]!='0_0' and i[1] != 'claim':
        label = int(i[1][0])-1
        # print(label)
        line = [str(label),i[0]]
        new_data.append(line)
        # f.writelines(' '.join(line)+'\n')
new_data = np.array(new_data)
print(len(new_data))

