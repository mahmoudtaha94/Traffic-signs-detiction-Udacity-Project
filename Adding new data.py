'''


Making a new dataset from the German traffic signs and Belgium traffic signs

Belgium dataset is at:
    http://www.vision.ee.ethz.ch/~timofter/traffic_signs/
    the files labeled: BelgiumTS for Classification

'''

from PIL import Image
import os
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split
path = './Training/'
folders=os.listdir(path)
counter=0  #counts how many image data
'''
working on the Belgium dataset


resize the images to 32x32x3 jpg
'''
for i in range(len(folders)):
    dirs = os.listdir( path+folders[i]+'/'+'old' )
    for item in dirs:  #reads the images from "old" folder and save to "resized" folder
        im = Image.open(path+folders[i]+'/'+'old'+'/'+item)
        f, e = os.path.splitext(path+folders[i]+'/'+item)
        imResize = im.resize((32,32), Image.ANTIALIAS)
        imResize.save(path+folders[i]+'/'+'resized'+'/'+item.replace('.ppm','')+ '.jpg')
        counter+=1

'''
make arrays of the new dataset

the folders must be sorted to make the labels array
'''
folders=['00000','00001','00002','00003','00004','00005','00006','00007','00008','00009',
         '00010','00011','00012','00013','00014','00015','00016','00017','00018','00019',
         '00020','00021','00022','00023','00024','00025','00026','00027','00028','00029',
         '00030','00031','00032','00033','00034','00035','00036','00037','00038','00039',
         '00040','00041','00042','00043','00044','00045','00046','00047','00048','00049',
         '00050','00051','00052','00053','00054','00055','00056','00057','00058','00059',
         '00060','00061']
first_item_in_dataset_flag=True #to initialize the arrays in the first photo
data_set_labels=np.zeros(counter)
k=0
label=43
for i in range(len(folders)):
    dirs = os.listdir( path+folders[i]+'/'+'resized' )
    for item in dirs:
        if first_item_in_dataset_flag:
            data_set=mpimg.imread(path+folders[i]+'/'+'resized'+'/'+item)
            data_set=data_set.reshape(1,32,32,3)
            data_set_labels[k]=label
            first_item_in_dataset_flag=False
            k+=1
        else:
            im = mpimg.imread(path+folders[i]+'/'+'resized'+'/'+item)
            im=im.reshape(1,32,32,3)
            data_set=np.concatenate((data_set,im),axis=0)
            data_set_labels[k]=label
            k+=1
    label+=1
X_train_n,X_temp,y_train_n,y_temp=train_test_split(data_set,data_set_labels,test_size=0.2,random_state=0)
X_valid_n,X_test_n,y_valid_n,y_test_n=train_test_split(X_temp,y_temp,test_size=0.5,random_state=0)
print(X_train_n.shape)
print(y_train_n.shape)
print(X_valid_n.shape)
print(y_valid_n.shape)
print(X_test_n.shape)
print(y_test_n.shape)
'''
Load the German dataset

'''
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file='./traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train_o, y_train_o = train['features'], train['labels']
X_valid_o, y_valid_o = valid['features'], valid['labels']
X_test_o, y_test_o = test['features'], test['labels']
print(X_train_o.shape)
print(y_train_o.shape)
print(X_valid_o.shape)
print(y_valid_o.shape)
print(X_test_o.shape)
print(y_test_o.shape)

'''
Add the Belgium set to the German set

'''
X_train=np.concatenate((X_train_n,X_train_o),axis=0)
X_valid=np.concatenate((X_valid_n,X_valid_o),axis=0)
X_test=np.concatenate((X_test_n,X_test_o),axis=0)
y_train=np.concatenate((y_train_n,y_train_o),axis=0)
y_valid=np.concatenate((y_valid_n,y_valid_o),axis=0)
y_test=np.concatenate((y_test_n,y_test_o),axis=0)
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
print(X_test.shape)
print(y_test.shape)

'''
save the new arrays to be imported in the model 

'''
with open('X_train','wb') as f:
    pickle.dump(X_train,f)
with open('y_train','wb') as f:
    pickle.dump(y_train,f)
with open('X_valid','wb') as f:
    pickle.dump(X_valid,f)
with open('y_valid','wb') as f:
    pickle.dump(y_valid,f)
with open('X_test','wb') as f:
    pickle.dump(X_test,f)
with open('y_test','wb') as f:
    pickle.dump(y_test,f)
print('finished')




