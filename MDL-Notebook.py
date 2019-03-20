import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

from keras.preprocessing import image
from tqdm import tqdm
from PIL import Image
from random import shuffle


import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.applications.densenet import DenseNet121
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

import numpy as np



# # Problems
# - Accuracy is good but no better than guess all one class. Think this could be solved by addressing class imbalance
# - Accuracy is only good if we take the binary crossentropy and not the full label accuracy. Will need to speak to the lecturer about how to measure performance for this type of multilabel data. -> suggested splitting into sublabels and report average accuracy of models vs each of the different single label classification tasks.
#
# # TODO
# - Need to balance the classes before passing them into the model. I.e. we need to take in more data to get a 50 50 split between having a disease and not, then run through the model. This should be possible as currently we're only processing 1% of the data. 10% is without any disease so that;s 20k. We then use another 20k with a disease.
# - Also need to add the gender and age into the x train so the model can use this information as well as the image.
# - May want to pass the data into a high res image generator or use the high res images, which would require using the GPU servers
# - To allow a more complex model to learn quickly on the gpu servers, may want to try using transfer learning from an existing model
#
# # Done
# - Need to change all the unknowns into positives as evidenced by the success of u-ones model on this paper: https://arxiv.org/pdf/1901.07031.pdf

# In[2]:


trainDf = pd.read_csv('/scratch/mdl31/CheXpert-v1.0-small/train.csv')


# In[3]:



# Remove anomalous dataline
trainDf = trainDf[trainDf.Sex != 'Unknown']
# Drop this column as it has many more classifications than lit suggests and shouldn't matter greatly for a CNN
# TODO try with and without this column
trainDf = trainDf.drop('AP/PA', 1)

def pathToID(path):
    pathList = path.split('/')
    return pathList[2][7:]

def pathToStudy(path):
    pathList = path.split('/')
    return pathList[3][5:]

# Convert all labels to a series of one-hot encoded labels.
# -1 is uncertain, 0 is negative, 1 is positive, nans are no mention of the disease in the text
trainDf = trainDf.fillna(0)
# N.B. this is replacing unknowns with true as per u-ones model here: https://arxiv.org/pdf/1901.07031.pdf
# This is essentialyl saying that if we're not sure of disease we say they have it.
# Just to be on the safeside and have better recall as we care more about recall than precision
trainDf = trainDf.replace(-1,1)


# Onehot encode the sex and the xray orientation
trainDf = trainDf.replace('Male',1)
trainDf = trainDf.replace('Female',0)
trainDf = trainDf.replace('Frontal',1)
trainDf = trainDf.replace('Lateral',0)

trainDf =trainDf.rename(index=str, columns={"Sex": "Male?",'Frontal/Lateral' :'Frontal1/Lateral0'})


#trainDf.insert(0,'Path', trainDf['Path'])
trainDf['Study'] = trainDf.Path.apply(pathToStudy)
trainDf['Patient ID'] = trainDf.Path.apply(pathToID)

# Rearrange Columns
cols = ['Patient ID', 'Study', 'Path', 'Age', 'Male?', 'Frontal1/Lateral0', 'No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']
trainDf = trainDf[cols]


def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    return files

def path_to_tensor(img_path,inputSize):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, color_mode = "grayscale", target_size=inputSize)
    # convert PIL.Image.Image type to 3D tensor with shape (x, x, 1)
    x = image.img_to_array(img)
    data = np.asarray( img, dtype="int32" )
    # convert 2D tensor to 3D tensor with shape (1, X, x) and return 3D tensor
    return data.reshape(1,inputSize[0],inputSize[1])

def paths_to_tensor(img_paths, inputSize):
    list_of_tensors = [path_to_tensor(img_path, inputSize) for img_path in img_paths]
    return np.array(list_of_tensors)


def path_to_tensor_channel_last_3colour(img_path,inputSize):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, color_mode = "grayscale", target_size=inputSize)
    # convert PIL.Image.Image type to 3D tensor with shape (x, x, 1)
    x = image.img_to_array(img)
    data = np.asarray( img, dtype="int32" )
    # convert 2D tensor to 3D tensor with shape (X, x, 3) and return 3D tensor
    return np.stack((data,)*3, axis=-1)


def paths_to_tensor_channel_last_3colour(img_paths, inputSize):
    list_of_tensors = [path_to_tensor_channel_last_3colour(img_path, inputSize) for img_path in img_paths]
    return np.array(list_of_tensors)


# In[5]:


inputSize = (224,224)

sample_size = 6000
targetColumn = [6]
colName = trainDf.columns.tolist()[targetColumn[0]]
print(f"This model will be targetting {colName} column")

# Create balanced dataset with 50% pos examples and 50% neg examples, only take frontal scans for now
pos = trainDf[(trainDf[colName] == 1) & (trainDf['Frontal1/Lateral0'] == 1 )]
neg = trainDf[(trainDf[colName] == 0) & (trainDf['Frontal1/Lateral0'] == 1 )]

# Deleting training dataframe in order to save memory and avoid OOM errors.
#del trainDf

posSample = pos.sample(int(sample_size/2))
negSample = neg.sample(int(sample_size/2))
sample = pd.concat([posSample,negSample])


x_train_paths, x_val_paths, y_train, y_val = train_test_split(sample.Path, sample[colName], stratify=sample[colName], random_state =2)


# The 3 channel option is required for the denseNet and other transfer learning models
# Single channel can be used on our models

#x_train = paths_to_tensor(x_train_paths,inputSize)#.astype('float32')/255
x_train3Channel = paths_to_tensor_channel_last_3colour(x_train_paths,inputSize)#.astype('float32')/255

#y_train = trainDf.iloc[:training_no,targetColumn] # to do all labels: trainDf.iloc[:training_no,8:]
#x_val = paths_to_tensor(x_val_paths,inputSize)#.astype('float32')/255
x_val3Channel = paths_to_tensor_channel_last_3colour(x_val_paths,inputSize)#.astype('float32')/255

#y_val = trainDf.iloc[training_no:training_no+val_no,targetColumn]

'''
model = Sequential()

model.add(Conv2D(64, (3,3), strides=(1,1), input_shape=(1,inputSize[0],inputSize[1])))
model.add(Conv2D(32, (3,3)))
model.add(Conv2D(16, (3,3)))

model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(32,activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(16,activation='relu'))
#model.add(Dropout(0.2))



model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
weightsFilePath="weights.best.hdf5"


# In[12]:


checkpoint = ModelCheckpoint(weightsFilePath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(x_train,y_train, epochs = 10, batch_size=32,  validation_data=(x_val, y_val), callbacks=[checkpoint])
model.load_weights(weightsFilePath)


# In[ ]:


# Plot the history of this model
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracies')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.show()


# In[ ]:


predictions = model.predict(x_val)


# In[ ]:


def checkAcc(predictions, truths):
    wrongs = 0
    for i,prediction in enumerate(predictions):
        truth = truths[i]
        for j, val in enumerate(prediction):
            if val >= 0.5 and truth[j] == 0:
                wrongs += 1
                # break
            if val < 0.5 and truth[j] == 1:
                wrongs += 1
                # break
    total = 41*len(predictions) # len(predictions)
    return (total - wrongs) / total, wrongs, total


checkAcc(predictions, y_val.values)


# In[9]:


y_val.values.shape


# In[7]:
'''

# Trnsfer learning model
model2 = Sequential()

model2.add(DenseNet121(input_shape=(224,224,3)))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(32,activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(16,activation='relu'))

# Turn off the trainability of the transfer model to save computational resource/memory
for layer in model2.layers:
    layer.trainable=False

model2.add(Dropout(0.3))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()
weightsFilePath2="weights2.best.hdf5"


# In[8]:


checkpoint2 = ModelCheckpoint(weightsFilePath2, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history2 = model2.fit(x_train3Channel,y_train, epochs = 10, batch_size=32,  validation_data=(x_val3Channel, y_val), callbacks=[checkpoint2])
model2.load_weights(weightsFilePath2)
