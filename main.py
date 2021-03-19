
# ## Problem Statement:
# 
# 
# Build a Resnet50 CNN classifier to pridict scores of these breeds only: 
# beagle, chihuahua, doberman, french_bulldog, golden_retriever, malamute, pug, saint_bernard, scottish_deerhound, tibetan_mastiff.
#  
# Importing libraries :

# %%
import pandas as pd 
import numpy as np
from numpy import save
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from random import randrange
import kaggle
from prettytable import PrettyTable

## KERAS
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import SGD
from keras import applications
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.activations import relu, softmax

#  
# # 2. Reading Data

# %%
# Reading csv file with pandas
df = pd.read_csv('labels.csv')
df.shape

#  
# The dataset has total of 10222 rows
# 

# %%
# Creating custom dog breed list
dogs = ['beagle', 'chihuahua', 'doberman', 'french_bulldog', 'golden_retriever', 'malamute', 'pug', 'saint_bernard', 'scottish_deerhound', 'tibetan_mastiff']
images_dir = 'train/'
labels_dir = 'labels.csv'


# %%
custom_breed = df.loc[df.breed.isin(dogs)].reset_index(drop=True)
custom_breed.shape

#  
# After filtering custom breeds we get a dataset of approx 850 rows.
# 

# %%
custom_breed.head()


# %%
custom_breed.breed.nunique()


# %%
custom_breed['breed'].value_counts().sum

#  
# The data has high number of images of scottish deerhound with 126 images, whereas golden retriver has lowest number of images with 64 only.
# 
#  
# 3. Displaying a random image of dog from the data. 
# 

# %%
random_dog = randrange(custom_breed.shape[0])
print(f'Random index location {random_dog}')


# %%
print('Dog Breed:',custom_breed.loc[random_dog,'breed'])
Image.open(images_dir + custom_breed.loc[random_dog, "id"] + ".jpg")

#  
# # 4. Data preprocessing
# 

# %%
# Reading images and converting each image into array and apending them to a list
image_list = []
label_list = []

for index, x in tqdm(custom_breed.iterrows()):
    image = cv2.imread(images_dir + x['id'] + '.jpg')
    image = cv2.resize(image, (300, 300))
    image_list.append(image)
    label_list.append(x['breed'])


# %%
print(f'No of Images: {len(image_list)} \nNo of Labels: {len(label_list)}')


# %%
# Convert list to array
data = np.array(image_list) #/ 255
data.shape


# %%
# Saving the data and labes into npy file on the system [Optional]
save('dog_images.npy', data)
save('labels.npy', label_list)


# %%
# Reading the files:
from numpy import load
photos = load('dog_images.npy')
labels = load('labels.npy')
print(photos.shape, labels.shape)


# %%
# Converting labels to numbers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
label_encoder = LabelEncoder()

vec = label_encoder.fit_transform(labels)

target = to_categorical(vec)
target

#  
# 

# %%
x_train, x_test, y_train, y_test = train_test_split(photos,target, test_size=0.3)


# %%
# Checking the size of train and test data
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# %%
num_class = y_test.shape[1]

#  
# # 5.1 Building the Model

# %%
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import MaxPooling2D


# %%
# Importing ResNet50 from Keras
from tensorflow.keras.applications.resnet50 import ResNet50
# load model
model = Sequential()

# Adding Resnet50 with pretrained imagenet weights
img_height,img_width = 300,300
model.add(ResNet50(weights="imagenet", pooling=max, include_top=False, input_shape= (img_height,img_width,3)))

# 2nd layer as Dense for classification
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

# Not training firstlayer 
model.layers[0].trainable = False


# %%
# Defining sgd optimizer


rate = 0.01
sgd = SGD(lr=rate)


# %%
model.compile(optimizer= sgd,  loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


# %%
# Traing the network for 10 epochs
model.fit(x_train, y_train, epochs = 10,batch_size=32, validation_split=0.3)


# %%
preds = model.evaluate(x_test, y_test)
print ("Loss = " , preds[0])
print ("Test Accuracy = ", preds[1])


# %%
y_pred_prob = model.predict(x_test)


# %%
y_pred = y_pred_prob.argmax(axis=-1)
y_pred = to_categorical(y_pred)

#  
# # 5.2 Calculating Performace

# %%
# Calculating Performace metrics 
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,  average='micro')
recall = recall_score(y_test, y_pred,  average='micro')
f1 = f1_score(y_test, y_pred,  average='micro')
auc = roc_auc_score(y_test, y_pred,  average='micro')


# %%
from prettytable import PrettyTable


# %%
x = PrettyTable()
x.field_names = ["Metrics", "Score"]
x.add_row(["Accuracy", round(accuracy,3)])
x.add_row(["Precision", round(precision,3)])
x.add_row(["Recall", round(recall,3)])
x.add_row(["F1 Score", round(f1,3)])
x.add_row(["ROC-AUC Score", round(auc,3)])
print(x)


# %%
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
confusion_df = pd.DataFrame(matrix, index = dogs, columns = dogs)
plt.figure(figsize = (10,8))
sn.heatmap(confusion_df, annot=True)


# %%
# Saving our model for future use:

model.save('dog_breed_model.h5')  

#  
# **Conclusion**: Our model performs well on the given data with overall accuracy of 0.95
# 
#  
# 6 Testing on custom image data
# 
# 

# %%
def breed_predictor(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (300, 300))
    # dog_img = Image.open(image_file)
    image = image.reshape(1,300,300,3)
    result_prob = model.predict(image)
    result = result_prob.argmax(axis=-1)
    result = label_encoder.inverse_transform(result)
    return f'Predicted Breed: {result[0]} with probability of {round(np.amax(result_prob)*100,2)}%'


# %%
file_n = 'gold_ret.jpg'
breed_predictor(file_n)


# %%
Image.open(file_n)


# %%
file_n = 'dobber.jpg'
breed_predictor(file_n)


# %%
Image.open(file_n)


# %%
file_n = 'saint_berni.jpg'
breed_predictor(file_n)


# %%
Image.open(file_n)


