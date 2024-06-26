# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 20:33:21 2021

@author: zhang
"""
import os
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, BatchNormalization, MaxPool2D ,Activation, MaxPooling2D


def data_table(folder):
    '''create a dataframe which has 'id' and 'label' columns. The id column is the path of each image
    and the label column contain 1 and 0 which indicate cancer cells exist or not 
    '''    
    p=os.walk(folder)
    list_empty=[]
    dict_empty={}
    for path, dir_list,file_list in p:
        for file_name in file_list:
            file_path=os.path.join(path,file_name)
            list_empty.append(file_path)            
    for file_path in list_empty:
        if 'non_can' in file_path:
            label=0
        elif 'cancer' in file_path:
            label=1
        dict_empty['{}'.format(file_path)]=label
    df = pd.DataFrame.from_dict(dict_empty, orient='index',columns=['label'])
    df = df.reset_index().rename(columns={'index':'id'})   
    df = shuffle(df)    
    return df

#folder where the images data stored
#f=r'G:\BaiduNetdiskDownload\train'
#G:\BaiduNetdiskDownload\train2\training
f=r'G:\BaiduNetdiskDownload\train2\training\less_data'

df_full=data_table(f)   

#define X and y
X_train=df_full['id']
y_train=df_full['label']

fv=r'G:\BaiduNetdiskDownload\train2\validation'
df_v=data_table(fv)   
#define X and y
X_test=df_v['id']
y_test=df_v['label']

# train and test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100) # split into test and train sets

def slice_load(file_list):
    ''' load the images'''    
    images=[]    
    for filename in file_list:        
        im = image.load_img(filename,target_size=(224, 224, 3)) 
        b = image.img_to_array(im)
        images.append(b)
    return images

X_train_image=slice_load(X_train)
X_train_array=np.array(X_train_image)/255

X_test_image=slice_load(X_test)
X_test_array=np.array(X_test_image)/255

np.save(r'G:\BaiduNetdiskDownload\train2\training\less_data\X_train_array.npy', X_train_array)
np.save(r'G:\BaiduNetdiskDownload\train2\training\less_data\y_train.npy', y_train)
np.save(r'G:\BaiduNetdiskDownload\train2\training\less_data\X_test_array.npy', X_test_array)
np.save(r'G:\BaiduNetdiskDownload\train2\training\less_data\y_test.npy', y_test)

#2024-2-23
X_train_array = np.load(r'G:\BaiduNetdiskDownload\train2\training\less_data\X_train_array.npy')/255
y_train = np.load(r'G:\BaiduNetdiskDownload\train2\training\less_data\y_train.npy')
X_test_array = np.load(r'G:\BaiduNetdiskDownload\train2\training\less_data\X_test_array.npy')/255
y_test = np.load(r'G:\BaiduNetdiskDownload\train2\training\less_data\y_test.npy')

X_train_array.shape
type(y_train)

#clear sessions
K.clear_session()
input_shape = (224, 224, 3)
# transfer learning with ResNet50V2
resMod = ResNet50V2(include_top=False, weights='imagenet',
				  input_shape=input_shape)

#frozen the layers in ResNet50V2
for layer in resMod.layers:
    layer.trainable = False

# build model
model = Sequential()
model.add(resMod)
model.add(tf.keras.layers.GlobalAveragePooling2D()) 
#1st Dense: (None, 60) 
model.add(keras.layers.Dense(60, activation='relu'))  
#regularization with penalty term
model.add(Dropout(0.2))
# 2nd Dense: (None, 50)
model.add(keras.layers.Dense(50, activation='relu'))
#regularization
model.add(keras.layers.BatchNormalization())  
# 2nd Dense: (None, 50)
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.BatchNormalization()) 
# Output Layer: (None, 1)  
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile 
model.compile(loss= 'binary_crossentropy',
              optimizer='adam',
              metrics=[
                  keras.metrics.BinaryAccuracy(name="accuracy"),
                  keras.metrics.Precision(name="precision"),
                  keras.metrics.Recall(name="recall"),
                  keras.metrics.AUC(name="auc") 
              ]            
 
 )

from keras.callbacks import EarlyStopping
# EarlyStopping 
callback = EarlyStopping(monitor='val_accuracy', patience=5) #val_loss
# add EarlyStopping
results = model.fit(X_train_array, y_train, batch_size=164, epochs=1000, verbose=1,
                    validation_split=0.2, callbacks=[callback], shuffle=True)  

model.evaluate(X_test_array, y_test)

results.history['val_accuracy']
#save model
#path = r'C:\Users\zhang\rnn_tsf\models\cnn\resnet_gastric_13_02_2014.keras'
path = r'C:\Users\zhang\rnn_tsf\models\cnn\resnet_gastric_23_02_2014.keras'
model.save(path)
# doN't save as H5model
#path_h5 = r'C:\Users\zhang\rnn_tsf\models\cnn\resnet_gastric_13_02_2014.h5'

# =============================================================================
# 
# #pip install ann_visualizer
# #pip install graphviz
# from ann_visualizer.visualize import ann_viz
# ann_viz(model, title="GTBR_resnet")
# 
# #pip install keras.utils
# #pip install pydot
# from keras.utils import plot_model
# import pydot
# import graphviz
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# 
# =============================================================================

#copy from vit_gastric_model.py

#=============================================================================

# Model evaluation
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
pROC = importr('pROC')
y_test_predicted = model(X_test_array)
y_test_predicted2 = y_test_predicted.numpy()

y_test_array = y_test
y_true_r = ro.IntVector(y_test_array)
y_scores_r = ro.FloatVector(y_test_predicted2)

# AUC and CI
roc_obj = pROC.roc(y_true_r, y_scores_r)
auc = pROC.auc(roc_obj)
ci = pROC.ci(auc, method="delong")
print("AUC:", auc[0])
print("95% CI:", tuple(ci))

# draw ROC
true_labels = y_test_array
predicted_scores = y_test_predicted2

ro.globalenv['true_labels'] = ro.FloatVector(true_labels)
ro.globalenv['predicted_scores'] = ro.FloatVector(predicted_scores)

# calculate ROC curve data
roc_curve_data = pROC.roc(ro.r('true_labels'), ro.r('predicted_scores'))

# extract ROC curve data
roc_curve = dict(zip(roc_curve_data.names, list(roc_curve_data)))

# True Positive Rate and False Positive Rate
fpr = np.array(roc_curve['specificities'][::-1])
tpr = np.array(roc_curve['sensitivities'][::-1])

# draw ROC
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

_, accuracy, precision, recall, auc = model.evaluate(X_test_array, y_test_array)
#history.history["accuracy"]

# accuracy CI
p = accuracy
n = len(y_test_array)
z = 1.96  # z value for 95%CI
se = np.sqrt(p * (1 - p) / n)

# CI
ci_lower = p - z * se
ci_upper = p + z * se

(ci_lower, ci_upper)


# confusion matrix
#import numpy as np
#from tensorflow.keras.models import Sequential  # 或者从 tensorflow import keras
from sklearn.metrics import confusion_matrix

predictions = y_test_predicted2  # x_test是测试特征数据

# 将预测结果转换为类别标签（这里以二分类为例）
predicted_classes = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test_array, predicted_classes)  

print(cm)
#[[165 286],
#[ 11 535]]
#：
# [[TN, FP]
#  [FN, TP]]
# recall accuracy rate and CI
from scipy.stats import binom

def calculate_confidence_interval(n, successes, confidence_level=0.95):
    """
    calculate the Binomial distribution CI
    :param n: number of experiment
    :param successes: number of sucess
    :param confidence_level: CI
    :return: CI（lower，upper）
    """
    alpha = 1 - confidence_level
    lower = binom.ppf(alpha / 2, n, successes / n) / n
    upper = binom.ppf(1 - alpha / 2, n, successes / n) / n
    return lower, upper

#TN
TP = 535  
FN = 11  
FP = 286  
total_positive = TP + FN  
predicted_positive = TP + FP  
recall = TP/(total_positive)
precision = TP/(TP+FP)
# Recall CI
recall_lower, recall_upper = calculate_confidence_interval(total_positive, TP)
print(f"Recall Confidence Interval: {recall_lower:.3f}, {recall_upper:.3f}")

# Accuracy CI
precision_lower, precision_upper = calculate_confidence_interval(predicted_positive, TP)
print(f"Precision Confidence Interval: {precision_lower:.3f}, {precision_upper:.3f}")

#2024-2-27 copy from gastric_cancer_pred_vit.py
#path = r'C:\Users\zhang\rnn_tsf\models\cnn\resnet_gastric_13_02_2014.keras'
path = r'C:\Users\zhang\rnn_tsf\models\cnn\resnet_gastric_23_02_2014.keras'
res_gas_model_keras = load_model(path)
y_test_predicted = res_gas_model_keras(X_test_array)
y_test_predicted2 = y_test_predicted.numpy()

# =============================================================================
# #2024-2-27 
# print(cm)
# [[327 124]
#  [ 24 522]] 
# TP = 522  
# FN = 24  
# FP = 124  
# =============================================================================
# use metrics to evalute model performance
from sklearn import metrics
true_labels = y_test_predicted2
predicted_labels =  y_test_predicted2 #[0, 1, 1, 0, 1, 0, 1, 1, 0, 1]

# accuracy
accuracy = metrics.accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# accuracy, recall, F1 score
precision = metrics.precision_score(true_labels, predicted_labels)
recall = metrics.recall_score(true_labels, predicted_labels)
f1_score = metrics.f1_score(true_labels, predicted_labels)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# AUC
fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted_labels)
auc = metrics.auc(fpr, tpr)
print("AUC:", auc)