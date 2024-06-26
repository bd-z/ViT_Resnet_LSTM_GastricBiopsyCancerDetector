# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:25:19 2024
raw: C:/Users/zhang/rnn_tsf/rnn_nltk_simplifiedVersion_last.py
@author: zhang
"""
import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
#pip install nltk
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from random import randint

wordList = np.load('C:\\Users\\zhang\\rnn_tsf\\nlp\\wordsList.npy')
wordList = wordList.tolist()
wordList = [word.decode('UTF-8') for word in wordList]
wordVectors = np.load('C:\\Users\\zhang\\rnn_tsf\\nlp\\wordVectors.npy')
maxSequLength = 250
filename = "ids_train_p"
ids_train_p = np.load("C:\\Users\\zhang\\rnn_tsf\\aclImdb_v1\\aclImdb\\train\\" + filename +".npy")
ids_train_n = np.load("C:\\Users\\zhang\\rnn_tsf\\aclImdb_v1\\aclImdb\\train\\" + "ids_train_n" +".npy")
ids_test_n = np.load("C:\\Users\\zhang\\rnn_tsf\\aclImdb_v1\\aclImdb\\train\\" + "ids_test_n" +".npy")
ids_test_p = np.load("C:\\Users\\zhang\\rnn_tsf\\aclImdb_v1\\aclImdb\\train\\" + "ids_test_p" +".npy")

def getTrainBatch(): 
    labels = []
    arr = np.zeros([batchSize, maxSequLength])    
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 10000) 
            labels.append(1)
            arr[i] = ids_train_p[num-1 : num]
        else:
            num = randint(1, 10000)
            labels.append(0)
            arr[i] = ids_train_n[num-1 : num]
    return arr, np.array(labels)
#c, d = getTrainBatch()

def getTestBatch2(): 
    labels = []
    batchSize = 2500 
    arr = np.zeros([batchSize, maxSequLength])    
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 2500) 
            labels.append(1)
            arr[i] = ids_test_p[num-1 : num]
        else:
            num = randint(1, 2500)
            labels.append(0)
            arr[i] = ids_test_n[num-1 : num]
    return arr, np.array(labels)

# build NN
lstmUnits = 64
numClasses = 1
iterations = 20001
numDimension = wordVectors.shape[1] 

class LSTMModel(tf.keras.Model):
    def __init__(self, lstmUnits, numClasses, numDimension, wordVectors):
        super(LSTMModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=wordVectors.shape[0],                                                   
                                                   output_dim=numDimension, 
                                                   weights=[wordVectors])
       
        self.lstm = tf.keras.layers.LSTM(lstmUnits, dropout=0.25)        
        self.dense = tf.keras.layers.Dense(numClasses, activation='sigmoid') 

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        return self.dense(x)

model = LSTMModel(lstmUnits, numClasses, numDimension, wordVectors)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)#CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# choose metrics to record the accuracy of train and test
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy') 
train_recall = tf.keras.metrics.Recall(name='train_recall')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy') 
test_recall = tf.keras.metrics.Recall(name='test_recall')

# train step function
@tf.function
def train_step(texts, labels):
    with tf.GradientTape() as tape:
        predictions = model(texts, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    train_recall.update_state(labels, predictions)

# test step function
@tf.function
def test_step(test, labels):
    predictions = model(test, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    test_recall.update_state(labels, predictions)

training_losses = []
training_accuracy = []
training_recall = []

test_losses_list = []
test_accuracy_list = []
test_recall_list = []
iterations_i = []

batchSize = 100
maxSequLength=250

for i in range(iterations):
    nextBatch, nextBatchLabels = getTrainBatch()
    train_step(nextBatch, nextBatchLabels)

    if i % 200 == 0:
        # every 200 iterations test
        testBatch, testBatchLabels = getTestBatch2()
        test_step(testBatch, testBatchLabels)
        
        training_losses.append(train_loss.result().numpy())
        training_accuracy.append(train_accuracy.result().numpy())
        training_recall.append(train_recall.result().numpy())
        
        test_losses_list.append(test_loss.result().numpy())
        test_accuracy_list.append(test_accuracy.result().numpy())
        test_recall_list.append(test_recall.result().numpy())
        iterations_i.append(i)

        print(f'''Iteration {i+1}/{iterations}..
              Training Loss: {train_loss.result()}.. 
              Training Accuracy: {train_accuracy.result()}..
              Training Recall:{train_recall.result()}..
              Test Loss: {test_loss.result()}..
              Test Accuracy: {test_accuracy.result()}..
              Test Recall:{test_recall.result()}''')        
 # every 1000 iterations save model
    if i % 1000 == 0:
        model.save_weights(f"C:\\Users\\zhang\\rnn_tsf\\models\\weights\\pretrained_lstm_{i}.ckpt")
        model.save(f"C:\\Users\\zhang\\rnn_tsf\\models\\pretrained_lstm_complete_model{i}.ckpt")

# plot after train
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(test_losses_list, label='Test Loss')
plt.title('Test Loss Over Time')
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
plt.plot(test_accuracy_list, label='Test Accuracy')
plt.title('Test Accuracy Over Time')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 3)
plt.plot(test_recall_list, label='Test Recall')
plt.title('Test Recall Over Time')
plt.xlabel('Iterations')
plt.ylabel('Recall')

plt.tight_layout()
plt.show()

# train
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(training_losses, label='Train Loss')
plt.title('Train Loss Over Time')
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.subplot(1, 3, 2)
plt.plot(training_accuracy, label='Train Accuracy')
plt.title('Train Accuracy Over Time')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')

plt.subplot(1, 3, 3)
plt.plot(training_recall, label='Train Recall')
plt.title('Train Recall Over Time')
plt.xlabel('Iterations')
plt.ylabel('Recall')

plt.tight_layout()
plt.show()

# test
my_review = '''this movie is so boring, I have on interest to see it. I falled 
in sleep when I saw it. I will not see this movie again. 
I will not recommend it either'''
#import re
strip_special_chars = re.compile("[^A-Za-z0-9]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, " ", string)

cleanedLine = cleanSentences(my_review)
split = cleanedLine.split()
firstFile = np.zeros((maxSequLength), dtype='int32')
numFiles_test = 1
fileCounter = 0
ids_test = np.zeros((numFiles_test , maxSequLength), dtype='int32')
indexCounter = 0
for word in split:    
    if indexCounter < len(firstFile):
        try:
            ids_test[fileCounter][indexCounter] = wordList.index(word)
        except ValueError:
            ids_test[fileCounter][indexCounter] = 399999  # vector for unknown words
        indexCounter = indexCounter + 1
        
predictions_test = model(ids_test, training=False)
predictions_test_numpy = predictions_test.numpy()[0]

# Evaluate model
# copy from vit_gastric_model.py to calculate confidence interval
testBatch, testBatchLabels = getTestBatch2()
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
pROC = importr('pROC')
y_test_predicted = model(testBatch, training=False)
y_test_predicted2 = y_test_predicted.numpy()
y_true_r = ro.IntVector(testBatchLabels)
y_scores_r = ro.FloatVector(y_test_predicted2)

# AUC CI
roc_obj = pROC.roc(y_true_r, y_scores_r)
auc = pROC.auc(roc_obj)
ci = pROC.ci(auc, method="delong")
print("AUC:", auc[0])
print("95% CI:", tuple(ci))

# plot AUC
true_labels = testBatchLabels
predicted_scores = [0 if val < 0.5 else 1 for val in y_test_predicted2]
# convert data to R
ro.globalenv['true_labels'] = ro.FloatVector(true_labels)
ro.globalenv['predicted_scores'] = ro.FloatVector(predicted_scores)

# calculate ROC curve data
roc_curve_data = pROC.roc(ro.r('true_labels'), ro.r('predicted_scores'))

# extract ROC curve data 
roc_curve = dict(zip(roc_curve_data.names, list(roc_curve_data)))

# plot ROC
fpr = np.array(roc_curve['specificities'][::-1])
tpr = np.array(roc_curve['sensitivities'][::-1])
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# use metrics 
from sklearn import metrics
true_labels = testBatchLabels
#predicted_labels =  y_test_predicted2 #[0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
predicted_labels = [0 if val < 0.5 else 1 for val in y_test_predicted2]

# Accuracy
accuracy = metrics.accuracy_score(true_labels, predicted_labels)
print("Accuracy:", accuracy)

# Precision, Recall, F1_score
precision = metrics.precision_score(true_labels, predicted_labels)
recall = metrics.recall_score(true_labels, predicted_labels)
f1_score = metrics.f1_score(true_labels, predicted_labels)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# =============================================================================
# print("Precision:", precision)
# Precision: 0.8211382113821138
# 
# print("Recall:", recall)
# Recall: 0.8888
# 
# print("F1 Score:", f1_score)
# F1 Score: 0.8536304264310411
# =============================================================================

# AUC
fpr, tpr, thresholds = metrics.roc_curve(true_labels, predicted_labels)
auc = metrics.auc(fpr, tpr)
print("AUC:", auc)

#AUC: 0.8476
# It looks better than that in trainning because performs:
#    [0 if val < 0.5 else 1 for val in y_test_predicted2]

# Accuracy CI
p = accuracy
y_test_array = testBatchLabels
n = len(y_test_array)
z = 1.96  # for 95% CI
# standard error
se = np.sqrt(p * (1 - p) / n)
# CI
ci_lower = p - z * se
ci_upper = p + z * se
(ci_lower, ci_upper)

# confusion matrix
#import numpy as np
#from tensorflow.keras.models import Sequential  # 或者从 tensorflow import keras
from sklearn.metrics import confusion_matrix
predictions = y_test_predicted2  
# Convert the prediction results into class labels 
# (using binary classification as an example)
predicted_classes = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test_array, predicted_classes)  
print(cm)
# [[TN, FP]
#  [FN, TP]]

# recall accuracy CI
from scipy.stats import binom
def calculate_confidence_interval(n, successes, confidence_level=0.95):
    """
    Calculate the confidence interval for a binomial distribution.
    n: Number of trials
    successes: Number of successes
    confidence_level: Confidence level
    return: Confidence interval (lower limit, upper limit)
    """
    alpha = 1 - confidence_level
    lower = binom.ppf(alpha / 2, n, successes / n) / n
    upper = binom.ppf(1 - alpha / 2, n, successes / n) / n
    return lower, upper

TP = 535  
FN = 11  
FP = 286  
total_positive = TP + FN  
predicted_positive = TP + FP 

# Recall CI
recall_lower, recall_upper = calculate_confidence_interval(total_positive, TP)
print(f"Recall Confidence Interval: {recall_lower:.3f}, {recall_upper:.3f}")

# Accuracy CI
precision_lower, precision_upper = calculate_confidence_interval(predicted_positive, TP)
print(f"Precision Confidence Interval: {precision_lower:.3f}, {precision_upper:.3f}")