# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:47:35 2024 
https://www.bilibili.com/video/BV18Q4y1o7NY/?spm_id_from=333.788&vd_source=86578d070dfbeb3051fcfcb3dd7e668c
refenence: https://keras.io/examples/vision/image_classification_with_vision_transformer/#implement-multilayer-perceptron-mlp
raw:C:/Users/zhang/rnn_tsf/vit_gastric_model.py
@author: zhang
"""
import os
os.environ["KERAS_BACKEND"] = "jax"  
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# load data
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV , cross_val_score
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import image
from PIL import Image

#2024-2-25 load data
X_train_array = np.load(r'G:\BaiduNetdiskDownload\train2\training\less_data\X_train_array.npy')/255
y_train_array = np.load(r'G:\BaiduNetdiskDownload\train2\training\less_data\y_train.npy')
X_test_array = np.load(r'G:\BaiduNetdiskDownload\train2\training\less_data\X_test_array.npy')/255
y_test_array = np.load(r'G:\BaiduNetdiskDownload\train2\training\less_data\y_test.npy')        
# Prepare the data
num_classes = 1#100
input_shape = (224, 224, 3) 
# Configure the hyperparameters
learning_rate = 0.0001
weight_decay = 0.0001
batch_size = 128 
num_epochs = 100  
image_size = 224 
patch_size = 32  
num_patches = (image_size // patch_size) ** 2
projection_dim = 256 
num_heads = 4
neuron_units = [  
     projection_dim * 2, 
     projection_dim,
] 
transformer_blocks = 8 
mlp_neuron_units = [2048, 1024]
data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
data_augmentation.layers[0].adapt(X_train_array)

data_augmentation_path = 'C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\train\\fifth_train\\data_augmentation.keras'
data_augmentation.save(data_augmentation_path)

# Implement multilayer perceptron (MLP)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Implement patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape0 = tf.shape(images)
        batch_size = input_shape0[0]        
        height = input_shape0[1]
        width = input_shape0[2]
        channels = input_shape0[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        
        patches = tf.image.extract_patches(images=images,
                                   sizes=[1, patch_size, patch_size, 1],
                                   strides=[1, patch_size, patch_size, 1],
                                   rates=[1, 1, 1, 1],
                                   padding='VALID')       
        
        patches = tf.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )     
                
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

# display patches for a sample image
plt.figure(figsize=(4, 4))
image = X_train_array[np.random.choice(range(X_train_array.shape[0]))]
images = X_train_array[np.random.choice(range(X_train_array.shape[0]), size=2, replace=False)]
plt.imshow(image.astype("uint8"))
plt.axis("off")
resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")
patches.shape

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

#Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)       
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.expand_dims(
            tf.range(0, self.num_patches, 1), axis=0
            )
             
        
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    
# Build the ViT model
def create_vit_classifier():
    inputs = keras.Input(shape=input_shape)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputs) #augmented
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
        #2024-2-25
        # Create multiple layers of the Transformer block.
    for _ in range(transformer_blocks):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1 (removed).            
            # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(attention_output)
            # MLP.
        neuron_units = [  
                 projection_dim * 2,
                 projection_dim 
            ]
        x3 = mlp(x3, hidden_units=neuron_units, dropout_rate=0.1) # transformer_units            
            # Skip connection 2.
        encoded_patches = layers.Add()([x3, x1])
        
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.     
    features = mlp(representation, hidden_units=mlp_neuron_units, dropout_rate=0.5)
    # Classify outputs
    # Create the Keras model.
    # for binary classification the last layer is updated as following
    result = layers.Dense(1, activation='sigmoid')(features)    
    model = keras.Model(inputs=inputs, outputs= result)
    return model

# Compile, train, and evaluate the mode
def run_experiment(model, X_test_array, y_test_array):
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc") 
        ],
    )

    checkpoint_filepath = 'C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\train\\six\\checkpoint.weights_6_2.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Stop training when validation loss is no longer decreasing
        patience= 8,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    history = model.fit(
        x=X_train_array,
        y=y_train_array,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_test_array, y_test_array),
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    return history

vit_classifier = create_vit_classifier()

# train
history = run_experiment(vit_classifier, X_test_array, y_test_array)

'''
history = run_experiment(vit_classifier)
19/19 [==============================] - 51s 2s/step - loss: 0.4092 - accuracy: 0.8481 - precision: 0.8568 - recall: 0.8834 - val_loss: 0.2838 - val_accuracy: 0.8740 - val_precision: 0.8588 - val_recall: 0.9500
28/28 [==============================] - 4s 155ms/step - loss: 0.3098 - accuracy: 0.8684 - precision: 0.8525 - recall: 0.9541
Test accuracy: 86.84%
Test precision: 85.25%
Test recall: 95.41%
'''
all_items = history.history.keys()
for item in all_items:
    print(item)

# save history
import pickle
history_pickle = 'C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\train\\fifth_train\\history.pkl'
with open(history_pickle, 'wb') as file:
    pickle.dump(history.history, file)

def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

plot_history("loss")
plot_history("accuracy")
plot_history("precision")
plot_history("recall")
plot_history("auc")

# read history
with open(history_pickle, 'rb') as file:
    loaded_history = pickle.load(file)

print(loaded_history.keys())  

def plot_loaded_history(item):
    plt.plot(loaded_history[item], label=item)
    plt.plot(loaded_history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

plot_loaded_history("loss") 
plot_loaded_history("accuracy")
plot_loaded_history("precision")
plot_loaded_history("recall")
plot_loaded_history("auc")

# SavedModel
fifth_vim_stomach = 'C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\train\\fifth_train\\fifth_vim_stomach_trained_small_data_model.keras'
# save as keras， HDF5 is out of date
#vit_classifier.save(fifth_vim_stomach)
# read keras
fifth_vim_stomach_loaded = tf.keras.models.load_model(fifth_vim_stomach)

# another test
tif_test_file = "C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\test\\TCGA-KB-A93J-01A-01-TS1 stomach cancer.tif"
os.path.exists(tif_test_file)
from tensorflow.keras.preprocessing import image
tif_test = image.load_img(tif_test_file,target_size=(224, 224, 3))
tif_test_array = np.array(tif_test)/255
patch_tif = tif_test_array[np.newaxis, :, :, :]
tensor_value = vit_classifier(patch_tif)
tensor_value = tensor_value.numpy()
print(tensor_value) 
specific_value = tensor_value[0, 0]
print(specific_value)
#1.0 : success

# Model Evaluation 1
# copy from gastric_cancer_pred_vit.py line 244 on 2024-2-26
loaded_vit_modle = create_vit_classifier()
checkpoint_filepath = 'C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\train\\fifth_train\\checkpoint.weights_5.h5'
loaded_vit_modle.load_weights(checkpoint_filepath)   
y_test_predicted = loaded_vit_modle(X_test_array)
y_test_predicted2 = y_test_predicted.numpy().reshape(-1)
optimizer = keras.optimizers.Adam(
    learning_rate=learning_rate, weight_decay=weight_decay
)
loaded_vit_modle.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc") 
    ],
)
_, accuracy, precision, recall, auc = loaded_vit_modle.evaluate(X_test_array, y_test_array)
# Model Evaluation 2
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
predicted_classes = (y_test_predicted2 > 0.5).astype(int)
accuracy1 = accuracy_score(y_test_array, predicted_classes)
precision1 = precision_score(y_test_array, predicted_classes)
recall1 = recall_score(y_test_array, predicted_classes)
auc1 = roc_auc_score(y_test_array, predicted_classes)
# Model Evaluation 3
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
pROC = importr('pROC')
y_test_predicted = vit_classifier(X_test_array)
y_test_predicted2 = y_test_predicted.numpy().reshape(-1)
y_true_r = ro.IntVector(y_test_array)
y_scores_r = ro.FloatVector(y_test_predicted2)
# AUC and CI
roc_obj = pROC.roc(y_true_r, y_scores_r)
auc = pROC.auc(roc_obj)
ci = pROC.ci(auc, method="delong")
# print
print("AUC:", auc[0])
print("95% CI:", tuple(ci))
_, accuracy, precision, recall, auc = vit_classifier.evaluate(X_test_array, y_test_array)
#accuracy and CI
p = accuracy
n = len(y_test_array)
z = 1.96  # for 95%CI
se = np.sqrt(p * (1 - p) / n)
# CI
ci_lower = p - z * se
ci_upper = p + z * se
print(f"Accuracy:{accuracy}")
print(f"Accuracy Confidence Interval:{ci_lower, ci_upper}")
# confusion matrix
#import numpy as np
#from tensorflow.keras.models import Sequential 
from sklearn.metrics import confusion_matrix
predictions = y_test_predicted2  
# convert prediction as classes (Binormal)
predicted_classes = (predictions > 0.5).astype(int)
cm = confusion_matrix(y_test_array, predicted_classes)  
print(cm)
#[[165 286]
# [ 11 535]]
# [[TN, FP]
#  [FN, TP]]
# recall and CI
from scipy.stats import binom
def calculate_confidence_interval(n, successes, confidence_level=0.95):
    """
    binormal CI    
    :param n: number of experiment
    :param successes: number of success
    :param confidence_level: CI
    :return: CI（lower，upper）
    """
    alpha = 1 - confidence_level
    lower = binom.ppf(alpha / 2, n, successes / n) / n
    upper = binom.ppf(1 - alpha / 2, n, successes / n) / n
    return lower, upper

TP = 537
FN = 9
FP = 287
total_positive = TP + FN 
predicted_positive = TP + FP 

# recall CI
recall_lower, recall_upper = calculate_confidence_interval(total_positive, TP)
print(f"Recall Confidence Interval: {recall_lower:.3f}, {recall_upper:.3f}")

# accuracy CI
precision_lower, precision_upper = calculate_confidence_interval(predicted_positive, TP)
print(f"Precision Confidence Interval: {precision_lower:.3f}, {precision_upper:.3f}")
print(f"Accuracy:{accuracy}")
print(f"Accuracy Confidence Interval:{ci_lower, ci_upper}")
print(f"Recall:{recall}")
print(f"Recall Confidence Interval: {recall_lower:.3f}, {recall_upper:.3f}")
print(f"Precision:{precision}")
print(f"Precision Confidence Interval: {precision_lower:.3f}, {precision_upper:.3f}")