import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

path = r'C:\Users\zhang\rnn_tsf\models\cnn\resnet_gastric_23_02_2014.keras'
res_gas_model_keras = load_model(path)

def slice_load(file_list):    
    images=[]    
    for filename in file_list:
        im = image.load_img(filename,target_size=(224, 224, 3)) 
        b = image.img_to_array(im)
        images.append(b)
    return images

def get_prediction(path):
    X_inpute_image=slice_load(path)
    X_inpute_array=np.array(X_inpute_image)/255
    pred = res_gas_model_keras.predict(X_inpute_array)
    print('Network prediction:', round(pred[0][0],4))
    return round(pred[0][0],4)
#path_c = [r'G:\BaiduNetdiskDownload\train2\validation\cancer_v\cancer_subset07\2017-06-10_19.46.43.ndpi.16.43169_13551.2048x2048.tiff']
#get_prediction(path_c)

# VIT model
os.environ["KERAS_BACKEND"] = "jax" 
from sklearn.model_selection import train_test_split, GridSearchCV , cross_val_score
input_shape = (224, 224, 3)
image_size = 224 
patch_size = 32 
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
neuron_units = [projection_dim * 2, projection_dim] 
transformer_blocks = 8 
mlp_neuron_units = [2048, 1024] 

data_augmentation_path = 'C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\train\\fifth_train\\data_augmentation.keras'
data_augmentation_loaded = tf.keras.models.load_model(data_augmentation_path)

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
    augmented = data_augmentation_loaded(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_blocks):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)              
        # MLP.
        x3 = mlp(x3, hidden_units=neuron_units, dropout_rate=0.1) # transformer_units
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_neuron_units, dropout_rate=0.5)    
    # for binary classification the last layer is updated as following
    result = layers.Dense(1, activation='sigmoid')(features)    
    model = keras.Model(inputs=inputs, outputs= result)
    return model

loaded_vit_modle = create_vit_classifier()
checkpoint_filepath = 'C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\train\\fifth_train\\checkpoint.weights_5.h5'
loaded_vit_modle.load_weights(checkpoint_filepath)   

def slice_load2(file_list):    
    images=[]    
    for filename in file_list:
        im = image.load_img(filename,target_size=(224, 224, 3))
 
        b = image.img_to_array(im)
        images.append(b)
    return images

def get_prediction_vit(path):
    tif_test1 = slice_load2(path)
    tif_test2 = np.array(tif_test1)/255    
    tensor_value = loaded_vit_modle(tif_test2)
    tensor_value = tensor_value.numpy()
    print('Vit prediction:', tensor_value[0, 0])
    return(tensor_value[0, 0]) 

# another test
#tif_test_file_no = ["C:\\Users\\zhang\\rnn_tsf\\vit\\real_data\\test\\Normal5.ndpi.16.5905_17200.2048x2048_noconcer.tiff"]
#get_prediction_vit(tif_test_file)

# NLP prediction 
#import tensorflow as tf
import re
wordList = np.load('C:\\Users\\zhang\\rnn_tsf\\nlp\\wordsList.npy')
wordList = wordList.tolist()
wordList = [word.decode('UTF-8') for word in wordList]
model_path = "C:\\Users\\zhang\\rnn_tsf\\models\\pretrained_lstm_complete_model20000.ckpt"
lstm_nlp_model  = tf.keras.models.load_model(model_path)

def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, " ", string)

def get_prediction_lstmNLP(review):    
    cleanedLine = cleanSentences(review)
    split = cleanedLine.split()
    maxSequLength = 250
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
    ids_test_float32 = tf.cast(ids_test, tf.float32)    
    predictions0 = lstm_nlp_model(ids_test_float32, training=False)
    predictions_numpy = predictions0.numpy()
    result = predictions_numpy[0][0]
    return(result)

review = '''this movie is so boring, I have on interest to see it. I falled 
in sleep when I saw it. I will not see this movie again. 
I will not recommend it either'''    
#result = get_prediction_lstmNLP(review)#"this movie is wonderful")
