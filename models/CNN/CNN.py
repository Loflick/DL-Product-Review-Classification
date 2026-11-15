"""CNN Model for Review Classification """
#AI Disclamier: AI was used in the devolpment of this code
#Additionally it was used to further knowledge of a 1D CNN


import numpy as np
import keras 
from keras import layers 
from dataclasses import dataclass

MAX_REVIEWS = 10
EMBED_DIM = 773 
NUM_CLASSES = 5

@dataclass
class HParams:
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 10
    activation: str = 'relu'
    n_filters: int = 100
    n_neurons: int = 128
    dropout: float = 0.5

def compute_mask(t):
    return keras.ops.sum(keras.ops.abs(t),axis=-1) > 0

# # attention:
# attn_mask = ___(mask)                              # (B, L, L)
# attn = layers.MultiHeadAttention(...)(x, x, attention_mask=attn_mask)
def compute_attn_mask(t):
    return keras.ops.repeat(keras.ops.expand_dims(t,axis=1), MAX_REVIEWS, axis=1)



class MaskedGlobalMaxPool(layers.Layer):
    def call(self, inputs):
        features, mask = inputs          # (B, L, F), (B, L) bool
        # 1. expand mask to (B, L, 1)
        mask = keras.ops.expand_dims(mask,-1)
        # 2. where mask is True keep features, else a very negative number
        features = keras.ops.where(mask, features,-1e9)
        # 3. max over the sequence axis -> (B, F)
        return keras.ops.max(features, axis=1)




def build_model(hp, num_classes=NUM_CLASSES, kernel_sizes=(3,4,5)):
    inputs = keras.Input(shape=(MAX_REVIEWS,EMBED_DIM))
    # in build_model, right after `inputs`:
    mask = keras.ops.cast(layers.Lambda(compute_mask)(inputs),"bool")      # (B, L) bool, True = real review
    attn_mask = keras.ops.cast(layers.Lambda(compute_attn_mask)(mask),"bool")
    # print(mask.dtype)
    # print(attn_mask.dtype)    
    
    #for 3 parallel convs
    conv_list = []
    for k in kernel_sizes: 
        conv_list.append(layers.Conv1D(
            filters=hp.n_filters,
            kernel_size= k,
            padding = "same",
            activation=hp.activation,
        )(inputs))

    #concatentate 
    x = layers.Concatenate(axis=-1)(conv_list)
    
    # step 4: MultiHeadAttention(self) -> add residual -> LayerNormalization -> THEN poo
    # Stage 4
    features = hp.n_filters * 3                  # what is this in terms of hp?
    attn = layers.MultiHeadAttention(
        num_heads=len(kernel_sizes),                                   # len(kernel_sizes)
        key_dim= (int)(features/len(kernel_sizes)),                                     # features / heads
        
    )(x,x,attention_mask = attn_mask)                                          # self-attention: which tensors?
    x = layers.Add()([x,attn])                         # residual
    x = layers.LayerNormalization()(x)

    # x = layers.GlobalMaxPool1D()(x)                      # pooling now comes AFTER
    x = MaskedGlobalMaxPool()([x,mask])
    x = layers.Dropout(hp.dropout)(x)
    x = layers.Dense(hp.n_neurons,activation=hp.activation)(x)
    outputs = layers.Dense(num_classes)(x) #5 class output



    return keras.Model(inputs,outputs)



# # pooling:
# x = MaskedGlobalMaxPool()([x, mask])
def build_baseline(hp, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(MAX_REVIEWS, EMBED_DIM))               # same input as the main model

    # collapse the 10 reviews into one vector per item
    x = keras.layers.GlobalAveragePooling1D()(inputs)

    x = keras.layers.Dropout(hp.dropout)(x)
    x = keras.layers.Dense(hp.n_neurons, activation=hp.activation)(x)
    outputs = keras.layers.Dense(5)(x)                        # raw logits, no activation

    return keras.Model(inputs, outputs)
def build_baseline_max(hp, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(MAX_REVIEWS, EMBED_DIM))               # same input as the main model

    # collapse the 10 reviews into one vector per item
    x = keras.layers.GlobalMaxPooling1D()(inputs)

    x = keras.layers.Dropout(hp.dropout)(x)
    x = keras.layers.Dense(hp.n_neurons, activation=hp.activation)(x)
    outputs = keras.layers.Dense(5)(x)                        # raw logits, no activation

    return keras.Model(inputs, outputs)


