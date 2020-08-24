import tensorflow as tf
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

from image_caption.utils import  CNN_Encoder, RNN_Decoder

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def feature_extractor_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model


class AttentionModel():

    def __init__(self, checkpoint_path="checkpoints/train/ckpt-6", caption_path = 'captions.pkl'):
        train_captions = pickle.load(open(caption_path, 'rb'))
        top_k = 5000
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        self.tokenizer.fit_on_texts(train_captions)
        train_seqs = self.tokenizer.texts_to_sequences(train_captions)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
        train_seqs = self.tokenizer.texts_to_sequences(train_captions)
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
        self.max_length = calc_max_length(train_seqs)

        embedding_dim   = 256
        units = 512
        vocab_size = top_k + 1
        feature_shape = 2048
        self.attention_feature_shape=64

        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,decoder=self.decoder,optimizer = optimizer)
        self.ckpt.restore(checkpoint_path)
        self.feature_extract_model = feature_extractor_model()
    
    def __evaluate_image(self, image):
        attention_plot = np.zeros((self.max_length, self.attention_feature_shape))
        hidden = self.decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = self.feature_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot

    def predict_caption(self, image_name):
        result, attention_plot = self.__evaluate_image(image_name)

        for i in result:
            if i=="<unk>":
                result.remove(i)
            else:
                pass
    
        caption = ' '.join(result).rsplit(' ', 1)[0]
        return caption    


    
    


