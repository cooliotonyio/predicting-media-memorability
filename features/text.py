from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from features.config import GLOVE_WORD_EMBEDDINGS_PATH, GLOVE_WORD_EMBEDDING_DIM, GLOVE_FEATURE_DIR


def load_GloVe_features(video_ids, path=GLOVE_FEATURE_DIR):
    features = []
    for video_id in video_ids:
        video_dir = f"{path}/{video_id}"
        video_features = []
        if os.path.exists(video_dir) and len(os.listdir(video_dir)) > 0:
            for caption_feature_file in np.sort(os.listdir(video_dir)):
                caption_feature = np.loadtxt(
                    f"{video_dir}/{caption_feature_file}", delimiter=",")
                video_features.append(caption_feature)
        else:
            print(f"GLoVe embeddings not found at {video_dir}")
        features.append(np.array(video_features))
    return features


def fit_tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer


def load_GLoVe_word_embeddings(path=GLOVE_WORD_EMBEDDINGS_PATH):
    """ Load pre-trained GLoVe word embeddings"""
    word_embeddings = {}
    with open(path, encoding='latin-1') as f:
        for line in f:
            word, embedding = line.split(maxsplit=1)
            embedding = np.fromstring(embedding, 'f', sep=' ')
            word_embeddings[word] = embedding
    return word_embeddings


def build_embedding_matrix(word_index, word_embeddings, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        word_embedding = word_embeddings.get(word)
        if word_embedding is not None:
            embedding_matrix[i] = word_embedding
    return embedding_matrix


def build_extractor(word_index, word_embeddings, embedding_dim, max_sequence_length):

    embedding_matrix = build_embedding_matrix(
        word_index, word_embeddings, embedding_dim)
    embedding_layer = Embedding(
        len(word_index) + 1,
        embedding_dim,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=max_sequence_length,
        trainable=False)
    input_layer = Input(shape=(max_sequence_length,), dtype='int32')

    output_layer = embedding_layer(input_layer)
    return Model(input_layer, output_layer)


def extract_text_features(video_ids, captions, tokenizer, word_embeddings, feature_dir, embedding_dim, max_sequence_length):

    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    sequences = pad_sequences(tokenizer.texts_to_sequences(
        captions), maxlen=max_sequence_length)

    extractor = build_extractor(
        tokenizer.word_index, word_embeddings, embedding_dim, max_sequence_length)
    features = extractor.predict(sequences)

    caption_counts = {vid: 0 for vid in np.unique(video_ids)}
    for video_id, feature in tqdm(list(zip(video_ids, features))):
        video_feature_dir = f"{feature_dir}/{video_id}"

        if not os.path.exists(video_feature_dir):
            os.mkdir(video_feature_dir)

        caption_feature_filename = f"{video_feature_dir}/{caption_counts[video_id]}.csv"
        np.savetxt(caption_feature_filename, feature, delimiter=",")
        caption_counts[video_id] += 1


def extract_GLoVe_features(video_ids, captions, tokenizer, max_sequence_length, feature_dir=GLOVE_FEATURE_DIR, embedding_dim=GLOVE_WORD_EMBEDDING_DIM):
    word_embeddings = load_GLoVe_word_embeddings()
    extract_text_features(video_ids, captions, tokenizer,
                          word_embeddings, feature_dir, embedding_dim, max_sequence_length)
