import copy
import pandas as pd
import numpy as np
import torch

from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

from tensorflow.keras.layers import Input, GRU, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf


def sort_predictions(predictions, actuals, video_ids):
    unique_vids = np.unique(video_ids)
    predictions_per_video = {vid: [] for vid in unique_vids}
    actual_per_video = {vid: None for vid in unique_vids}
    for prediction, actual, video_id in zip(predictions, actuals, video_ids):
        predictions_per_video[video_id].append(prediction)
        if actual_per_video[video_id] is None:
            actual_per_video[video_id] = actual
        assert actual_per_video[video_id] == actual or np.isnan(actual_per_video[video_id]), \
            f"Actual values are not equal, {actual_per_video[video_id]} {actual}"
    actuals = []
    for vid in unique_vids:
        actuals.append(actual_per_video[vid])
    return predictions_per_video, actuals, unique_vids


def aggregate_predictions(predictions, actuals, video_ids, aggregator):
    predictions_per_video, actuals, unique_vids = sort_predictions(
        predictions, actuals, video_ids)
    aggregated_predictions = [
        aggregator(predictions_per_video[vid]) for vid in unique_vids]
    return aggregated_predictions, actuals, unique_vids


def split_training(data: pd.DataFrame, shuffle=True, split=0.8):
    ids = np.random.permutation(
        list(data.index)) if shuffle else list(data.index)
    split_index = int(len(ids) * split)
    return data.loc[ids[:split_index]], data.loc[ids[split_index:]]


def build_matrixes(data, target_name, feature_name, dtype=np.float32, is_test=False):
    targets = []
    features = []
    video_ids = []
    for video_id, row in data.iterrows():
        feature = np.array(row[feature_name])
        target = np.nan if is_test else row[target_name]
        if len(feature) == 0:
            print(
                f"WARNING: Video '{video_id}' has no '{feature_name}' features")
            continue
        if len(feature.shape) == 1:
            targets.append(target)
            video_ids.append(video_id)
            features.append(feature)
        elif len(feature.shape) == 2:
            for embedding in feature:
                targets.append(target)
                video_ids.append(video_id)
                features.append(embedding)
        elif len(feature.shape) == 3:
            for embedding in feature:
                targets.append(target)
                video_ids.append(video_id)
                features.append(embedding)
        else:
            raise RuntimeError(
                f"Provided feature has unexpected number of dimensions: {len(feature.shape)}")
    target_matrix = np.array(targets, dtype=dtype).reshape((len(targets), 1))
    feature_matrix = np.array(features, dtype=dtype)
    return feature_matrix, target_matrix, video_ids


def get_predictions(model_type, model, features, targets, video_ids, aggregate_with=np.median):
    if "svr" == model_type:
        predictions = model.predict(features)
        actuals = targets.ravel()
    elif "gru" == model_type:
        predictions = model.predict(features).ravel()
        actuals = targets.ravel()
    elif "bayesian_ridge" == model_type:
        predictions = model.predict(features)
        actuals = targets.ravel()
    else:
        raise ValueError(f"'{model_type}' is not a valid model type")

    return aggregate_predictions(predictions, actuals, video_ids, aggregator=aggregate_with)

#############
# Using GRU #
#############


def build_gru_model(input_dim, gru_units, gru_dropout, lin_dropout, hidden_dim, seed):
    input_layer = Input(shape=input_dim)
    x = GRU(
        units=gru_units,
        dropout=gru_dropout,
        recurrent_dropout=gru_dropout,
        return_sequences=False
    )(input_layer)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(lin_dropout, seed=seed)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(lin_dropout, seed=seed)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(lin_dropout, seed=seed)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, output_layer)


def train_gru(features_train,
              targets_train,
              features_valid,
              targets_valid,
              gru_units=64,
              gru_dropout=0.75,
              lin_dropout=0.25,
              hidden_dim=1024,
              learning_rate=1e-3,
              num_epochs=15,
              batch_size=32,
              verbose=False,
              seed=1):
    tf.random.set_seed(seed)
    input_dim = features_train[0].shape
    print("input dimensions:", input_dim) if verbose else 0
    print("hidden dimension:", hidden_dim) if verbose else 0
    model = build_gru_model(
        input_dim=input_dim,
        gru_units=gru_units,
        gru_dropout=gru_dropout,
        lin_dropout=lin_dropout,
        hidden_dim=hidden_dim,
        seed=seed
    )

    optimizer = Adam(lr=learning_rate, decay=learning_rate / num_epochs)

    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mse", "mae", "mape"]
    )

    early_stopping_monitor = EarlyStopping(
        monitor="val_loss",
        patience=num_epochs // 5,
        verbose=verbose,
        restore_best_weights=True
    )

    H = model.fit(features_train, targets_train,
                  validation_data=(features_valid, targets_valid),
                  epochs=num_epochs,
                  shuffle=False,
                  batch_size=32,
                  use_multiprocessing=True,
                  workers=8,
                  callbacks=[early_stopping_monitor],
                  verbose=False)

    train_losses = H.history["loss"]
    valid_losses = H.history["val_loss"]

    return model, train_losses, valid_losses


#############
# Using SVR #
#############


def train_svr(features_train, targets_train):
    model = SVR(kernel='rbf', C=0.1, epsilon=0.001, gamma='scale')
    model.fit(features_train, targets_train.ravel())
    return model

#######################
# Using BayesianRidge #
#######################


def train_bayesian_ridge(features_train, targets_train):
    model = BayesianRidge()
    model.fit(features_train, targets_train.ravel())
    return model


def train_model(model_type, features_train, targets_train, features_valid, targets_valid, model_parameters):

    np.random.seed(model_parameters["random_seed"])

    if "svr" == model_type:
        model = train_svr(features_train, targets_train)

    elif "gru" == model_type:
        model, train_losses, valid_losses = train_gru(
            features_train, targets_train, features_valid, targets_valid,
            hidden_dim=model_parameters["hidden_dim"],
            num_epochs=model_parameters["num_epochs"],
            gru_units=model_parameters["gru_units"],
            batch_size=model_parameters["batch_size"],
            learning_rate=model_parameters["learning_rate"],
            verbose=True,
            gru_dropout=model_parameters["gru_dropout"],

        )

    elif "bayesian_ridge" == model_type:
        model = train_bayesian_ridge(features_train, targets_train)

    else:
        raise ValueError(f"'{model_type}' is not a valid model type")
    return model
