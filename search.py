import pandas as pd
import numpy as np
from scipy import stats
import os

import features.config as fconf
from features.video import load_C3D_features
from features.image import load_ResNet152_features
from features.audio import load_VGGish_features
from features.text import load_GloVe_features
from target_augmentation import add_position_delta, calculate_alpha_and_memorability
from train import split_training, build_matrixes, get_predictions, train_model


PREDICTIONS_DIR = "predictions"


def main(train_data, test_data, is_short_term, feature, seed, aggregate_with=np.median):

    np.random.seed(seed)

    target = "m_75" if is_short_term else "part_2_scores"

    if "glove" == feature:
        model_type = "gru"
    elif "resnet152" == feature:
        model_type = "svr"
    elif "c3d" == feature:
        model_type = "svr"
    elif "vggish" == feature:
        model_type = "bayesian_ridge"

    model_name = f"{feature}_{model_type}"
    model_parameters = {
        "random_seed": seed
    }
    if model_type == "gru":

        model_parameters["num_epochs"] = 150
        model_parameters["hidden_dim"] = 64
        model_parameters["learning_rate"] = 1e-3
        model_parameters["batch_size"] = 64
        model_parameters["gru_units"] = 64
        model_parameters["gru_dropout"] = 0.8

    # Data prep
    training_data, validation_data = split_training(train_data)

    features_train, targets_train, video_ids_train = build_matrixes(
        training_data, target_name=target, feature_name=feature)
    features_valid, targets_valid, video_ids_valid = build_matrixes(
        validation_data, target_name=target, feature_name=feature)
    features_test, targets_test, video_ids_test = build_matrixes(
        test_data, target_name=target, feature_name=feature, is_test=True)

    # Training
    model = train_model(model_type, features_train, targets_train,
                        features_valid, targets_valid, model_parameters)

    # Evaluation
    pred_train, actual_train, vid_train = get_predictions(
        model_type, model, features_train, targets_train, video_ids_train, aggregate_with=aggregate_with)
    pred_valid, actual_valid, vid_valid = get_predictions(
        model_type, model, features_valid, targets_valid, video_ids_valid, aggregate_with=aggregate_with)
    pred_test, actual_test, vid_test = get_predictions(
        model_type, model, features_test, targets_test, video_ids_test, aggregate_with=aggregate_with)

    valid_spearman_rank, _ = stats.spearmanr(actual_valid, pred_valid)

    predictions = np.concatenate([pred_train, pred_valid, pred_test])
    actuals = np.concatenate([actual_train, actual_valid, actual_test])
    video_ids = np.concatenate([vid_train, vid_valid, vid_test])
    in_training_set = np.array(np.concatenate(
        [np.ones(len(pred_train)), np.zeros(len(pred_valid + pred_test))]), dtype=bool)

    default_prediction = np.mean(predictions)
    for vid in train_data.index:
        if vid not in video_ids:
            video_ids = np.append(video_ids, vid)
            predictions = np.append(predictions, default_prediction)
            in_training_set = np.append(
                in_training_set, vid in training_data.index)
            actuals = np.append(actuals, train_data.loc[vid][target])

    for vid in test_data.index:
        if vid not in video_ids:
            video_ids = np.append(video_ids, vid)
            predictions = np.append(predictions, default_prediction)
            in_training_set = np.append(in_training_set, False)
            actuals = np.append(actuals, np.nan)

    # Save predictions
    save_predictions(model_name, video_ids, actuals, predictions, in_training_set,
                     model_parameters, is_short_term, model_type, valid_spearman_rank
                     )


def add_features_to_df(dfs, set_names, label, feature_dir, load_func):
    for df, set_name in zip(dfs, set_names):
        df[label] = load_func(
            df.index, fconf.set_dataset(set_name, feature_dir))


def save_predictions(model_name, video_ids, actuals, predictions, in_training_set,
                     model_parameters, is_short_term, model_type, valid_spearman_rank,
                     predictions_dir=PREDICTIONS_DIR):

    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)

    model_data_dir = f"{predictions_dir}/model_data.csv"
    if not os.path.exists(model_data_dir):
        model_data = pd.DataFrame(columns=["name", "seed", "is_short_term", "validation_spearman_rank", "type",
                                           "feature", "predictions", "notes", "parameters"])
    else:
        model_data = pd.read_csv(model_data_dir)

    model_dir = f"{predictions_dir}/{model_name}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    pred_filename = f"{model_dir}/{'st' if is_short_term else 'lt'}-{model_parameters['random_seed']}.csv"

    pred_data = pd.DataFrame({
        "video_id": video_ids,
        "prediction": predictions,
        "actual": actuals,
        "in_training_set": in_training_set
    }).sort_values("video_id")

    pred_data.to_csv(pred_filename, index=False)

    model_info = {
        "feature": feature,
        "seed": model_parameters["random_seed"],
        "is_short_term": is_short_term,
        "validation_spearman_rank": np.around(valid_spearman_rank, 4),
        "name": model_name,
        "type": model_type,
        "predictions": pred_filename,
        "notes": "",
        "parameters": model_parameters
    }

    model_data.append(model_info, ignore_index=True).to_csv(
        model_data_dir, index=False)

    print("####################################################")
    print(
        f"SEED {model_parameters['random_seed']}, ST? {is_short_term}, feature {feature}")
    print("SPEARMAN: ", np.around(valid_spearman_rank, 4))
    print("Saved model info and predictions: ", model_info)
    print("####################################################")


if __name__ == "__main__":
    testing_set_data = pd.read_csv(
        "testing_set/test_urls.csv").set_index("video_id")
    training_set_data = pd.read_csv(
        "training_set/scores_v2.csv").set_index("video_id")

    dfs = [testing_set_data, training_set_data]
    set_names = ["testing_set", "training_set"]

    add_features_to_df(dfs, set_names, "glove",
                       fconf.GLOVE_FEATURE_DIR, load_GloVe_features)
    add_features_to_df(dfs, set_names, "resnet152",
                       fconf.RESNET152_FEATURE_DIR, load_ResNet152_features)
    add_features_to_df(dfs, set_names, "c3d",
                       fconf.C3D_FEATURE_DIR, load_C3D_features)
    add_features_to_df(dfs, set_names, "vggish",
                       fconf.VGGISH_FEATURE_DIR, load_VGGish_features)

    train_data = training_set_data
    test_data = testing_set_data

    # Target augmentation
    annotations = add_position_delta(pd.read_csv(
        "training_set/short_term_annotations_v2.csv"))

    big_t = int(np.around(np.mean(annotations["t"])))
    label = f"m_{big_t}"
    _alpha, adjusted_score = calculate_alpha_and_memorability(
        annotations, T=big_t)
    train_data[label] = adjusted_score
    test_data[label] = np.nan

    for is_short_term in [True, False]:
        for seed in [42, 1, 9, 8, 7]:
            for feature in ["glove", "vggish", "c3d", "resnet152"]:
                main(train_data, test_data, is_short_term, feature, seed)
