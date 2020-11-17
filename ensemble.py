import pandas as pd
import numpy as np
from scipy import stats

PREDICTIONS_DIR = "predictions"


def get_prediction_df(df):
    preds = []
    names = []
    actual = None
    video_ids = None
    is_training = None

    for _, row in df.iterrows():
        prediction_data = pd.read_csv(row["predictions"])
        if actual is None:
            actual = np.array(prediction_data["actual"])
            actual_no_nan = actual[~np.isnan(actual)]
            is_training = np.array(prediction_data["in_training_set"])
            video_ids = prediction_data["video_id"]

        file_actual = np.array(prediction_data["actual"])
        assert np.allclose(actual_no_nan, file_actual[~np.isnan(file_actual)])
        if not np.equal(is_training, np.array(prediction_data["in_training_set"])).all():
            for a, b, vid in zip(is_training, prediction_data["in_training_set"], video_ids):
                if a != b:
                    print(a, b, vid)
        assert np.equal(video_ids, np.array(prediction_data["video_id"])).all()

        names.append(row["name"])
        preds.append(prediction_data["prediction"])

    pred_df = pd.DataFrame({
        "actual": actual,
        "video_id": video_ids,
        "is_training": is_training,
    })

    for pred, name in zip(preds, names):
        pred_df[name] = pred

    return pred_df.set_index("video_id")


def get_valid_matrixes(df):
    not_train_df = df[df["is_training"] == False]
    valid_df = not_train_df[np.logical_not(np.isnan(not_train_df["actual"]))]
    targets = valid_df["actual"]
    vids = valid_df.index
    feature_df = valid_df.drop(["actual", "is_training"], axis=1)
    return np.array(feature_df), np.array(targets), vids, list(feature_df.columns)


def get_test_matrixes(df):
    not_train_df = df[df["is_training"] == False]
    test_df = not_train_df[np.isnan(not_train_df["actual"])]
    targets = test_df["actual"]
    vids = test_df.index
    feature_df = test_df.drop(["actual", "is_training"], axis=1)
    return np.array(feature_df), np.array(targets), vids, list(feature_df.columns)


def generate_splits(n, num_splits):
    if num_splits == 1:
        yield [n]
    elif n == 0:
        yield [0 for _ in range(num_splits)]
    else:
        for i in range(n + 1):
            for subsplit in generate_splits(n-i, num_splits-1):
                yield [i] + subsplit


def calculate_splits(n, features, feature_matrix, target_matrix, seed=1):
    num_splits = len(features)

    df = pd.DataFrame(columns=["spearman_rank", "p-value"] + list(features))

    for split in generate_splits(n, num_splits):
        fractions = np.array(split) / n
        spearman_rank, p = stats.spearmanr(
            target_matrix,
            np.dot(feature_matrix, fractions))
        weights = {
            feature: weight for feature, weight in zip(features, fractions)
        }
        weights["spearman_rank"] = spearman_rank
        weights["p-value"] = p
        df = df.append(weights, ignore_index=True)

    return df.sort_values("spearman_rank", ascending=False).reset_index(drop=True)


def get_ensemble_predictions(feature_matrix, features, split):
    weights = [split[f] for f in features]
    return np.dot(feature_matrix, weights)


if __name__ == "__main__":
    model_data = pd.read_csv(f"{PREDICTIONS_DIR}/model_data.csv")

    for is_short_term in [True, False]:
        for seed in [42, 1, 9, 8, 7]:

            run_df = model_data[model_data["seed"] == seed]
            run_df = run_df[run_df["is_short_term"] == is_short_term]
            pred_df = get_prediction_df(run_df)

            X_valid, y_valid, vids_valid, features = get_valid_matrixes(
                pred_df)
            X_test, y_test, vids_test, _features = get_test_matrixes(pred_df)
            assert features == _features

            split_df = calculate_splits(20, features, X_valid, y_valid)
            split_df.to_csv(
                f"{'st' if is_short_term else 'lt'}_ensemble_{seed}.csv")

            best_split = split_df.iloc[0]
            print("#############################")
            print(best_split)

            preds_valid = get_ensemble_predictions(
                X_valid, features, best_split)
            print(f"SPEARMAN RANK: {stats.spearmanr(preds_valid, y_valid)[0]}")

            preds_test = get_ensemble_predictions(X_test, features, best_split)
            confidence = np.ones(len(preds_test))

            final = pd.DataFrame({
                "videoname": vids_test,
                "memorability_score": preds_test,
                "confidence": confidence
            })

            final.to_csv(
                f"me19mem_ucb_{'shorterm' if is_short_term else 'longterm'}_run{seed}.csv",
                header=False, index=False
            )
