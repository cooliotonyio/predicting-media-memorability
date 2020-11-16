import pandas as pd
import numpy as np
import pickle
import os

from feature_models.vggish import vggish_input, vggish_params, vggish_slim
import features.config as fconf
from features.audio import extract_VGGish_features
from features.image import extract_resnet152_features
from features.text import fit_tokenizer, extract_GLoVe_features
from features.emotion import extract_emotions

DATASETS = ["training_set", "development_set", "testing_set"]
CAPTION_FILES = {
    "training_set": "text_descriptions.csv",
    "development_set": "dev_text_descriptions.csv",
    "testing_set": "test_text_descriptions.csv"
}
FEATURE_MODEL_DIR = "feature_models/"


def get_boolean_input(prompt):
    while True:
        ans = input(prompt).strip().lower()
        if ans == "y":
            return True
        elif ans == "n":
            return False
        print("Invalid response.")


def extract_features(
        datasets=DATASETS,
        caption_files=CAPTION_FILES,
        frame_subdir="Frames",
        feature_model_dir=FEATURE_MODEL_DIR,
        audio_subdir="Audio"):

    extract_all = get_boolean_input(
        "Extract all features? (y/n) Otherwise extract on per-feature basis.")

    if extract_all or get_boolean_input("Extract emotion features? (y/n)"):
        for dataset in datasets:
            feature_dir = fconf.set_dataset(dataset, fconf.EMOTION_FEATURE_DIR)
            print(f"Extracting emotion features for {dataset}...")
            _ = extract_emotions(
                frame_path=f"{dataset}/{frame_subdir}", feature_dir=feature_dir)
        print("Finished emotion feature extraction.\n")

    if extract_all or get_boolean_input("Extract text (GLoVe) features? (y/n)"):
        caption_paths = [
            f"{dataset}/{caption_files[dataset]}" for dataset in datasets]
        all_caption_data = pd.concat(
            [pd.read_csv(path) for path in caption_paths])
        all_video_ids = all_caption_data["video_id"]
        all_captions = all_caption_data["description"]

        caption_count = all_video_ids.groupby(all_video_ids).count()
        print(
            f"\tFound {len(all_captions)} captions for {len(all_video_ids.unique())} videos")
        print("\tAvg num of captions per video:", np.mean(caption_count))
        print("\tMax num of captions per video:", np.max(caption_count))
        print("\tMin num of captions per video:", np.min(caption_count))

        sequence_lengths = [len(caption.split()) for caption in all_captions]
        max_sequence_length = np.max(sequence_lengths)

        print("Max sequence length:", max_sequence_length)
        print("Min sequence length:", np.min(sequence_lengths))
        print("Avg sequence length:", np.mean(sequence_lengths))

        print("Fitting tokenizer...")
        tokenizer = fit_tokenizer(all_captions)

        for dataset, caption_path in zip(datasets, caption_paths):
            print(f"Extracting GLoVe features from {caption_path}...")
            caption_data = pd.read_csv(caption_path)
            video_ids = caption_data["video_id"]
            captions = caption_data["description"]
            feature_dir = fconf.set_dataset(dataset, fconf.GLOVE_FEATURE_DIR)
            extract_GLoVe_features(
                video_ids=video_ids,
                captions=captions,
                tokenizer=tokenizer,
                max_sequence_length=max_sequence_length,
                feature_dir=feature_dir)

        tokenizer_pickle_path = f"{feature_model_dir}/caption_tokenizer.pickle"
        print(f"Saving tokenizer at {tokenizer_pickle_path}")
        with open(tokenizer_pickle_path, "wb") as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Finished GLoVe feature extraction.\n")

    if extract_all or get_boolean_input("Extract image (ResNet152) features? (y/n)"):
        for dataset in datasets:
            frame_dir = f"{dataset}/{frame_subdir}"
            feature_dir = fconf.set_dataset(
                dataset, fconf.RESNET152_FEATURE_DIR)
            print(f"Extracting ResNet152 features for {dataset}...")
            extract_resnet152_features(
                image_dir=frame_dir, features_dir=feature_dir)
        print("Finished ResNet152 feature extraction.\n")

    if extract_all or get_boolean_input("Extract audio (VGGish) features? (y/n)"):
        for dataset in datasets:
            audio_dir = f"{dataset}/{audio_subdir}"
            video_ids = [f.split(".")[0]
                         for f in os.listdir(audio_dir) if f[-4:] == ".wav"]
            feature_dir = fconf.set_dataset(dataset, fconf.VGGISH_FEATURE_DIR)
            print(f"Extracting VGGish features for {dataset}...")
            extract_VGGish_features(
                video_ids,
                vggish_slim=vggish_slim,
                vggish_params=vggish_params,
                vggish_input=vggish_input,
                model_ckpt_path=f"{feature_model_dir}/vggish/vggish_model.ckpt",
                embedding_dir=feature_dir,
                audio_dir=audio_dir)
        print("Finished VGGish feature extraction.\n")


if __name__ == "__main__":
    extract_features()
