import numpy as np
import os
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from features.config import VGGISH_FEATURE_DIR, AUDIO_DIR


def load_VGGish_features(video_ids, path=VGGISH_FEATURE_DIR):
    features = {video_id: [] for video_id in video_ids}
    for filename in os.listdir(path):
        array = np.loadtxt(f"{path}/{filename}")
        if len(array.shape) == 1:
            array = np.reshape(array, (1, array.shape[0]))
        features[int(filename[:-4])].append(array)
    return [np.array(features[video_id])[0] if len(features[video_id]) else np.array([]) for video_id in video_ids]


def extract_VGGish_features(
        video_ids, vggish_slim, vggish_params, vggish_input, model_ckpt_path,
        audio_dir=AUDIO_DIR, embedding_dir=VGGISH_FEATURE_DIR):

    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)

    wav_filenames = [f"{audio_dir}/{video_id}.wav" for video_id in video_ids]
    embedding_filenames = [
        f"{embedding_dir}/{video_id}.csv" for video_id in video_ids]

    lengths = []
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, model_ckpt_path)

        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        for wav_filename, embedding_filename in tqdm(list(zip(wav_filenames, embedding_filenames))):
            if os.path.exists(wav_filename) and not os.path.exists(embedding_filename):
                examples_batch = vggish_input.wavfile_to_examples(wav_filename)
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})
                lengths.append(embedding_batch.shape[0])
                np.savetxt(embedding_filename, embedding_batch)
            else:
                print(
                    f"Skipped extracting {embedding_filename} from {wav_filename}")

    if len(lengths):
        print("Max number of VGGish embedding for a video:", np.max(lengths))
        print("Min number of VGGish embedding for a video:", np.min(lengths))
        print("Avg number of VGGish embedding for a video:", np.mean(lengths))
