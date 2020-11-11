DATASET = "training_set"


# Images (Frames)
FRAME_DIR = f"{DATASET}/Frames"
NUM_FRAMES_PER_VIDEO = 8  # number of frame we want to extract
RESNET152_FEATURE_DIR = f"{DATASET}/Features/ResNet152"
HOG_FEATURE_DIR = f"{DATASET}/Features/HOG"
LBP_FEATURE_DIR = f"{DATASET}/Features/LBP"

# Audio
AUDIO_DIR = f"{DATASET}/Audio"
VGGISH_FEATURE_DIR = f"{DATASET}/Features/VGGish"

# Emotion
EMOTION_FEATURE_DIR = f"{DATASET}/Features/Emotion"
MAX_FACES = 8
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Video
C3D_FEATURE_DIR = f"{DATASET}/Features/C3D"

# Text
GLOVE_WORD_EMBEDDINGS_PATH = "data/glove.6B/glove.6B.300d.txt"
GLOVE_WORD_EMBEDDING_DIM = 300
GLOVE_FEATURE_DIR = f"{DATASET}/Features/GloVe"


def set_dataset(dataset, constant):
    '''Given constant is a string such as 'training_set/Video', returns '{dataset}/Video'. '''
    return f"{dataset}{constant[len(DATASET):]}"
