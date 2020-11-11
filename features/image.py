import numpy as np
import os
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
from tqdm import tqdm

from features.config import RESNET152_FEATURE_DIR, HOG_FEATURE_DIR, LBP_FEATURE_DIR, FRAME_DIR, NUM_FRAMES_PER_VIDEO


def load_ResNet152_features(video_ids, path=RESNET152_FEATURE_DIR):
    features = {video_id: [] for video_id in video_ids}
    for filename in os.listdir(path):
        features[int(filename[:5])].append(np.loadtxt(f"{path}/{filename}"))
    return [np.array(features[video_id])[0] for video_id in video_ids]


def load_HOG_features(video_ids, path=HOG_FEATURE_DIR):
    '''Histogram of Oriented Gradients'''
    features = {video_id: [] for video_id in video_ids}
    for filename in os.listdir(path):
        video_id = int(filename[:5])
        features[video_id].append(np.loadtxt(
            f"{path}/{filename}", delimiter=","))
    return [np.array(features[video_id]) for video_id in video_ids]


def load_LBP_features(video_ids, path=LBP_FEATURE_DIR):
    '''Local Binary Pattern'''
    features = {video_id: [] for video_id in video_ids}
    for filename in os.listdir(path):
        video_id = int(filename[:5])
        features[video_id].append(np.loadtxt(
            f"{path}/{filename}", delimiter=","))
    return [np.array(features[video_id]) for video_id in video_ids]


def extract_image_features(model, features_dir, image_batches, idx_to_video_id, cuda=torch.cuda.is_available()):
    model = model.cuda().eval() if cuda else model.eval()
    for batch, idx in tqdm(image_batches):
        feature_path = f"{features_dir}/{idx_to_video_id[int(idx[0])]}.csv"
        if not os.path.exists(feature_path):
            assert np.all((idx == idx[0]).numpy(
            )), "Batch contains frames from multiple videos"
            with torch.no_grad():
                features = model(batch.cuda() if cuda else batch).cpu()
            np.savetxt(feature_path, features.numpy())


def extract_resnet152_features(image_dir=FRAME_DIR, features_dir=RESNET152_FEATURE_DIR, num_frames_per_video=NUM_FRAMES_PER_VIDEO):

    if not os.path.isdir(features_dir):
        os.mkdir(features_dir)

    # Load pretrained resnet
    resnet152 = torchvision.models.resnet152(pretrained=True)
    # Replace last layer with identity
    resnet152.fc = torch.nn.Identity()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    images = ImageFolder(image_dir, transform=preprocess)
    batched_images = torch.utils.data.DataLoader(
        images, batch_size=num_frames_per_video)

    idx_to_video_id = {idx: str(video_id)
                       for video_id, idx in images.class_to_idx.items()}

    extract_image_features(resnet152, features_dir,
                           batched_images, idx_to_video_id)
