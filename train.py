import copy
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import torch


def sort_predictions(predictions, actuals, video_ids):
    unique_vids = np.unique(video_ids)
    predictions_per_video = {vid: [] for vid in unique_vids}
    actual_per_video = {vid: None for vid in unique_vids}
    for prediction, actual, video_id in zip(predictions, actuals, video_ids):
        predictions_per_video[video_id].append(prediction)
        if actual_per_video[video_id] is None:
            actual_per_video[video_id] = actual
        assert actual_per_video[video_id] == actual, \
            f"Actual values are not equal, {actual_per_video[video_id]} {actual}"
    actuals = []
    for vid in unique_vids:
        actuals.append(actual_per_video[vid])
    return predictions_per_video, actuals, unique_vids


def aggregate_predictions(predictions, actuals, video_ids, aggregator):
    predictions_per_video, actuals, unique_vids = sort_predictions(
        predictions, actuals, video_ids)
    aggregated_predictions = []
    for vid in unique_vids:
        aggregated_predictions.append(aggregator(predictions_per_video[vid]))
    return aggregated_predictions, actuals, unique_vids


def split_training(data: pd.DataFrame, shuffle=True, split=0.8):
    ids = np.random.permutation(
        list(data.index)) if shuffle else list(data.index)
    split_index = int(len(ids) * split)
    return data.loc[ids[:split_index]], data.loc[ids[split_index:]]


def build_matrixes(data, target_name, feature_name, dtype=np.float32):
    targets = []
    features = []
    video_ids = []
    for video_id, row in data.iterrows():
        feature = np.array(row[feature_name])
        target = row[target_name]
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
        else:
            raise RuntimeError(
                f"Provided feature has unexpected number of dimensions: {len(feature.shape)}")
    target_matrix = np.array(targets, dtype=dtype).reshape((len(targets), 1))
    feature_matrix = np.array(features, dtype=dtype)
    return feature_matrix, target_matrix, video_ids


def get_predictions(model_type, model, features, targets, video_ids, aggregate_with=np.median):
    if "two_layer_nn" == model_type:
        predictions, actuals = get_nn_predictions(model, features, targets)
    elif "svr" == model_type:
        predictions, actuals = get_svr_predictions(model, features, targets)
    else:
        raise ValueError(f"'{model_type}' is not a valid model type")

    return aggregate_predictions(predictions, actuals, video_ids, aggregator=aggregate_with)

########################
# Using Neural network #
########################


class FlatDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x


def train_two_layer_nn(features_train,
                       targets_train,
                       features_valid,
                       targets_valid,
                       hidden_dim=256,
                       num_epochs=15,
                       cuda=True,
                       batch_size=32,
                       learning_rate=1e-3,
                       verbose=False):
    data_train = FlatDataset(features_train, targets_train)
    data_valid = FlatDataset(features_valid, targets_valid)
    input_dim = len(features_train[0])
    output_dim = 1 if len(targets_train.shape) == 1 else len(targets_train[0])
    print("input dimensions:", input_dim) if verbose else 0
    print("output dimension:", output_dim) if verbose else 0
    model = TwoLayerNet(input_dim, output_dim, hidden_dim=hidden_dim)

    device = torch.device("cuda") if cuda else torch.device("cpu")
    model = model.float().to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)
    dataloader_train = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(data_valid, batch_size=1)

    def train(model, dataloader_train, dataloader_valid, loss_fn, optimizer, scheduler, device):

        for is_training in [True, False]:  # Epoch is a training followed by validation

            model.train() if is_training else model.eval()

            running_loss = 0
            for features, targets in (dataloader_train if is_training else dataloader_valid):
                features = features.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(is_training):
                    outputs = model(features)
                    loss = loss_fn(outputs, targets)
                    if is_training:
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()

            if is_training:
                scheduler.step()

            if is_training:
                train_loss = running_loss
            else:
                valid_loss = running_loss

        return train_loss, valid_loss

    best_valid_loss = 9999999
    best_model_state_dict = copy.deepcopy(model.state_dict())
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        print(
            f"--------------- Epoch {epoch} ----------------") if verbose else 0
        train_loss, valid_loss = train(model=model,
                                       dataloader_train=dataloader_train,
                                       dataloader_valid=dataloader_valid,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       device=device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print("New Best Validiation Loss!!!", valid_loss) if verbose else 0
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print("Training Loss:", train_loss) if verbose else 0
        print("Validation Loss:", valid_loss) if verbose else 0

    print("\n\nFINISHED TRAINING") if verbose else 0
    print(f"Best validation lost: {best_valid_loss}") if verbose else 0
    best_model = model
    best_model = best_model.eval()
    best_model.load_state_dict(best_model_state_dict)
    return best_model, train_losses, valid_losses


def get_nn_predictions(model, features_valid, targets_valid, cuda=True):

    device = torch.device("cuda") if cuda else torch.device("cpu")
    dataloader = torch.utils.data.DataLoader(
        FlatDataset(features_valid, targets_valid), batch_size=1)

    predictions = []
    actuals = []

    for feature, actual in dataloader:
        with torch.set_grad_enabled(False):
            feature = feature.to(device)
            output = model(feature)
            predictions.append(output.cpu().numpy()[0])
            actuals.append(actual.cpu().numpy()[0])

    return predictions, actuals

#############
# Using SVR #
#############


def train_svr(features_train, targets_train):
    model = SVR(kernel='rbf', C=0.1, epsilon=0.001, gamma='scale')
    model.fit(features_train, targets_train.ravel())
    return model


def get_svr_predictions(model, features_valid, targets_valid):
    predictions = model.predict(features_valid)
    actuals = targets_valid.ravel()
    return predictions, actuals
