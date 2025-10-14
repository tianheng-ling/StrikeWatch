import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, ConcatDataset, DataLoader


def load_data(data_config: dict):
    data_files = [
        f for f in os.listdir(data_config["data_file_path"]) if not f.startswith(".")
    ]
    data_dict = {}
    for file_name in data_files:
        person_name, strike_type = (
            file_name.split("_")[0],
            file_name.split("_")[1].split(".")[0],
        )
        df = pd.read_csv(os.path.join(data_config["data_file_path"], file_name))
        df = df[data_config["feature_cols"]]

        if person_name not in data_dict:
            data_dict[person_name] = {}

        data_dict[person_name][strike_type] = df
        data_dict[person_name]["data_len"] = len(df)

    return data_dict


def balance_data(data_dict: dict):
    min_data_len = min([data_dict[person]["data_len"] for person in data_dict])

    for _, data in data_dict.items():
        for strike_type in list(data.keys()):
            if strike_type == "data_len":
                continue
            data[strike_type] = data[strike_type][:min_data_len]
        data["data_len"] = min_data_len

    return data_dict


LABEL_MAP = {
    "forefootstrike": 0,
    "heelstrike": 1,  # as abnormal
}


class StrikeWatchDataset(Dataset):
    def __init__(
        self,
        data_dict: dict,
        window_size: int,
        downsampling_rate: int,
        stride: int,
    ):
        self.window_size = window_size
        self.downsampling_rate = downsampling_rate
        self.stride = stride
        self.features, self.labels = self.create_dataset(data_dict)

    def create_dataset(self, data_dict: dict):
        features, labels = [], []

        for strike_type, label in LABEL_MAP.items():
            if strike_type not in data_dict:
                raise ValueError(f"Strike type {strike_type} not found in data_dict.")
            df = data_dict[strike_type]

            for i in range(0, len(df) - self.window_size + 1, self.stride):
                sample = df.iloc[i : i + self.window_size].values
                processed_sample = sample[:: self.downsampling_rate]

                features.append(processed_sample)
                labels.append(label)

        return np.array(features), np.array(labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )

    def __len__(self):
        return len(self.features)


def split_data(
    data_dict: dict,
    window_size: int,
    stride: int,
    downsampling_rate: int,
    target_person: str,
    data_split_approach: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
):
    train_dataset = []
    val_dataset = []
    test_dataset = []

    assert target_person in data_dict, f"Target person {target_person} not found."

    if data_split_approach == "PP":
        target_person_data = data_dict[target_person]
        total_len = target_person_data["data_len"]

        test_size = int(total_len * test_ratio)
        val_size = int(total_len * val_ratio)
        train_size = total_len - test_size - val_size

        train_dict = {
            k: v[:train_size] for k, v in target_person_data.items() if k != "data_len"
        }
        val_dict = {
            k: v[train_size : train_size + val_size]
            for k, v in target_person_data.items()
            if k != "data_len"
        }
        test_dict = {
            k: v[train_size + val_size :]
            for k, v in target_person_data.items()
            if k != "data_len"
        }

        train_dataset.append(
            StrikeWatchDataset(train_dict, window_size, downsampling_rate, stride)
        )
        val_dataset.append(
            StrikeWatchDataset(val_dict, window_size, downsampling_rate, stride)
        )
        test_dataset.append(
            StrikeWatchDataset(test_dict, window_size, downsampling_rate, stride)
        )

    elif data_split_approach == "LOPO":
        for person, person_data in data_dict.items():
            if person == target_person:
                continue

            total_len = person_data["data_len"]
            val_size = int(total_len * val_ratio)
            train_size = total_len - val_size

            train_dict = {
                k: v[:train_size] for k, v in person_data.items() if k != "data_len"
            }
            val_dict = {
                k: v[train_size:] for k, v in person_data.items() if k != "data_len"
            }

            train_dataset.append(
                StrikeWatchDataset(train_dict, window_size, downsampling_rate, stride)
            )
            val_dataset.append(
                StrikeWatchDataset(val_dict, window_size, downsampling_rate, stride)
            )

        target_person_data = data_dict[target_person]
        test_size = int(target_person_data["data_len"] * test_ratio)
        test_dict = {
            k: v[-test_size:] for k, v in target_person_data.items() if k != "data_len"
        }
        test_dataset.append(
            StrikeWatchDataset(test_dict, window_size, downsampling_rate, stride)
        )

    else:
        raise ValueError(f"Invalid data_split_approach: {data_split_approach}")

    return (
        ConcatDataset(train_dataset),
        ConcatDataset(val_dataset),
        ConcatDataset(test_dataset),
    )


def normalize_data(train_dataset: list, val_dataset: list, test_dataset: list):

    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_features, train_labels = next(iter(train_loader))
    val_features, val_labels = next(iter(val_loader))
    test_features, test_labels = next(iter(test_loader))

    train_features_np = train_features.numpy().reshape(len(train_features), -1)
    val_features_np = val_features.numpy().reshape(len(val_features), -1)
    test_features_np = test_features.numpy().reshape(len(test_features), -1)

    scaler = StandardScaler()
    scaler.fit(train_features_np)

    train_dataset_normalized = scaler.transform(train_features_np).reshape(
        train_features.shape
    )
    val_dataset_normalized = scaler.transform(val_features_np).reshape(
        val_features.shape
    )
    test_dataset_normalized = scaler.transform(test_features_np).reshape(
        test_features.shape
    )

    train_dataset_normed = [
        (torch.tensor(f, dtype=torch.float32), l)
        for f, l in zip(train_dataset_normalized, train_labels)
    ]
    val_dataset_normed = [
        (torch.tensor(f, dtype=torch.float32), l)
        for f, l in zip(val_dataset_normalized, val_labels)
    ]
    test_dataset_normed = [
        (torch.tensor(f, dtype=torch.float32), l)
        for f, l in zip(test_dataset_normalized, test_labels)
    ]

    return train_dataset_normed, val_dataset_normed, test_dataset_normed


def get_data(data_config: dict):

    target_person = data_config["target_person"]
    data_split_approach = data_config["data_split_approach"]

    data_dict = load_data(data_config)

    balanced_data_dict = balance_data(data_dict)
    train_dataset, val_dataset, test_dataset = split_data(
        balanced_data_dict,
        window_size=data_config["window_size"],
        stride=data_config["stride"],
        downsampling_rate=data_config["downsampling_rate"],
        target_person=target_person,
        data_split_approach=data_split_approach,
    )

    train_dataset_normed, val_dataset_normed, test_dataset_normed = normalize_data(
        train_dataset, val_dataset, test_dataset
    )

    return (
        train_dataset_normed,
        val_dataset_normed,
        test_dataset_normed,
    )
