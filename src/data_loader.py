from typing import Tuple
from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple


def prepare_dataframes(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares dataframes for the training and validation datasets by walking through each directory.

    Args:
        dataset (str): Path to the dataset directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataframes for the training and validation datasets.
    """

    image_dir = Path(f'./{dataset}')
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)

    labels = pd.Series(labels, name='Label')
    image_df = pd.concat([filepaths, labels], axis=1)

    train_df, test_df = train_test_split(image_df,
                                         test_size=0.2,
                                         shuffle=True,
                                         random_state=42)

    return train_df, test_df


def load_data(train_df: pd.DataFrame,
              test_df: pd.DataFrame,
              img_height: int = 224,
              img_width: int = 224,
              batch_size: int = 32):
    """
    Loads and preprocesses the dataset from dataframes using PyTorch.

    Args:
        train_df (pd.DataFrame): DataFrame containing filepaths and labels for the training data.
        test_df (pd.DataFrame): DataFrame containing filepaths and labels for the test data.
        img_height (int): Height of the input images. Default is 224.
        img_width (int): Width of the input images. Default is 224.
        batch_size (int): Size of the batches of data. Default is 32.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test data loaders.
    """
    # Define transformations for the training data
    train_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Define transformations for the test data
    test_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create a custom dataset for the training and test data
    train_dataset = datasets.ImageFolder(os.path.dirname(train_df['Filepath'].iloc[0]).rsplit('/', 1)[0], transform=train_transforms)
    test_dataset = datasets.ImageFolder(os.path.dirname(test_df['Filepath'].iloc[0]).rsplit('/', 1)[0], transform=test_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader