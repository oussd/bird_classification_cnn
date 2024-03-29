from pathlib import Path
from data_loader import prepare_dataframes, load_data
import pandas as pd

# Test prepare_dataframes function
dataset_path = '/home/student/bird_classification_cnn/data'  # Replace this with the actual path to your dataset
train_df, test_df = prepare_dataframes(dataset_path)
print("Train DataFrame:")
print(train_df.head())
print("\nTest DataFrame:")
print(test_df.head())

# Test load_data function
# Mock DataFrame for testing
train_filepaths = ['/home/student/bird_classification_cnn/data/train/ABBOTTS BABBLER/001.jpg', '/home/student/bird_classification_cnn/data/train/ABBOTTS BABBLER/002.jpg', '/home/student/bird_classification_cnn/data/train/ABBOTTS BABBLER/003.jpg']
train_labels = ['label1', 'label2', 'label3']
test_filepaths = ['/home/student/bird_classification_cnn/data/test/ABBOTTS BABBLER/1.jpg', '/home/student/bird_classification_cnn/data/testABBOTTS BABBLER//2.jpg', '/home/student/bird_classification_cnn/data/test/ABBOTTS BABBLER/3.jpg']
test_labels = ['label1', 'label2', 'label3']
train_df = pd.DataFrame({'Filepath': train_filepaths, 'Label': train_labels})
test_df = pd.DataFrame({'Filepath': test_filepaths, 'Label': test_labels})

# Call load_data function
train_data, test_data = load_data(train_df, test_df)

# Inspect the generated DirectoryIterator objects
print("Train Data:")
print(train_data)
print("\nTest Data:")
print(test_data)
