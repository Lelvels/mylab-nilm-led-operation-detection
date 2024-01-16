import pandas as pd
import numpy as np
from dotenv import load_dotenv
import glob
import os

class DataRepository:
    def __init__(self, env_path: str):
        self.env_path = env_path
        load_dotenv(self.env_path)
        self.ORIGINAL_CURRENT_DATA_PATH = os.getenv("ORIGINAL_CURRENT_DATA_PATH")
        self.ORIGINAL_FFT_DATA_PATH = os.getenv("ORIGINAL_FFT_DATA_PATH")
        self.UNCLEAN_DATA_FILE = os.getenv("UNCLEAN_DATA_REV_FILE")
        self.CLEAN_DATA_APPROACH1_FILE = os.getenv("CLEAN_DATA_APPROACH1_FILE")
    
    def load_original_current_data(self):
        # Initialize lists to store data
        amplitude_data = []
        label_data = []
        file_name_data = []
        # Iterate through each prefix
        for prefix in ['error', 'normal', 'zero', 'overcurrent', 'overheating']:
            pattern = f"{self.ORIGINAL_CURRENT_DATA_PATH}/{prefix}_*.csv"
            files = glob.glob(pattern)
            df = None
            # Read Amplitude column from each file
            for file in files:
                df = pd.read_csv(file)
                # Extract Amplitude column
                amplitude = df['Amplitude'].values.tolist()
                # Store data in lists
                amplitude_data.append(amplitude)
                label_data.append(prefix)
                file_name_data.append(os.path.basename(file))
        # Convert lists to numpy arrays if needed
        amplitude_array = np.array(amplitude_data)
        label_array = np.array(label_data)
        file_name_array = np.array(file_name_data)
        return amplitude_array, label_array, file_name_array
    
    def load_original_fft_data(self):
        # Initialize lists to store data
        amplitude_data = []
        label_data = []
        file_name_data = []
        frequency_array = []
        # Iterate through each prefix
        for prefix in ['error', 'normal', 'zero', 'overcurrent', 'overheating']:
            pattern = f"{self.ORIGINAL_FFT_DATA_PATH}/{prefix}_*.csv"
            files = glob.glob(pattern)
            df = None
            # Read Amplitude column from each file
            for file in files:
                df = pd.read_csv(file)
                # Extract Amplitude column
                amplitude = df['Amplitude'].values.tolist()
                # Store data in lists
                amplitude_data.append(amplitude)
                label_data.append(prefix)
                file_name_data.append(os.path.basename(file))
        # Convert lists to numpy arrays if needed
        amplitude_array = np.array(amplitude_data)
        label_array = np.array(label_data)
        file_name_array = np.array(file_name_data)
        frequency_array = np.array(df['Frequency'].values.tolist())
        return amplitude_array, label_array, file_name_array, frequency_array
    
    def read_current_data_with_label(self, file_path):
        # Extract label from the file name
        label = os.path.basename(file_path).split('_')[0]  # Assuming the file name format is 'label_data.csv'
        # Read CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Assuming the column containing current values is named 'Current'
        data = df['Current'].tolist()
        return label, data
    
    def import_current_data_from_file_list(self, file_list):
        data_arr, labels = [], []
        for file in file_list:
            label, data = self.read_current_data_with_label(file_path=f"{self.ORIGINAL_CURRENT_DATA_PATH}/{file}")
            labels.append(label)
            data_arr.append(data)
        labels = np.array(labels)
        data_arr = np.array(data_arr)
        return data_arr, labels
    
    def read_fft_data_with_label(self, file_path):
        # Extract label from the file name
        label = os.path.basename(file_path).split('_')[0]  # Assuming the file name format is 'label_data.csv'
        # Read CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Assuming the column containing current values is named 'Current'
        data = df['Amplitude'].tolist()
        return label, data
    
    def import_fft_data_from_file_list(self, file_list):
        data_arr, labels = [], []
        for file in file_list:
            label, data = self.read_fft_data_with_label(file_path=f"{self.ORIGINAL_FFT_DATA_PATH}/{file}")
            labels.append(label)
            data_arr.append(data)
        labels = np.array(labels)
        data_arr = np.array(data_arr)
        return data_arr, labels
    
    def load_unclean_file_names(self):
        #Read xlsx file
        train_files_df = pd.read_excel(self.UNCLEAN_DATA_FILE, sheet_name="train_dataset")
        validation_files_df = pd.read_excel(self.UNCLEAN_DATA_FILE, sheet_name="validation_dataset")
        test_files_df = pd.read_excel(self.UNCLEAN_DATA_FILE, sheet_name="test_dataset")
        #get the files names
        train_files = train_files_df['files'].values.tolist()
        validation_files = validation_files_df['files'].values.tolist()
        test_files = test_files_df['files'].values.tolist()
        return train_files, validation_files, test_files
    
    def load_current_data(self, clean_data: bool):
        X_train, y_train = [], []
        X_validation, y_validation = [], []
        X_test, y_test = [], []
        if(clean_data):
            file_name = self.CLEAN_DATA_APPROACH1_FILE
        else:
            file_name = self.UNCLEAN_DATA_FILE
        #Read xlsx file
        train_files_df = pd.read_excel(file_name, sheet_name="train_dataset")
        validation_files_df = pd.read_excel(file_name, sheet_name="validation_dataset")
        test_files_df = pd.read_excel(file_name, sheet_name="test_dataset")
        #get the files names
        train_files = train_files_df['files'].values.tolist()
        validation_files = validation_files_df['files'].values.tolist()
        test_files = test_files_df['files'].values.tolist()
        #get all the data
        X_train, y_train = self.import_current_data_from_file_list(train_files)
        X_validation, y_validation = self.import_current_data_from_file_list(validation_files)
        X_test, y_test = self.import_current_data_from_file_list(test_files)
        return X_train, y_train, X_validation, y_validation, X_test, y_test
        
    def load_fft_data(self, clean_data: bool):
        X_train, y_train = [], []
        X_validation, y_validation = [], []
        X_test, y_test = [], []
        if(clean_data):
            file_name = self.CLEAN_DATA_APPROACH1_FILE
        else:
            file_name = self.UNCLEAN_DATA_FILE
        #Read xlsx file
        train_files_df = pd.read_excel(file_name, sheet_name="train_dataset")
        validation_files_df = pd.read_excel(file_name, sheet_name="validation_dataset")
        test_files_df = pd.read_excel(file_name, sheet_name="test_dataset")
        #get the files names
        train_files = train_files_df['files'].values.tolist()
        validation_files = validation_files_df['files'].values.tolist()
        test_files = test_files_df['files'].values.tolist()
        #get all the data
        X_train, y_train = self.import_fft_data_from_file_list(train_files)
        X_validation, y_validation = self.import_fft_data_from_file_list(validation_files)
        X_test, y_test = self.import_fft_data_from_file_list(test_files)    
        return X_train, y_train, X_validation, y_validation, X_test, y_test
    
    def count_labels(self, labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        return label_counts
