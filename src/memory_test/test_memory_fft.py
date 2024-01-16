from memory_profiler import profile
from data_repository import DataRepository
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def find_top_k_indices(amplitudes, k):
    # Get the indices of the top 50 elements
    top_k_indices = np.argsort(amplitudes)[-k:][::-1]
    highest_ampls = amplitudes[top_k_indices]
    return highest_ampls

def get_x_by_top_ampls(k, ampls):
    X = []
    for ampl in ampls:
        X.append(find_top_k_indices(amplitudes=ampl, k=k))
    return np.array(X)

def find_top_k_indices(amplitudes, k):
    # Get the indices of the top 50 elements
    top_k_indices = np.argsort(amplitudes)[-k:][::-1]
    highest_ampls = amplitudes[top_k_indices]
    return highest_ampls

def get_x_by_top_ampls(k, ampls):
    X = []
    for ampl in ampls:
        X.append(find_top_k_indices(amplitudes=ampl, k=k))
    return np.array(X)

@profile(precision=8)
def check_model():
    data_repo = DataRepository("/home/mrcong/Code/nilm_as/src/.env")
    print("[+] Getting data:")
    train_ampls, y_train, validation_ampls, y_validation, test_ampls, y_test = data_repo.load_fft_data(clean_data=True)
    # Chuyển đổi danh sách labels thành mã số
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    print("[+] Getting only top features:")
    X_train = get_x_by_top_ampls(k=1, ampls=train_ampls)
    print("[+] Training model")
    svm_fft = SVC(kernel='rbf', random_state=42, probability=True)
    svm_fft.fit(X_train, y_train)
    rf_fft = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    rf_fft.fit(X_train, y_train)
    xgboost_fft = XGBClassifier(objective='binary:logistic', tree_method="gpu_hist")
    xgboost_fft.fit(X_train, y_train)
    

if __name__ == '__main__':
    check_model()