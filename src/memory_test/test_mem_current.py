from memory_profiler import profile
from data_repository import DataRepository
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

@profile(precision=8)
def check_model():
    data_repo = DataRepository("/home/mrcong/Code/nilm_as/src/.env")
    print("[+] Getting data:")
    X_Train, y_train, _, _, _, _ = data_repo.load_current_data(clean_data=True)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    print("[+] Training model")
    svm_fft = SVC(kernel='rbf', random_state=42, probability=True)
    svm_fft.fit(X_Train, y_train)
    rf_fft = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    rf_fft.fit(X_Train, y_train)
    xgboost_fft = XGBClassifier(objective='binary:logistic', tree_method="gpu_hist")
    xgboost_fft.fit(X_Train, y_train)
    
if __name__ == '__main__':
    check_model()