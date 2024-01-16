from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict

class MyConfidentLearning:    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.pred_probs = None
        self.thresholds = None
        
    def get_out_of_sample_proba(self) -> np.ndarray:
        model = XGBClassifier(tree_method="gpu_hist", enable_categorical=True)
        print("[+] Getting out of sample probality")
        self.pred_probs = cross_val_predict(model, self.X, self.y, method='predict_proba', cv=5)
        print(f"[-] Finished getting out of sample probality with shape: {self.pred_probs.shape}")
        return self.pred_probs
        
    def compute_class_thresholds(self) -> np.ndarray:
        print("[+] Computing thresholds")
        n_examples, n_classes = self.pred_probs.shape
        self.thresholds = np.zeros(n_classes)
        for k in range(n_classes):
            count = 0
            p_sum = 0
            for i in range(n_examples):
                if self.y[i] == k: #this explain the p^(y~=j;x,theta), the noisy label is equal class k
                    count += 1
                    p_sum += self.pred_probs[i, k]
            self.thresholds[k] = p_sum / count
        print(f"[-] Finished compute thresholds: {self.thresholds}")
        return self.thresholds
    
    def compute_confident_joint(self) -> np.ndarray:
        print("[+] Computing confident joint")
        n_examples, n_classes = self.pred_probs.shape
        confident_joint = np.zeros((n_classes, n_classes), dtype=np.int64)
        for data_idx in range(n_examples):
            i = self.y[data_idx]    #y_noise
            j = None                #y_true -> to find
            #Lưu ý điểm mình bị sai: vị trí của chúng không ứng với label
            p_j = -1
            for candidate_j in range(n_classes):
                p = self.pred_probs[data_idx, candidate_j]
                if p >= self.thresholds[candidate_j] and p > p_j:
                    j = candidate_j
                    p_j = p
            if j is not None:
                confident_joint[i][j] += 1
        print("[-] Finished compute confident joint:")
        print(confident_joint)
        return confident_joint
    
    # def filter_label_issues_by_self_confidence(self, pred_probs: np.ndarray, num_label_issues: np.int64):
    #     self_confidence = []
    #     for i in range(pred_probs.shape[0]):
    #         self_confidence.append(pred_probs[i, self.y[i]])
    #     self_confidence = np.array(self_confidence)
    #     ranked_indices = np.argsort(self_confidence)
    #     issue_idx = ranked_indices[:num_label_issues]
    #     return issue_idx
    
    def find_label_issues(self):
        """
        This is the implementation from approach 1 - method 2, where you estimates the errors from 
        the confident joint. However, the author didn't mention the proper way to prune the data. 
        Args:
            pred_probs (np.ndarray): _description_
            num_label_issues (np.int64): _description_

        Returns:
            issue_indices: 
        """
        print("[+] Finding labels issue indeces:")
        n_examples, n_classes = self.pred_probs.shape
        issue_indices = []
        for data_idx in range(n_examples):
            i = self.y[data_idx]    #y_noise
            j = None                #y_true -> to find
            p_j = -1
            for candidate_j in range(n_classes):
                p = self.pred_probs[data_idx, candidate_j]
                if p >= self.thresholds[candidate_j] and p > p_j:
                    j = candidate_j
                    p_j = p
            if j is not None:
                if i != j:
                    issue_indices.append(data_idx)
        print(f"Issue indices: {len(issue_indices)}")
        return issue_indices
        
    def find_label_errors(self):
        issue_index = None
        self.get_out_of_sample_proba()
        self.compute_class_thresholds()
        C = self.compute_confident_joint()
        num_label_issues = np.sum(C - np.diag(np.diag(C)))
        print(f"[+] Number of label issues: {num_label_issues}")
        print('[+] Estimated noise rate: {:.1f}%'.format(100*num_label_issues / self.pred_probs.shape[0]))
        issue_index = self.find_label_issues()
        return issue_index
        