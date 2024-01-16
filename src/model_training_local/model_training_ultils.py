import inspect
import numpy
import sys
import math
import types

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, auc, roc_curve, confusion_matrix, precision_score, recall_score, f1_score

#Refactor from: https://github.com/jpmml/sklearn2pmml/blob/master/sklearn2pmml/util/__init__.py
class ModelMemCalculation():
    def __init__(self) -> None:
        pass
    
    def fqn(self, obj):
        clazz = obj if inspect.isclass(obj) else obj.__class__
        return ".".join([clazz.__module__, clazz.__name__])
    
    def is_instance_attr(self, obj, name):
        if not hasattr(obj, name):
            return False
        if name.startswith("__") and name.endswith("__"):
            return False
        v = getattr(obj, name)
        if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType, types.FunctionType, types.MethodType)):
            return False
        # See https://stackoverflow.com/a/17735709/
        attr_type = getattr(type(obj), name, None)
        if isinstance(attr_type, property):
            return False
        return True
    
    def get_instance_attrs(self, obj):
        names = dir(obj)
        names = [name for name in names if self.is_instance_attr(obj, name)]
        return names

    def sizeof(self, obj, with_overhead = False):
        if with_overhead:
            return sys.getsizeof(obj)
        return obj.__sizeof__()
    
    def bytes_to_mb(self, value):
        return value*math.pow(10, -6)
    
    def bytes_to_kb(self, value):
        return value*math.pow(10, -3)

    def deep_sizeof(self, obj, with_overhead = False, verbose = False):
        # Primitive type values
        if obj is None:
            return obj.__sizeof__()
        elif isinstance(obj, (int, float, str, bool, numpy.int64, numpy.float32, numpy.float64)):
            return obj.__sizeof__()
        # Iterables
        elif isinstance(obj, list):
            sum = self.sizeof([], with_overhead = with_overhead) # Empty list
            for v in obj:
                v_sizeof = self.deep_sizeof(v, with_overhead = with_overhead, verbose = False)
                sum += v_sizeof
            return sum
        elif isinstance(obj, tuple):
            sum = self.sizeof((), with_overhead = with_overhead) # Empty tuple
            for i, v in enumerate(obj):
                v_sizeof = self.deep_sizeof(v, with_overhead = with_overhead, verbose = False)
                sum += v_sizeof
            return sum
        # Numpy ndarrays
        elif isinstance(obj, numpy.ndarray):
            sum = self.sizeof(obj, with_overhead = with_overhead) # Array header
            sum += (obj.size * obj.itemsize) # Array content
            return sum
        # Reference type values
        else:
            qualname = self.fqn(obj)
            # Restrict the circle of competence to Scikit-Learn classes
            # if not (qualname.startswith("_abc.") or qualname.startswith("sklearn.")):
                # raise TypeError("The object (class {0}) is not supported ".format(qualname))
            sum = self.sizeof(object(), with_overhead = with_overhead) # Empty object
            names = self.get_instance_attrs(obj)
            if names:
                if verbose:
                    print("| Attribute | `type(v)` | `deep_sizeof(v)` |")
                    print("|---|---|---|")
                for name in names:
                    v = getattr(obj, name)
                    v_type = type(v)
                    v_sizeof = self.deep_sizeof(v, with_overhead = with_overhead, verbose = False)
                    sum += v_sizeof
                    if verbose:
                        print("| {} | {} | {} |".format(name, v_type, v_sizeof))
            return sum
    
    def current_mem_usage(self):
        ''' Memory usage in kB '''
        with open('/proc/self/status') as f:
            memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
        print("Memory:", np.round(float(memusage.strip())), "kB")
        
class ModelEvaluationUltis():
    def __init__(self) -> None:
        pass
    
    def evaluate_and_print_results(self, y_pred, y_pred_proba, y_test, label_encoder: LabelEncoder):
        # Tính toán độ chính xác
        accuracy = accuracy_score(y_test, y_pred)
        #conf_matrix to return
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Tính toán F1-Score
        f1_macro = f1_score(y_test, y_pred, average='macro')
        #Precision and recall
        precision = precision_score(y_test, y_pred, average='macro')  
        recall = recall_score(y_test, y_pred, average='macro')
        # Tính AUC cho từng lớp và tính trung bình (macro-average) AUC
        n_classes = len(np.unique(y_test))  # Assuming labels start from 0
        auc_values = []
        for i in range(n_classes):
            # Create binary labels for each class
            y_test_class = np.where(y_test == i, 1, 0)
            auc = roc_auc_score(y_test_class, y_pred_proba[:, i])
            auc_values.append(auc)
        # Tính trung bình AUC (macro-average)
        macro_avg_auc = np.mean(auc_values)  # Avoid division by zero if auc_values is empty
        #Revert transform
        y_test = label_encoder.inverse_transform(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)
        # Print results
        print(classification_report(y_test, y_pred))
        print()
        print("Custom reports")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print(f"F1-Score (Macro): {f1_macro}")
        print(f"Macro-average AUC (SVM): {macro_avg_auc}")
        return conf_matrix

    def plot_auc_curve(self, y_pred_proba, y_test, label_encoder: LabelEncoder):
        sns.set(font_scale=1.5)
        # Tính AUC cho từng lớp và tính trung bình (macro-average) AUC
        n_classes = len(np.unique(y_test))  # Assuming labels start from 0
        # Plot ROC curve for each class
        plt.figure(figsize=(8, 8))
        for i in range(n_classes):
            class_name = label_encoder.inverse_transform([i])[0]
            y_test_class = np.where(y_test == i, 1, 0)
            fpr, tpr, _ = roc_curve(y_test_class, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

    def plot_conf_matrix(self, conf_matrix):
        sns.set(font_scale=2)
        # Define the class labels
        class_labels = ['error', 'normal', 'overcurrent', 'overheating', 'zero']
        # Create a dataframe from the confusion matrix
        df_cm = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
        # Set the plot size
        plt.figure(figsize=(10, 7))
        # Plot the heatmap
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g', annot_kws={"size": 25})
        # Set plot title and labels
        plt.xlabel('Predicted label', labelpad=15)
        plt.ylabel('True label', labelpad=15)
        plt.show()
        
    def evaluate_models(self, model, X_test, y_test):
        # Dự đoán nhãn cho tập test
        y_pred = model.predict(X_test)
        # Tính toán độ chính xác
        accuracy = accuracy_score(y_test, y_pred)
        # Tính toán F1-Score
        f1_macro = f1_score(y_test, y_pred, average='macro')
        #Precision and recall
        precision = precision_score(y_test, y_pred, average='macro')  
        recall = recall_score(y_test, y_pred, average='macro')
        # Dự đoán scores để tính AUC
        y_test_scores = model.predict_proba(X_test)
        # Tính AUC cho từng lớp và tính trung bình (macro-average) AUC
        n_classes = len(np.unique(y_test))  # Assuming labels start from 0
        auc_values = []
        for i in range(n_classes):
            # Create binary labels for each class
            y_test_class = np.where(y_test == i, 1, 0)
            auc = roc_auc_score(y_test_class, y_test_scores[:, i])
            auc_values.append(auc)
        # Tính trung bình AUC (macro-average)
        macro_avg_auc = np.mean(auc_values) if len(auc_values) > 0 else 0  # Avoid division by zero if auc_values is empty
        return precision, recall, accuracy, f1_macro, macro_avg_auc