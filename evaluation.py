import numpy as np

#import sklean libraries
import sklearn

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import accuracy_score

def acuracycheckl(true_value , predicted_value) :
    print('Accuracy score',accuracy_score(true_value , predicted_value))
    print('Confusion matrix',confusion_matrix(true_value , predicted_value))
    print(precision_score(true_value, predicted_value))
    print(recall_score(true_value, predicted_value))
    print(f1_score(true_value,predicted_value))