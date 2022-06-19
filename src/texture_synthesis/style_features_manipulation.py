import torch
import torch.nn as nn
from sklearn import svm
import numpy as np

def style_attribute_extraction_svm(style_tensor_set1: list, style_tensor_set2: list, C=1):
    assert C > 0
    X = torch.stack(style_tensor_set1 + style_tensor_set2).cpu().numpy()
    y = [0] * len(style_tensor_set1) + [1] * len(style_tensor_set2)
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X, y)
    hyperplane_normal = torch.tensor(np.copy(clf.coef_[0]), dtype=torch.float32)
    style_vector = nn.functional.normalize(hyperplane_normal, dim=0) * 100
    return style_vector, clf
        
def style_attribute_extraction_means(style_tensor_set1: list, style_tensor_set2: list):
    mean1 = torch.mean(torch.stack(style_tensor_set1), 0)
    mean2 = torch.mean(torch.stack(style_tensor_set2), 0)
    style_diff = mean2 - mean1
    style_vector = nn.functional.normalize(style_diff, dim=0) * 100
    return style_vector, None
