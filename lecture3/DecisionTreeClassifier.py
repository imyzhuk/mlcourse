import numpy as np
import pandas as pd
from collections import Counter

def calc_gini(ser):
    ser_len = ser.count()
    probabilies = ser.value_counts() / ser_len
    return 1 - np.sum(probabilies ** 2)

def calc_entropy(ser):
    ser_len = ser.count()
    probabilies = ser.value_counts() / ser_len
    
    return np.abs(np.sum(np.log2(probabilies) * probabilies))

def best_split(X, y, apply_criterion):    
    best_info_gain = 0
    best_threshold = None
    best_feat = None
    
    for col in X.columns:
        curr_feat = X[col]
        unique_thresholds = np.unique(curr_feat)
        feat_elemenets_count = curr_feat.count()
        curr_info_param = apply_criterion(y)
        
        if len(unique_thresholds) == 0:
            continue
        
        for threshold_index in range(len(unique_thresholds) - 1):
            threshold = (unique_thresholds[threshold_index + 1] - unique_thresholds[threshold_index]) / 2 + unique_thresholds[threshold_index]
            left_mask = curr_feat <= threshold
            right_mask = curr_feat > threshold
        
            left_group = y[left_mask]
            right_group = y[right_mask]
        
            left_info_param = apply_criterion(left_group)
            right_info_param = apply_criterion(right_group)
            probability_to_be_put_to_left_group = left_group.count() / feat_elemenets_count
            probability_to_be_put_to_right_group = right_group.count() / feat_elemenets_count
        
            weighted_info_param = probability_to_be_put_to_left_group * left_info_param \
                + probability_to_be_put_to_right_group * right_info_param
        
            info_gain = curr_info_param - weighted_info_param
        
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold
                best_feat = col
                
    return best_threshold, best_feat

def get_most_common_el(vect):
    return Counter(vect).most_common(1)[0][0]

class Node:
    def __init__(
        self, 
        value = None, 
        left = None,
        right = None,
        threshold = None,
        feature = None
    ):
        self.value = value
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature 

class MyDecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None):
        if max_depth is not None and not isinstance(max_depth, int):
            raise TypeError('max_depth must be int')
        if max_depth is not None and max_depth < 1:
            raise ValueError('max_depth must be more or equal to one (1)')
            
        self.criterion = criterion
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)
        return self
        
    def _split(self, X, y):
        if self.criterion == 'gini':
            apply_criterion = calc_gini
        else:
            apply_criterion = calc_entropy
            
        return best_split(X, y, apply_criterion)

    def _build_tree(self, X, y, depth):
        best_threshold, best_feat = self._split(X, y)

        if best_feat is None:
            return Node(value=get_most_common_el(y))
            
        if self.max_depth is not None and self.max_depth == depth:
            return Node(value=get_most_common_el(y))
            
        left_mask = X[best_feat] <= best_threshold
        right_mask = X[best_feat] > best_threshold

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(left=left_child, right=right_child, feature=best_feat, threshold=best_threshold)

    def _get_value_by_series(self, sample, node):
        if node.value is not None:
            return node.value
    
        sample_value_by_feature = sample[node.feature]
        
        if sample_value_by_feature <= node.threshold:
            return self._get_value_by_series(sample, node.left)
    
        return self._get_value_by_series(sample, node.right)

    def predict(self, X):
        return np.array(
            [self._get_value_by_series(sample, self.root) for sample in X.iloc]
        )