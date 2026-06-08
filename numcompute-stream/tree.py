"""
Decision tree classifier with streaming support. Supports Gini impurity and entropy split criteria.
"""
from __future__ import annotations
import numpy as np

#internal node structure

class _Node:
    """A single node in the decision tree.
 
    Attributes:
    feature_index : int or None
        Index of the feature used for splitting (None for leaf nodes).
    threshold : float or None
        Split threshold value (None for leaf nodes).
    left : _Node or None
        Left child (feature value <= threshold).
    right : _Node or None
        Right child (feature value > threshold).
    value : object or None
        Predicted class label (leaf nodes only).
    is_leaf : bool
        Whether this node is a terminal leaf.
    depth : int
        Depth of this node in the tree.
    """
 
    __slots__ = ("feature_index", "threshold", "left", "right",
                 "value", "is_leaf", "depth")
 
    def __init__(self, depth: int = 0):
        self.depth = depth
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False
 
 

