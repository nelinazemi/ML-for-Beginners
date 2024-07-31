import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label
        self.children = {}

    def __repr__(self):
        if self.feature is not None:
            return f'DecisionNode(feature="{self.feature}, children={self.children}")'
        else:
            return f'LeafNode(label="{self.label}")'


def entropy(labels):
    probs = labels.value_counts() / len(labels)
    return sum(-probs * np.log2(probs))


def information_gain(data, feature, target):
    # entropy of parent
    entropy_parent = entropy(data[target])

    # entropy of children
    entropy_children = 0
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weight = len(subset) / len(data)
        entropy_children += weight * entropy(subset[target])

    return entropy_parent - entropy_children


def make_tree(data, target):
    # leaf node?
    if len(data[target].unique()) == 1:
        return Node(label=data[target].iloc[0])

    # find feature names
    features = data.drop(target, axis=1).columns

    # find the best feature using greedy search and information gain
    max_gain_idx = np.argmax([information_gain(data, feature, target) for feature in features])
    best_feature = features[max_gain_idx]

    # create a decision node
    node = Node(feature=best_feature)

    # find unique values in the best feature
    unique_values = data[best_feature].unique()

    # loop over the unique values of the best feature to create sub-trees
    for value in unique_values:
        subset = data[data[best_feature] == value].drop(best_feature, axis=1)
        # display(subset)
        # create a sub-tree
        node.children[value] = make_tree(subset, target)
    return node


if __name__ == '__main__':
    df = pd.read_csv('data/baseball.csv')
    df.columns = ['outlook', 'temperature', 'humidity', 'wind', 'play']
    decision_tree = make_tree(df, 'play')
