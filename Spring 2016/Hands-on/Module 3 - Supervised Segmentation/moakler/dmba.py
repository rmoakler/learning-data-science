import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn import datasets
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import StringIO
import pydot
from IPython.display import Image, display

def single_feature_plot(X, Y, feature_name):
    plt.rcParams['figure.figsize'] = [20.0, 1.0]

    ax = plt.gca()

    color = ["red" if x == 0 else "blue" for x in Y]
    plt.scatter(X[feature_name], [1] * len(X[feature_name]), c=color, s=50)
    plt.xlabel(feature_name)
    plt.ylim([0.8, 1.2])

    ax.yaxis.set_visible(False)

    plt.show()

def entropy(target):
    # Get the number of users
    n = len(target)
    
    # Count how frequently each unique value occurs
    counts = np.bincount(target).astype(float)
    
    # Initialize entropy
    entropy = 0
    
    # If the split is perfect, return 0
    if len(counts) <= 1 or 0 in counts:
        return entropy
    
    # Otherwise, for each possible value, update entropy
    for count in counts:
        entropy += math.log(count/n, len(counts)) * count/n
    
    # Return entropy
    return -1 * entropy

def information_gain(X, Y, feature_name, threshold):
    # Dealing with numpy arrays makes this slightly easier
    target = np.array(Y)
    feature = np.array(X[feature_name])
    
    # Cut the feature vector on the threshold
    feature = (feature < threshold)
    
    # Initialize information gain with the parent entropy
    ig = entropy(target)
    
    # For both sides of the threshold, update information gain
    for level, count in zip([0, 1], np.bincount(feature).astype(float)):
        ig -= count/len(feature) * entropy(target[feature == level])
    
    # Return information gain
    return ig

def get_highest_ig(X, Y, feature_name):
    maximum_ig = 0
    maximum_ig_threshold = 0

    for current_threshold in X[feature_name]:
        current_ig = information_gain(X, Y, feature_name, threshold=current_threshold)
        
        if current_ig > maximum_ig:
            maximum_ig = current_ig
            maximum_ig_threshold = current_threshold

    return "The maximum IG of " + str(maximum_ig) + " was found when number_of_pets = " + str(maximum_ig_threshold)

def print_tree(tree):
    dot_data = StringIO.StringIO()
    export_graphviz(tree, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    display(Image(graph.create_png()))


def decision_surface(data, target, model, surface=True, probabilities=False, cell_size=.01):
    plt.rcParams['figure.figsize'] = [10.0, 10.0]

    # Get bounds
    x_min, x_max = data[data.columns[0]].min(), data[data.columns[0]].max()
    y_min, y_max = data[data.columns[1]].min(), data[data.columns[1]].max()
    
    # Create a mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_size), np.arange(y_min, y_max, cell_size))
    meshed_data = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    
    if model != None:
        # Predict on the mesh
        if probabilities:
            Z = model.predict_proba(meshed_data)[:, 1].reshape(xx.shape)
        else:
            Z = model.predict(meshed_data).reshape(xx.shape)
    
    # Plot mesh and data
    plt.title("Decision Surface")

    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    
    if surface and model != None:
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm_r, alpha=0.4)
    
    color = ["red" if t == 0 else "blue" for t in target]
    
    plt.scatter(data[data.columns[0]], data[data.columns[1]], color=color)
    plt.show()
