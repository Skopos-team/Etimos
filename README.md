# Etimos

"Ready" - Machine Learning Visualization Library 

## Overview

The objective is to automatize the concept of data visualization. With few methods and plots all the most important pattern in the data can be discovered. The user has to pass the target_label and the objectuve of the task (classification, regression). Each method takes as input the dataframe and some parameters to better visualize the plots.

### Installation and requirements

The requirements are:

- python
- matplotlib
- seaborn

Then you can clone the repository or install it from pip.

	$ pip install etimos 

## Usage - From Jupyter

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from etimos.visualizer import DataExplorer

# Example of Classification Dataset
from sklearn.datasets import load_breast_cancer

LABEL_TAG = "target"
PROBLEM = "classification"

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
y = np.reshape(y, (-1, 1))
data = np.concatenate((X, y), axis=1)

df = pd.DataFrame(data)
df.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", \
    "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "target"]

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the Object to explore the data
explorer = DataExplorer(label_tag=LABEL_TAG, problem=PROBLEM)

# First Data Exploration
explorer.first_exploration(df)

# Pearson Correlation Heatmap
explorer.features_correlation(df)

# Pair Plots
explorer.pair_plots(df)

# Plot the distribution of each feature
explorer.plot_features_distribution(df)

# PLot Box for each feature
explorer.plot_features_box_plot(df)

# Check if Train and Test are equally distirbuted
explorer.plot_train_test_distribution(X_train, X_test)

```