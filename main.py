import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from etimos.visualizer import DataExplorer

# Example of Classification Dataset
from sklearn.datasets import load_breast_cancer

# Example of Regression Dataset
from sklearn.datasets import load_boston

def classification():

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

	explorer = DataExplorer(label_tag=LABEL_TAG, problem=PROBLEM)

	return explorer, df, X, y


def regression():

	LABEL_TAG = "target"
	PROBLEM = "regression"

	boston = load_boston()
	X = boston.data
	print(X.shape)
	y = boston.target
	y = np.reshape(y, (-1, 1))
	data = np.concatenate((X, y), axis=1)

	df = pd.DataFrame(data)
	df.columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
		"10", "11", "12", "target"]

	explorer = DataExplorer(label_tag=LABEL_TAG, problem=PROBLEM)

	return explorer, df, X, y

def main():

	# explorer, df = classification()
	explorer, df, X, y = regression()

	# Split train and test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

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


	return

main()