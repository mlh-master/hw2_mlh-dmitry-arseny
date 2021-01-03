import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import random

# matplotlib inline

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 12

random.seed(1)

X = pd.read_csv("HW2_data.csv")
# print(X.sample(8, random_state=5))

X_train, X_test = train_test_split(X, test_size = 0.20, random_state=1)
logreg = LogisticRegression(solver='saga', multi_class='ovr', penalty='none', max_iter=10000)

print(X_train)

