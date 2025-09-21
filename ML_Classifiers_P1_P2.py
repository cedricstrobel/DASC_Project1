# Import Librarys
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.preprocessing import LabelEncoder

# X is features y is prediction (This for me to remember)
# Set pathing for base and validation sets
pa_path = r"C:\Users\andre\OneDrive - University of Arkansas\Desktop\Andrew_Branch_ML\project_adult.csv"
pvi_path = r"C:\Users\andre\OneDrive - University of Arkansas\Desktop\Andrew_Branch_ML\project_validation_inputs.csv"

# Read the files in
pa = pd.read_csv(pa_path)
pvi = pd.read_csv(pvi_path)


def preprocess_data(df):
    # Drop unnamed col, don't need
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # Handle missing values (simple drop, can be improved later)
    df = df.dropna()

    # Separate categorical and numeric columns
    cat_cols = df.select_dtypes(include=['object']).columns
    income_col = df['income'].copy()
    cat_cols = cat_cols.drop('income')

    num_cols = df.select_dtypes(exclude=['object']).columns


    # Incode income as ordinal since it is, all other categorical cols are not ordinal they are nominal
    income_col = income_col.map({'<=50K': 0, '>50K': 1})


    # Use of OneHotEncoder over LabelEncoder is since most of the Categorical cols are not ordinal there nominal so it wont make since to use label encoder
    # OneHotEncoder will help in this by breaking out the num of unique values found in the categorical col and break those out into multiple cols
    # Example: the Class col may have Private, Local-gov, or Never-worked values in it. These will get broken out into 3 cols for each unique value for that observation and if it was local-gov for that obervation it will be marked with a 1 and the other two are 0
    # Deciding to keep ? values in cols for workclass, occupation, and native-country since there could be some important pattern info that can come from it
    # the ? values will get encoded, can always fix later if issue

    # Apply OneHotEncoder
    ohe = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')
    encoded = ohe.fit_transform(df[cat_cols])

    # Convert encoded array back to DataFrame
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols), index=df.index)

    # Encode numerical values
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Concatenate numeric and encoded categorical features
    df = pd.concat([df[num_cols], encoded_df, income_col.rename('income')], axis=1)

    X = df.drop(columns=['income'], axis=1)
    y = df[['income']]


    return X.values, y.values.ravel()

def preprocess_data_validation(df):
    # Drop unnamed col, don't need
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # Handle missing values (simple drop, can be improved later)
    df = df.dropna()

    # Separate categorical and numeric columns
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(exclude=['object']).columns


    # Use of OneHotEncoder over LabelEncoder is since most of the Categorical cols are not ordinal there nominal so it wont make since to use label encoder
    # OneHotEncoder will help in this by breaking out the num of unique values found in the categorical col and break those out into multiple cols
    # Example: the Class col may have Private, Local-gov, or Never-worked values in it. These will get broken out into 3 cols for each unique value for that observation and if it was local-gov for that obervation it will be marked with a 1 and the other two are 0
    # Deciding to keep ? values in cols for workclass, occupation, and native-country since there could be some important pattern info that can come from it
    # the ? values will get encoded, can always fix later if issue

    # Apply OneHotEncoder
    ohe = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')
    encoded = ohe.fit_transform(df[cat_cols])

    # Convert encoded array back to DataFrame
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols), index=df.index)

    # Encode numerical values
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Concatenate numeric and encoded categorical features
    df = pd.concat([df[num_cols], encoded_df], axis=1)






    return df.values



# Apply preprocessing
X, y = preprocess_data(pa)
X_validation = preprocess_data_validation(pvi)

#%%

class Perceptron:
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        # if you use libarry versions stated by book use float_, else if new use float64
        # self.b_ = np.float_(0.)
        self.b_ = np.float64(0.)

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

#%%
ppn = Perceptron(eta=0.01, n_iter=500, random_state=1)
ppn.fit(X, y)

# Plot miscalssificaiotn error for each epoch
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()