# ==========================
# TRAIN AND TESTING SAMPLES
# ==========================

#     1. One Hot Encoding
#     2. Train and Testing sample split

# =================
# One Hot Encoding:
# =================

import ED
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import missing_adhoc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
import warnings

pd.set_option("display.width", 1100)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.float_format", lambda x: "%.0f" %x)
warnings.filterwarnings("ignore")


def One_Hot_Encoding_Func_Stat(dataframe):
    
    '''One Hot Encoding Function'''
    
    encoded_dataframes = []

    for col in dataframe.columns:

        y = pd.get_dummies(dataframe[col])
        n = len(pd.unique(dataframe[col])) # count unique values per column
        dataframe_ = y.drop(y.columns[n-1], axis=1) # drop last column
        encoded_dataframes.append(dataframe_)

    df_loan_categorical_encoded = pd.concat(encoded_dataframes, axis = 1)
    
    return df_loan_categorical_encoded

#df_loan_categorical_encoded = One_Hot_Encoding_Func_Stat(missing_adhoc.df_loan_categorical_mode)


# ==================================
# Machine Learning One Hot Encoding
# ==================================

def One_Hot_Encoding_Func_Machi(dataframe):
    
    '''One Hot Encoding Function'''
    
    encoded_dataframes = []

    for col in dataframe.columns:
        y = pd.get_dummies(dataframe[col])
        encoded_dataframes.append(y)

    df_loan_categorical_encoded = pd.concat(encoded_dataframes, axis = 1)
    
    return df_loan_categorical_encoded

df_loan_categorical_encoded = One_Hot_Encoding_Func_Machi(missing_adhoc.df_loan_categorical_mode)


# ================================================
# Sample partitioning into train and testing sets
# ================================================

# =========================================================
# Defining Independent and Dependent variables - Statistics
# =========================================================

def Create_X_Y(dataframe_float, dataframe_categorical, target):
    
    df_default_lgd_total_partition = pd.concat([dataframe_float, dataframe_categorical], axis = 1)
    X= df_default_lgd_total_partition.drop(labels=[target.name], axis=1)
    Y = df_default_lgd_total_partition[target.name]
    
    return X, Y

# X, Y = Create_X_Y(dataframe_float = ED.df_loan_float, dataframe_categorical = df_loan_categorical_encoded\
#               , target = ED.df_loan_float["GB"])

# ============================================
# Defining Independent and Dependent variables
# ============================================

# ===============================
# Sample imbalance investigation
# ===============================

def Sample_Imbalance(dataframe_total):
    
    X, Y = Create_X_Y(dataframe_float = ED.df_loan_float, dataframe_categorical = df_loan_categorical_encoded\
              , target = ED.df_loan_float["GB"])

    plt.hist(Y, weights = np.ones(len(Y))/len(Y))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
    return 

#Imbalance = Sample_Imbalance(dataframe_total=ED.df_loan_float["GB"])

# ============================
# Training and Testing samples
# ============================

def Split_Xtrain_Ytrain(dataframe_float, dataframe_categorical,target, testsize, randomstate):
    
    X, Y = Create_X_Y(dataframe_float = ED.df_loan_float, dataframe_categorical = df_loan_categorical_encoded\
              , target = ED.df_loan_float["GB"])

    X_train_lgd, X_test_lgd, Y_train_lgd, Y_test_lgd = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    return X_train_lgd, X_test_lgd, Y_train_lgd, Y_test_lgd

X_train, X_test, Y_train, Y_test = Split_Xtrain_Ytrain(ED.df_loan_float, df_loan_categorical_encoded, ED.df_loan_float["GB"]\
                                                            , testsize=0.33, randomstate=42)

X_train = X_train.drop(labels=["_freq_"], axis=1) # temp, for mach it has to be dropped

X_test = X_test.drop(labels=["_freq_"], axis=1) # temp
