
import streamlit as st 
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import norm
import pylab
import statsmodels.stats.diagnostic as sd
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import scipy
from scipy import stats
from math import *
import pickle

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from class_diagnostics import (ResidualsPlot, BreushPaganTest, NormalityTest, DurbinWatsonTest,
                               PartialPlots, LevStudQuaRes, CooksDisQuantRes)
from class_modelperf import ModelPerfomance

with open('glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

file_path = "KGB.sas7bdat"
data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
miss = ImputationCat(df_loan_categorical)
imputer_cat = miss.simple_imputer_mode()

custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

instance = OneHotEncoding(custom_rcParams, imputer_cat, "statistics")
x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]
x_test = sm.add_constant(x_test.values)
y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
threshold = 0.47
randomstate = 42
ccpalpha = 0

def settings():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    pd.set_option("display.width", 3000)
    pd.set_option("display.max_columns", 3000)
    pd.set_option("display.max_rows", 3000)
    pd.set_option("display.float_format", lambda x: "%.0f" %x)
    warnings.filterwarnings("ignore")
