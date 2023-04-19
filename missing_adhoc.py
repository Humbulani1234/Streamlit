
# ==============================
# MCAR adhoc tests vs MNAR, MAR
# ==============================

# ======
# Plots
# ======

import ED
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import logging

# create logger

logger = logging.getLogger('no_spam')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#add formatter to ch

ch.setFormatter(formatter)

# add ch to logger

logger.addHandler(ch)

# ===============================Starting=====================================================

logger.info('Starting')

def Categorical_missingness_Crosstab_Plot(independent, target):
    
    '''Plot cross tab'''
    
    missingness = independent.isnull()
    cross_tab = pd.crosstab(target, missingness, normalize="columns", dropna=True).apply(lambda r: round(r,2), axis=1)

    ax = cross_tab.plot(kind='bar', width=0.15, ylabel="Number Absorbed",color=["#003A5D","#A19958"]\
   ,edgecolor="tab:grey",linewidth=1.5)

    l = {"Not-Absorbed":"#003A5D", "Absorbed":"#A19958"}
    labels = list(l.keys())
    handles = [plt.Rectangle((5,5),10,10, color=l[label]) for label in labels]
    plt.legend(handles, labels, fontsize=7, bbox_to_anchor=(1.13,1.17), loc="upper left", title="legend",shadow=True)

    plt.title("Number Absorbed for each Gender", fontsize=9, pad=12)
    plt.xlabel("Gender",fontsize=7.5)
    plt.xticks(fontsize=7.5)
    plt.ylabel('Number Absorbed', fontsize = 7.5)
    plt.yticks(fontsize=7.5)
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)
        
    for c in ax.containers:
        ax.bar_label(c, label_type='edge', fontsize=7)
     
    return cross_tab

#logger.debug('Categorical_missingness_Crosstab_Plot: {}, {}={}'.format(ED.df_loan_categorical["RESID"].tolist(), ED.df_loan_float["GB"].tolist()\
    #, answer))

def Categorical_missingness_Pivot_Plot(independent, target):
      
    '''Categorical Plot for greater than 2 categories'''
    
    missingness = independent.isnull()
    df = pd.concat([missingness, target], axis=1) 
    df_pivot = pd.pivot_table(df, index=independent.name, values=target.name, aggfunc=len, fill_value=0)\
                                          #.apply(lambda x: x/float(x.sum()))

    d = df_pivot.plot(kind="bar", width=0.1, color=["#003A5D","#A19958"], fontsize=7.5\
                         , edgecolor="tab:grey",linewidth=1.5)

    d.legend(title="legend", bbox_to_anchor=(1, 1.02), loc='upper left', fontsize=6.5, shadow=True)
    
    plt.title("Race and Absorption for Gender", fontsize=7.5, pad=12)
    plt.xlabel('Absorbed', fontsize=7)
    plt.xticks(fontsize=7)
    plt.ylabel('Number Absorbed', fontsize = 7)
    plt.yticks(fontsize=7)
    plt.xlabel(" ")
    plt.rcParams["figure.figsize"] = (2.7,2.5)
    plt.rcParams["legend.title_fontsize"] = 7

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)


    return df_pivot

# ======
# Tests
# ======

def Chi_Square_Missingness_Categorical_Test(independent, target):
    
    '''Missing variables Test - Adhoc'''
    
    missingness = independent.isnull()
    h_chi = pd.crosstab(missingness, target)
    chi_val, p_val, dof, expected = chi2_contingency(h_chi)
    
    return chi_val, p_val

# ===============================================================================================================================

# ==========================================
# Simple Imputation -- through Python API's
# ==========================================

def Simple_Imputer_mode(dataframe):
    
    df_loan_categorical_mode = dataframe.copy(True)
    mode_imputer = SimpleImputer(strategy="most_frequent")
    df_loan_categorical_mode.iloc[:,:] = mode_imputer.fit_transform(df_loan_categorical_mode)

    return df_loan_categorical_mode

df_loan_categorical_mode = Simple_Imputer_mode(dataframe=ED.df_loan_categorical)

# ================
# Ordinal Encoding
# ================

def Ordinal_Encode_with_NAN(independent_series, dataframe): # for one column, then procedural
    
    '''Ordinal Encoding with missing values'''
    
    y = OrdinalEncoder() # instatiate ordinal encoder class
    name = independent_series # pass in the independent series for a missing column, (name = name of column)
    name_not_null = independent_series[independent_series.notnull()] # removes null values from column
    reshaped_vals = name_not_null.values.reshape(-1,1) # extract series values only and reshape them for
    encoded_vals = y.fit_transform(reshaped_vals) # function takes in array
    dataframe.loc[independent_series.notnull(), independent_series.name] = np.squeeze(encoded_vals)
    
    return dataframe

# ===============
# KNN Imputation
# ===============

def KNN_Imputation(dataframe):
    
    dataframe_array = dataframe.to_numpy().astype(float)
    dataframe_impute_KNN = impy.fast_knn(dataframe_array)

    return pd.DataFrame(dataframe_impute_KNN)  

# ====================================================================================================
# concatenate the imputed dataframes(categorical/float) into one total dataframe for further analysis
# ====================================================================================================

def Concatenate_total_df(dataframefloat, dataframecategorical):

    df_loan_total_no_missing = pd.concat([ED.df_loan_float, df_loan_categorical_mode], axis = 1)
    return df_loan_total_no_missing

df_loan_total_no_missing = Concatenate_total_df(ED.df_loan_float, df_loan_categorical_mode)

#logger.debug('Concatenate_total_df: {}, {}={}'.format(ED.df_loan_float, df_loan_categorical_mode, Concatenate_total_df))
#logger.info('Finished')