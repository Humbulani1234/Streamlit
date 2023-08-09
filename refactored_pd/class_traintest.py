

"""

     ==========================
     TRAIN AND TESTING SAMPLES
     ==========================

         1. One Hot Encoding
         2. Train and Testing sample split

     =================
     One Hot Encoding:
     =================

"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
import warnings

from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat

# ------------------------------------------------------Settings---------------------------------------------------------------

pd.set_option("display.width", 1100)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.float_format", lambda x: "%.0f" %x)
warnings.filterwarnings("ignore")

# -----------------------------------------------------Class OneHotEncoding-----------------------------------------------------

class OneHotEncoding(Base):

    def __init__(self, custom_rcParams, df_nomiss_cat, which):

        super().__init__(custom_rcParams)

        self.df_nomiss_cat = df_nomiss_cat
        self.which = which

    def one_hot_encoding(self, which):

        if self.which == False:

            def onehot_encoding_machine(self):
            
                '''One Hot Encoding Function'''
            
                encoded_dataframes = []

                for col in self.df_nomiss_cat.columns:
                    y = pd.get_dummies(self.df_nomiss_cat[col])
                    encoded_dataframes.append(y)

                df_cat_onehotenc = pd.concat(encoded_dataframes, axis = 1)
                
                return df_cat_onehotenc

        if self.which == True:

            def onehot_encoding_statistics(self):
            
                '''One Hot Encoding Function'''
            
                encoded_dataframes = []

                for col in self.df_nomiss_cat.columns:
                    y = pd.get_dummies(self.df_nomiss_cat[col])
                    n = len(pd.unique(self.df_nomiss_cat[col])) 
                    self.df_nomiss_cat = y.drop(y.columns[n-1], axis=1) 
                    encoded_dataframes.append(self.df_nomiss_cat)

                df_cat_onehotenc = pd.concat(encoded_dataframes, axis = 1)
                
                return df_cat_onehotenc

    def create_xy_frames(self, df_float, target):
        
        if self.which == True:

            df_cat = self.one_hot_encoding(True)

            df_total_partition = pd.concat([df_float, df_cat], axis = 1)
            x = df_total_partition.drop(labels=[target.name], axis=1)
            y = df_total_partition[target.name]
            
            return x, y

        if self.which == False:

            df_cat = self.one_hot_encoding(False)

            df_total_partition = pd.concat([df_float, df_cat], axis = 1)
            x = df_total_partition.drop(labels=[target.name], axis=1)
            y = df_total_partition[target.name]

            return x, y

    def sample_imbalance(self, df_float, target):
    
        x, y = self.create_xy_frames(df_float, target)

        self.fig, self.axs = plt.subplots(1,1)
        
        self.axs.hist(y, weights = np.ones(len(y))/len(y))
        super().plotting("Normality Test", "x", "y")
        self.axs.hist(y, weights = np.ones(len(y))/len(y))

        self.axs.yaxis.set_major_formatter(PercentFormatter(1))
        
        return self.fig

    def split_xtrain_ytrain(self, df_float, target):
    
        x, y = self.create_xy_frames(df_float, target)

        x_train_pd, x_test_pd, y_train_pd, y_test_pd = train_test_split(x, y, test_size=0.3, random_state=42)

        x_train_pd = x_train_pd.drop(labels=["_freq_"], axis=1) # temp, for mach it has to be dropped

        x_test_pd = x_test_pd.drop(labels=["_freq_"], axis=1) # temp

        
        return x_train_pd, x_test_pd, y_train_pd, y_test_pd

# ---------------------------------------------------------Testing-------------------------------------------------------


if __name__ == "__main__":


    file_path = "./KGB.sas7bdat"
    data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
    miss = ImputationCat(df_cat=df_loan_categorical)
    imputer_cat = miss.simple_imputer_mode()
    print(imputer_cat)
    to_view = miss.concatenate_total_df(df_loan_float, imputer_cat)

    #print(to_use)

    custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

    # x_test = X_test 
    # x_train = X_train
    # y_test = Y_test
    # y_train = Y_train.to_frame()
    # threshold = 0.47
    # func = GLM_Binomial_fit

    instance = OneHotEncoding(custom_rcParams, imputer_cat, True)
    #print(instance.df_nomiss_cat)
    y = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]
    si = instance.sample_imbalance(df_loan_float, df_loan_float["GB"])
    plt.show()































