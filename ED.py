
import pandas as pd
import warnings
import sys
import os

pd.set_option("display.width", 30000)
pd.set_option("display.max_columns", 30000)
pd.set_option("display.max_rows", 30000)
pd.set_option("display.float_format", lambda x: "%.0f" %x)
warnings.filterwarnings("ignore")

path = 'KGB.sas7bdat'

def Data_download(file_path):
    
    '''Data Download function'''
    
    df_loan = pd.read_sas(file_path)
    return df_loan

df_loan = Data_download(file_path = path)

def Data_cleaning(df):
    
    '''Various data cleaning functionalities'''
    
    data_types = df.dtypes
    df_loan_categorical = df.select_dtypes(object)
    df_loan_float = df_loan.select_dtypes(float)

    for i in range(df_loan_categorical.shape[0]):
        for j in range(df_loan_categorical.shape[1]):

            if type(df_loan_categorical.iloc[i,j]) == bytes:
                
                y = df_loan_categorical.iloc[i,j].decode("utf-8")
                df_loan_categorical.replace(df_loan_categorical.iloc[i,j], y, inplace=True)

            else:
                pass

    df_loan_categorical['PRODUCT'] = df_loan_categorical['PRODUCT'].replace('Others','OT')
    df_loan_categorical['NAT'] = df_loan_categorical['NAT'].replace('Others','RS')

    return data_types, df_loan_categorical, df_loan_float

data_types, df_loan_categorical, df_loan_float = Data_cleaning(df=Data_download(file_path = path))