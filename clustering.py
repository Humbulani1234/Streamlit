#========================================
# DATA CLUSTERING AND DIMENSION REDUCTION
# ========================================

# =======================
# K_Prototype Clustering
# =======================

# values = numerical variables
import ED
import missing_adhoc
from kmodes.kprototypes import KPrototypes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import seaborn as sns

values = list(np.arange(1,20,1))
dataframe = missing_adhoc.df_loan_total_no_missing
#print(dataframe.columns.tolist())
categorical_columns = list(np.arange(20, 28, 1))
cluster_no = 5 


columns = dataframe.columns.tolist()
#print(columns)

float_columns = []

for i in values:
    float_columns.append(columns[i])

#print(float_columns)


#from kmodes.kprototypes import KPrototypes

def K_Prototypes_Clustering(dataframe, cluster_no, categorical):
    
    loan_norm = missing_adhoc.df_loan_total_no_missing.copy()
    scaler = preprocessing.MinMaxScaler()
    loan_norm[float_columns] = scaler.fit_transform(loan_norm[float_columns])

    kproto = KPrototypes(n_clusters=cluster_no, init="Cao")
    clusters = kproto.fit_predict(loan_norm, categorical=categorical_columns)

    # join data with labels

    labels = pd.DataFrame(clusters)
    labeled_df_loan = pd.concat([missing_adhoc.df_loan_total_no_missing, labels], axis=1)
    labeled_df_loan = labeled_df_loan.rename({0: "labels"}, axis=1)
     
    return labeled_df_loan

dataframes = K_Prototypes_Clustering(dataframe, cluster_no, categorical=categorical_columns)
#print(dataframes.columns.tolist())
#print(dataframe)

# ==================
# K_Prototype Plots
# ==================

def K_prototype_plot(independent, target, dataframe):
    
    plt.figure(figsize=(3.5,3.5), dpi=100)
    
    ax = sns.swarmplot(x= independent, y= target, data=dataframe, hue="labels", zorder=0)
    ax.legend()
    
    return plt.show()

#K_prototype_plot(dataframes['CHILDREN'], dataframes['GB'], dataframes)
