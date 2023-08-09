# ===================================
# MODEL ALTERNATIVES - Decision Tree
# ===================================

# We investigate Decision Trees as a model alternative to GLM - Binomial

# ==========
# Base Tree
# ==========

# ===============
# Fit a base tree
# ===============

from sklearn.tree import DecisionTreeClassifier
import train_test1
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

def DT_Classification_fit(X_train, Y_train, randomstate, ccpalpha):
    
    '''DT Classification fit'''
    
    clf_dt = DecisionTreeClassifier(random_state = randomstate, ccp_alpha=ccpalpha)
    clf_dt = clf_dt.fit(X_train, Y_train)
    
    return clf_dt

DT_classification = DT_Classification_fit(train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=0)

# ====================
# Base tree prediction
# ====================

def Predict_binary_DT(func, X_test, Y_test, X_train, Y_train, randomstate, ccpalpha):
    
    '''Predict function'''
    
    clf_dt = func(X_train, Y_train, randomstate, ccpalpha)
    
    predict_DT = clf_dt.predict(X_test)
    predict_DT_Series = pd.Series(predict_DT)

    return predict_DT

Predict = Predict_binary_DT(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, train_test1.X_test\
    , train_test1.Y_test, randomstate=42, ccpalpha=0)

# ==========================
# Base tree Confusion matrix
# ==========================

def Confusion_matrix_plot_DT(func, X_train, Y_train, X_test, Y_test, randomstate, ccpalpha):
    
    predict_DT_Series = Predict_binary_DT(func, X_test, Y_test, X_train, Y_train, randomstate, ccpalpha)
    
    z = confusion_matrix(Y_test, predict_DT_Series)
    z_1 = ConfusionMatrixDisplay(z, display_labels = ["No Default", "Yes Default"])
    z_1.plot()
    
    return z_1.plot() 

# Confusion = Confusion_matrix_plot_DT(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, train_test1.X_test\
#     , train_test1.Y_test, randomstate=42, ccpalpha=0)

# ==============
# Base tree plot
# ==============

def Plot_DT(func, X_train, Y_train, randomstate, ccpalpha):
    
    clf_dt = func(X_train, Y_train, randomstate, ccpalpha)
    
    plt.figure(figsize = (12, 8))
    
    return plot_tree(clf_dt, filled=True, rounded=True, class_names=["No Default", "Yes Default"]\
                                       , feature_names = X_train.columns)   

#print(Plot_DT(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=0))

# ================================
# Pruned tree by Cross Validation
# ================================

# ===============================
# Extracting alphas for pruning
# ===============================

def Pruning(func, X_train, Y_train, randomstate, ccpalpha):
    
    clf_dt = func(X_train, Y_train, randomstate, ccpalpha)
    path = clf_dt.cost_complexity_pruning_path(X_train, Y_train) # determine values for alpha
    ccp_alphas = path.ccp_alphas # extract different values for alpha
    ccp_alphas = ccp_alphas[:-1] # exclude the maximum value for alpha
    
    return ccp_alphas

#print(Pruning(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=0))

# ===============================
# Cross validation for best alpha
# ===============================

def Cross_Validate_Alphas(func, X_train, Y_train, randomstate, ccpalpha):
    
    alpha_loop_values = []
    
    ccp_alphas = Pruning(func, X_train, Y_train, randomstate, ccpalpha)

    for ccp_alpha in ccp_alphas:

        clf_dt = DecisionTreeClassifier(random_state=randomstate, ccp_alpha=ccp_alpha)
        scores = cross_val_score(clf_dt, X_train, Y_train, cv=5)
        alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
    
    alpha_results = pd.DataFrame(alpha_loop_values, columns=["alpha", "mean_accuracy", "std"])
    alpha_results.plot(x="alpha", y="mean_accuracy", yerr="std", marker="o"\
                                              , linestyle="--")
    
    return alpha_results, #plt.show()

#print(Cross_Validate_Alphas(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=0))

# ==========================
# Extraction of ideal alpha
# ==========================

def Ideal_Alpha(func, X_train, Y_train, threshold_1, threshold_2, randomstate, ccpalpha):
    
    alpha_results = Cross_Validate_Alphas(func, X_train, Y_train, randomstate, ccpalpha)[0]
    
    ideal_ccp_alpha = alpha_results[(alpha_results["alpha"] > threshold_1)\
     & (alpha_results["alpha"] < threshold_2)]["alpha"]

    ideal_ccp_alpha = ideal_ccp_alpha.values.tolist()
    print(ideal_ccp_alpha)
    
    return ideal_ccp_alpha[0]

x = Ideal_Alpha(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, threshold_1=0.0019, threshold_2=0.0021\
    , randomstate=42, ccpalpha=0)

# ===================
# Final Tree fitting
# ===================

# ======================================
# Ideal alpha value for pruning the tree
# ======================================

ideal_ccp_alpha = Ideal_Alpha(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, threshold_1=0.0019\
    , threshold_2=0.0021, randomstate=42, ccpalpha=0)

# ====================
# Pruned tree fitting
# ====================

clf_dt_pruned = DT_Classification_fit(train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=ideal_ccp_alpha)

# ===================================
# Prediction and perfomance analytics
# ===================================

# ========
# Predict
# ========

predict_DT_Series = Predict_binary_DT(DT_Classification_fit, train_test1.X_train.iloc[0].values.reshape(1,-1), train_test1.Y_train, train_test1.X_test\
     , train_test1.Y_test, randomstate=42, ccpalpha=ideal_ccp_alpha)

# # ======================
# # Confusion matrix plot
# ======================

Confusion_matrix_plot = Confusion_matrix_plot_DT(DT_Classification_fit, train_test1.X_train, train_test1.Y_train\
     , train_test1.X_test, train_test1.Y_test, randomstate=42, ccpalpha=ideal_ccp_alpha)

# =================
# Plot final tree
# =================

Plot_tree = Plot_DT(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=ideal_ccp_alpha)
