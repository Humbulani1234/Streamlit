
# =================
# MODEL ASSESSMENT
# =================

# =======================
# Perfomance measurement
# =======================

# ==========================================
# ROC Curve Analytics and Optimal threshold
# ==========================================
# 
import GLM_Bino
import train_test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def ROC_Curve_Analytics(function, X_test, Y_test, X_train, Y_train):
    
    res = (function(X_train, Y_train))[1]
    predict_probability = res.predict(X_test)

    fpr,tpr,thresholds = metrics.roc_curve(Y_test, predict_probability)

    plt.plot(fpr,tpr)

    optimal_idx = np.argmax(tpr-fpr)
    optimal_thres = thresholds[optimal_idx]
    
    return optimal_thres, #plt.show()

#print(ROC_Curve_Analytics(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.Y_test, train_test.X_train\
#, train_test.Y_train))

# ========================================
# Prediction Function @ maximal threshold
# ========================================

def Predict(function, X_test, Y_test, X_train, Y_train, threshold):
     
    res = function(X_train, Y_train)[1]
    predict_probability = res.predict(X_test)
    k = predict_probability.values.tolist()
    predict_binary = k.copy()

    for i in range(Y_test.shape[0]):

        if predict_binary[i] < threshold:
            predict_binary[i] = 1
            
        else: 
            predict_binary[i] = 0
        
        predict_binary = pd.Series(predict_binary)

    return predict_binary

#p = (Predict(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.Y_test, train_test.X_train, train_test.Y_train\
#, threshold=0.5))

#======================
#Confusion Matrix Plot
#======================

def Confusion_matrix_plot(function, X_test, Y_test, X_train, Y_train, threshold):
    
    predict_binary = Predict(function, X_test, Y_test, X_train, Y_train, threshold)
    
    z = confusion_matrix(Y_test, predict_binary, labels = [0, 1])
    z_1 = ConfusionMatrixDisplay(z, display_labels = ["No Default", "Yes Default"])
    
    return z, z_1.plot()

a = (Confusion_matrix_plot(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.Y_test, train_test.X_train\
    , train_test.Y_train, threshold=0.5))

def Prediction(function, X_test, X_train, Y_train):
     
    res = function(X_train, Y_train)[1]
    predict_probability = res.predict(X_test)
    k = [round(i,10) for i in predict_probability.values.tolist()]
    predict_binary = k.copy()

    return predict_binary

#print(Prediction(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.X_train, train_test.Y_train))