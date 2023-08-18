
"""
    =================
    MODEL ASSESSMENT
    =================
    
    And

    =======================
    Perfomance measurement
    =======================
    
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from glm_binomial import glm_binomial_fit

# --------------------------------------------------------Model Perfomance class----------------------------------------------------------

with open('glm_binomial.pkl','rb') as file:
        loaded_model = pickle.load(file)

class ModelPerfomance(Base):

    def __init__(self, custom_rcParams, x_test, y_test, threshold):

        super().__init__(custom_rcParams)

        self.x_test = x_test
        self.y_test = y_test
        self.threshold = threshold
        self.predict_glm = loaded_model.predict(self.x_test)
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.y_test, self.predict_glm)

    def roc_curve_analytics(self):
    
        """ Roc curve analytics and plot """

        self.fig, self.axs = plt.subplots(1,1)
        self.axs.plot(self.fpr, self.tpr)

        super().plotting("Roc Curve", "fpr", "tpr")

        return self.fig
   
    def optimal_threshold(self):

        self.optimal_idx = np.argmax(self.tpr - self.fpr)
        self.optimal_thres = self.thresholds[self.optimal_idx]
        
        return self.optimal_thres

    def binary_prediction(self):
         
        """ Prediction Function @ maximal threshold """

        self.k = self.predict_glm.tolist()
        self.predict_binary = self.k.copy()

        for i in range(self.y_test.shape[0]):

            if self.predict_binary[i] < self.threshold:

                self.predict_binary[i] = 1               
        
            else: 

                self.predict_binary[i] = 0
            
            self.predict_binary = pd.Series(self.predict_binary)

        return self.predict_binary


    def confusion_matrix_plot(self):
        
        """ confusion matrix plot """

        self.fig, self.axs = plt.subplots(1,1) # find refactoring method
        predict_binary = self.binary_prediction()       
        conf_matrix = confusion_matrix(self.y_test, predict_binary, labels = [0, 1])
        conf_matrix_plot = ConfusionMatrixDisplay(conf_matrix, display_labels = ["No Default", "Yes Default"])
        conf_matrix_plot.plot(cmap="Blues", ax=self.axs, values_format="d")       
        conf_matrix_plot.ax_.set_title("Confusion Matrix", fontsize=15, pad=18)
        conf_matrix_plot.ax_.set_xlabel("Predicted Label",fontsize=14)
        conf_matrix_plot.ax_.set_ylabel('True Label', fontsize = 14)

        return self.fig

    def probability_prediction(self):
         
        self._z = [round(i,10) for i in self.predict_glm.tolist()]
        prediction_prob = self._z.copy()

        return prediction_prob

# -----------------------------------------------------------------------Testing---------------------------------------------------------    

# if __name__ == "__main__":
    
#     file_path = "KGB.sas7bdat"
#     data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
#     miss = ImputationCat(df_cat=df_loan_categorical)
#     imputer_cat = miss.simple_imputer_mode()
#     to_view = miss.concatenate_total_df(df_loan_float, imputer_cat)

#     custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

#     instance = OneHotEncoding(custom_rcParams, imputer_cat, "statistics")
#     #instance.sample_imbalance(df_loan_float, df_loan_float["GB"])
    
#     x_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]
#     y_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[2]
#     y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
#     x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]

#     x_test = sm.add_constant(x_test.values)
#     y_train_shape = y_train.values.reshape(-1,1)

#     #pdb.set_trace()

#     m = (glm_binomial_fit(y_train_shape, x_train))[1]
#     a = m.predict(x_test).round(10)

#     # Model Perfomance
    
#     threshold = 0.47
#     func = glm_binomial_fit

#     p = ModelPerfomance(custom_rcParams, x_test, y_test, threshold)
#     r = p.confusion_matrix_plot()
#     plt.show()
