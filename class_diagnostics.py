
""" 
    ==================
    Diagonostics Tests
    ==================
 
       Hypothesis Tests and Visual Plots:
     
         1. Quantile Residuals - Residuals for Discrete GLMs
         2. Breush Pagan Test - Heteroskedasticity of Variance
         3. Normal Residuals Test
         4. Durbin Watson Test - Test for Errors Serial Correlation
         5. Leverage Studentized Quantile Residuals
         6. Partial Residuals Plots
         7. Cooks Distance Quantile Residuals

"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import norm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from scipy.stats import probplot, normaltest
from math import sqrt
import statsmodels.api as sm
import pickle
import statsmodels.stats.diagnostic as sd

from class_modelperf import ModelPerfomance
from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from glm_binomial import glm_binomial_fit

# ----------------------------------------------------Base Class-----------------------------------------------------------

class QuantileResiduals(ModelPerfomance):

    def quantile_residuals(self):

        residuals = []

        try:

            if not isinstance(self.x_test, np.ndarray):

                raise TypeError("must be an instance of a numpy-ndarray")
            
            self.predict_probability = super().probability_prediction()

            if self.y_test.shape[0] is None:

                raise IndexError ("index empty")

            for i in range(self.y_test.shape[0]):

                if 0 <= self.threshold <= 1:

                    if (self.predict_probability[i] < self.threshold):

                        u_1 = np.random.uniform(low=0, high=self.predict_probability[i])
                        residuals.append(norm.ppf(u_1))

                    else:

                        u_2 = np.random.uniform(low=self.predict_probability[i], high=1)
                        residuals.append(norm.ppf(u_2))

                elif (self.threshold < 0 or self.threshold > 1):

                    raise ValueError("threshold outside bounds: [0-1]")

            quantile_residuals_series = pd.Series(residuals).round(2)

            return quantile_residuals_series

        except (TypeError, ValueError, IndexError) as e:

            print("Error:", e)

            return None

#------------------------------------------------------------Residuals Plot---------------------------------------

class ResidualsPlot(QuantileResiduals):

    def plot_quantile_residuals(self):

        """ Residuals Plot """

        self.fig, self.axs = plt.subplots(1,1)

        try:

            quantile_residuals_series = super().quantile_residuals()

            if quantile_residuals_series is None:

                raise ValueError ("residuals empty")

            self.axs.plot(quantile_residuals_series.index, quantile_residuals_series.values)
            super().plotting("humbu", "x", "y")

            return self.fig
        
        except ValueError as v:

            print("Error:", v)

            return None

# -------------------------------------------------Breush Pagan Test---------------------------------------------------

class BreushPaganTest(QuantileResiduals):


    def breush_pagan_quantile(self):

        """ Breush Pagan Test for Hetereskedasticity of variance """

        quantile_residuals_series = super().quantile_residuals()

        try:

            if quantile_residuals_series is None:
                raise ValueError ("residuals empty")

            self.test = sd.het_breuschpagan(quantile_residuals_series, self.x_test)

            return self.test
        
        except ValueError as v:

            print("Error:", v)

            return None

# ------------------------------------------------------Normality Test-----------------------------------------------

class NormalityTest(QuantileResiduals):

    def normality_test_quantile(self):

        """ normal test statistics """

        quantile_residuals_series = super().quantile_residuals()
        self.normal_test = normaltest(quantile_residuals_series)

        return self.normal_test

    def plot_normality_quantile(self):

       """ normality plot"""

       self.fig, self.axs = plt.subplots(1,1)
       quantile_residuals_series = super().quantile_residuals()
       self.qqplot = stats.probplot(quantile_residuals_series, dist="norm")
       self.axs.plot(self.qqplot[0][0],self.qqplot[0][1], marker='o', linestyle='none')
       super().plotting("Normality Test", "x", "y")
        
       return self.fig

# ------------------------------------------------Durbin Watson Test-----------------------------------------------------

class DurbinWatsonTest(QuantileResiduals):

    def durbin_watson_quantile(self):

        """ Durbin Watson Test for Residuals correlation range(1,5 - 2) """

        quantile_residuals_series = super().quantile_residuals()
        self.durbin_watson_corr_test = durbin_watson(quantile_residuals_series)

        return self.durbin_watson_corr_test

# ----------------------------------------------Partial Plots-------------------------------------------------------

class PartialPlots(QuantileResiduals):

    def partial_plots_quantile(self, ind_var):

       """ Partial Plots - Residuals vs Features """

       self.fig, self.axs = plt.subplots(1,1)
       quantile_residuals_series = super().quantile_residuals()
       self.xlabel_name = ind_var.name
       self.axs.scatter(ind_var, quantile_residuals_series)
       super()._plotting("Partial Plot", self.xlabel_name, "y")
        
       return self.fig

# -------------------------------------------------Leverage Studentised Residuals-----------------------------------------

class LevStudQuaRes(QuantileResiduals):

    def plot_lev_stud_quantile(self):

       """ Outliers and Influence """

       self.fig, self.axs = plt.subplots(1,1)
       res = self.function(self.x_train, self.y_train)[1]
       quantile_residuals_series = super().quantile_residuals()
       hat_matrix = np.round(res.get_hat_matrix_diag(),2)
       self.lev_stud_res = []

       for i in range(len(quantile_residuals_series)):
            
        self.lev_stud_res.append(quantile_residuals_series[i]/(sqrt(1-hat_matrix[i])))

       self.axs.plot(pd.Series(self.lev_stud_res).index, pd.Series(self.lev_stud_res).values)
       super()._plotting("Leverage Studentised Residuals", "x", "y")
        
       return self.fig

# -------------------------------------------------Cooks Distance Residuals---------------------------------------------

class CooksDisQuantRes(QuantileResiduals):

    def plot_cooks_dis_quantile(self):

        """ Cooks Distance Plot """

        self.fig, self.axs = plt.subplots(1,1)
        res = self.function(self.x_train, self.y_train)[1]
        quantile_residuals_series = super().quantile_residuals()
        hat_matrix = np.round(res.get_hat_matrix_diag(),2)
        self.d = []

        for i in range(len(quantile_residuals_series)):
            
            self.d.append((quantile_residuals_series[i]**2/3000)*(hat_matrix[i]/(1-hat_matrix[i])))

        self.axs.plot(pd.Series(self.d).index, pd.Series(self.d).values)
        super()._plotting("Leverage Studentised Residuals", "x", "y")

        return self.fig

# -----------------------------------------------Testing-------------------------------------------

# if __name__ == "__main__":

#     file_path = "./KGB.sas7bdat"
#     data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
#     miss = ImputationCat(df_cat=df_loan_categorical)
#     imputer_cat = miss.simple_imputer_mode()
#     #print(imputer_cat)
#     to_view = miss.concatenate_total_df(df_loan_float, imputer_cat)

#     #print(to_use)

#     # custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

#     custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

#     instance = OneHotEncoding(custom_rcParams, imputer_cat, "statistics")
#     #instance.sample_imbalance(df_loan_float, df_loan_float["GB"])
    
#     x_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]
#     y_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[2]
#     y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
#     x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]

#     #pdb.set_trace()

#     x_test = sm.add_constant(x_test.values)
 
#     #pdb.set_trace()

#     y_train_shape = y_train.values.reshape(-1,1)

#     #pdb.set_trace()

#     m = (glm_binomial_fit(y_train_shape, x_train))[1]

#     a = m.predict(x_test).round(10)

#     # Model Perfomance
    
#     threshold = 0.47
#     func = glm_binomial_fit

#     p = ModelPerfomance(custom_rcParams, x_test, y_test, threshold)
#     #c = p.confusion_matrix_plot()
#     #r = p.confusion_matrix_plot()
#     #plt.show()

#     # Diagnostics
    
#     # a = QuantileResiduals(custom_rcParams, func, x_test, y_test, x_train, y_train, threshold)
#     # b = a.quantile_residuals()
#     # print(r)

#     b = ResidualsPlot(custom_rcParams, x_test, y_test, threshold)
#     c = b.plot_quantile_residuals()
#     plt.show()
