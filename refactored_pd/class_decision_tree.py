

"""

	===================================
	MODEL ALTERNATIVES - Decision Tree
	===================================

	We investigate Decision Trees as a model alternative to GLM - Binomial

	==========
	Base Tree
	==========

	===============
	Fit a base tree
	===============


"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

from class_modelperf import ModelPerfomance
from class_traintest import OneHotEncoding

# -----------------------------------------------------------------Class DecisionTree------------------------------------------------


class BaseDecisonTree(OneHotEncoding, ModelPerfomance):

	""" Fit a base tree """

	def__init__(self, custom_rcParams, df_nomiss_cat, which, func, x_test, y_test,
	             x_train, y_train, threshold, randomstate, ccpalpha):

	    super().__init__(custom_rcParams, df_nomiss_cat, which)

	    ModelPerfomance.__init__(self, func, x_test, y_test, x_train, y_train, threshold)

	    self.randomstate = randomstate
	    self.ccpalpha = ccpalpha


	def dt_classification_fit(self):
	    
	    ''' DT Classification fit '''
	    
	    clf_dt = DecisionTreeClassifier(self.randomstate, self.ccpalpha)
	    clf_dt = clf_dt.fit(self.x_train, self.y_train)
	    
	    return clf_dt


	def dt_binary_prediction(self):
	    
	    ''' Base tree prediction '''
	    
	    clf_dt = self.function(self.x_train, self.y_train, self.randomstate, self.ccpalpha)
	    
	    predict_dt = clf_dt.predict(self.x_test)
	    predict_dt_series = pd.Series(predict_dt)

	    return predict_dt_series


	def dt_confusion_matrix_plot(self):

		""" Base tree Confusion matrix """

	    predict_dt_series = self.dt_binary_prediction()       
        conf_matrix = confusion_matrix(self.y_test, predict_dt_series, labels = [0, 1])

        conf_matrix_plot = ConfusionMatrixDisplay(conf_matrix, display_labels = ["No Default", "Yes Default"])
        conf_matrix_plot.plot(cmap="Blues", ax=self.axs, values_format="d")       
        conf_matrix_plot.ax_.set_title("Confusion Matrix", fontsize=15, pad=18)
        conf_matrix_plot.ax_.set_xlabel("Predicted Label",fontsize=14)
        conf_matrix_plot.ax_.set_ylabel('True Label', fontsize = 14)

        return self.fig


	def plot_dt(self):

		""" Base tree plot """
	    
	    clf_dt = self.function(self.x_train, self.y_train, self.randomstate, self.ccpalpha)

	    plot_tree(clf_dt, filled = True, rounded = True, 
	    	      class_names = ["No Default", "Yes Default"]
	              , feature_names = self.x_train.columns, ax = sself.axs)   

	    return self.fig  


# --------------------------------------------------------------Class PrunedTree------------------------------------------------------------

class PrunedDecisionTree(BaseDecisonTree):


	def pruning(self):

		""" Extracting alphas for pruning """
	    
	    clf_dt = self.function(self.x_train, self.y_train, self.randomstate, self.ccpalpha)
	    path = clf_dt.cost_complexity_pruning_path(self.x_train, self.y_train) 
	    ccp_alphas = path.ccp_alphas 
	    ccp_alphas = ccp_alphas[:-1] 
	    
	    return ccp_alphas


	def cross_validate_alphas(self):
	    
        """ Cross validation for best alpha """

	    alpha_loop_values = []
	    
	    ccp_alphas = self.pruning()

	    for ccp_alpha in ccp_alphas:

	        clf_dt = DecisionTreeClassifier(random_state=self.randomstate, ccp_alpha=self.ccp_alpha)
	        scores = cross_val_score(clf_dt, self.x_train, self.y_train, cv=5)
	        alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
	    
	    alpha_results = pd.DataFrame(alpha_loop_values, columns=["alpha", "mean_accuracy", "std"])
	    alpha_results.plot(ax = self.axs, x = "alpha", y = "mean_accuracy", yerr = "std"
	    	               , marker = "o" , linestyle = "--")
	    
	    return alpha_results, self.fig


	def ideal_alpha(self, threshold_1, threshold_2):
	    
        """ Extraction of ideal alpha """

	    alpha_results = Cross_Validate_Alphas(func, X_train, Y_train, randomstate, ccpalpha)[0]
	    
	    ideal_ccp_alpha = alpha_results[(alpha_results["alpha"] > threshold_1)\
	     & (alpha_results["alpha"] < threshold_2)]["alpha"]

	    ideal_ccp_alpha = ideal_ccp_alpha.values.tolist()
	    
	    return ideal_ccp_alpha[0]

    
    def dt_pruned_tree(self):
	

		                     """ Ideal alpha value for pruning the tree """

		ideal_ccp_alpha = Ideal_Alpha(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, threshold_1=0.0019\
		                              , threshold_2=0.0021, randomstate=42, ccpalpha=0)

		                     """ Pruned tree fitting """

		pruned_clf_dt = DT_Classification_fit(train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=ideal_ccp_alpha)

		                     """ Prediction and perfomance analytics """

		pruned_predict_dt = Predict_binary_DT(DT_Classification_fit, train_test1.X_train.iloc[0].values.reshape(1,-1), train_test1.Y_train, train_test1.X_test\
		                                     , train_test1.Y_test, randomstate=42, ccpalpha=ideal_ccp_alpha)

		                     """ Confusion matrix plot """

		pruned_confusion_matrix = Confusion_matrix_plot_DT(DT_Classification_fit, train_test1.X_train, train_test1.Y_train\
		     , train_test1.X_test, train_test1.Y_test, randomstate=42, ccpalpha=ideal_ccp_alpha)

		                     """ Plot final tree """

		pruned_plot_tree = Plot_DT(DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=ideal_ccp_alpha)

        return ideal_ccp_alpha, clf_dt_pruned, pruned_predict_dt, pruned_confusion_matrix, pruned_plot_tree


# --------------------------------------------------------Testing--------------------------------------------------------------------------------

# if __name__ == "__main__":

	file_path = "./KGB.sas7bdat"
    data_types, df_loan_categorical, df_loan_float = data_cleaning(file_path)    
    miss = ImputationCat(df_cat=df_loan_categorical)
    imputer_cat = miss.simple_imputer_mode()
    #print(imputer_cat)
    to_view = miss.concatenate_total_df(df_loan_float, imputer_cat)

    #print(to_use)

    # custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

    custom_rcParams = {"figure.figsize": (8, 6), "axes.labelsize": 12}

    instance = OneHotEncoding(custom_rcParams, imputer_cat, True)
    #instance.sample_imbalance(df_loan_float, df_loan_float["GB"])
    
    x_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[0]
    y_train = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[2]
    y_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[3]
    x_test = instance.split_xtrain_ytrain(df_loan_float, target=df_loan_float["GB"])[1]

    #pdb.set_trace()

    x_test = sm.add_constant(x_test.values)
 
    #pdb.set_trace()

    y_train_shape = y_train.values.reshape(-1,1)

    #pdb.set_trace()

    m = (glm_binomial_fit(y_train_shape, x_train))[1]

    a = m.predict(x_test).round(10)

    # Model Perfomance
    
    threshold = 0.47
    func = glm_binomial_fit

    p = ModelPerfomance(custom_rcParams, func, x_test, y_test, x_train, y_train, threshold)
    #c = p.confusion_matrix_plot()
    r = p.confusion_matrix_plot()


