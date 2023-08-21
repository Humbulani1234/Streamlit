

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
import statsmodels.stats.diagnostic as sd

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from class_diagnostics import (ResidualsPlot, BreushPaganTest, NormalityTest, DurbinWatsonTest,
                               PartialPlots, LevStudQuaRes, CooksDisQuantRes)
from class_modelperf import ModelPerfomance
from class_decision_stream import DecisionStream
import data_stream
from class_decision_tree import DecisionTree

# -------------------------------------------------------------------BaseClass--------------------------------------------------------

class BaseStreamlit():

    def __init__(self, title: str, image, subheader: str, classifier_name: tuple):

        title = """ <div style="border: 2px solid black; padding:5px; box-shadow:3px 3px 7px grey
                        display:flex; align-items:center; justify-content:center; text-align:center;
                        font-family:Arial,sans-serif;font-weight:bold; font-size:24px;">

                    Probability of Default Prediction
                """

        subheader = """ <div style="border: 2px solid black; padding:5px; box-shadow:3px 3px 7px grey
                        display:flex; align-items:center; justify-content:center; text-align:center;
                        font-family:Arial,sans-serif;font-weight:bold; font-size:24px;">

                    Various Perfomance Plots
                """

        self.title = st.markdown(title, unsafe_allow_html=True)
        self.legend_1 = st.markdown("<legend></legend>", unsafe_allow_html=True)
        self.image = Image.open(image)
        st.image(self.image, use_column_width=True)
        self.subheader = st.markdown(subheader, unsafe_allow_html=True)
        self.legend_2 = st.markdown("<legend></legend>", unsafe_allow_html=True)
        self.legend_2 = st.markdown("<legend></legend>", unsafe_allow_html=True)
        self.classifier_name = st.sidebar.selectbox('Select classifier', classifier_name)

# ---------------------------------------------------------Logistic------------------------------------------------------------------

class Logistic(ResidualsPlot, BreushPaganTest, NormalityTest, DurbinWatsonTest,
               PartialPlots, LevStudQuaRes, CooksDisQuantRes, ModelPerfomance):

    def log_get_dataset(self, data):

        st.dataframe(data)
        st.write('Shape of independent variables training dataframe:', data.shape)

    def log_get_diagnostics(self, name, ind_var):

        data = None

        if name=='Quantile Res':

            st.write('Quantile Residuals',super().normality_test_quantile())
            data = super().plot_quantile_residuals()

        elif name=='Breush_Pagan_Test':

            st.write('Breush_Pagan_Test',super().breush_pagan_quantile())

        elif name=='Normal_Residual_Test':

            st.write('Normal_Residual_Test',super().normality_test_quantile())
            data = super().plot_normality_quantile()

        elif name=='Durbin_Watson_Test':

            st.write('Durbin_Watson_Test',super().durbin_watson_quantile())        
            data = super().plot_quantile_residuals()

        elif name=='Partial_Plots':

            data = super().partial_plots_quantile(ind_var)

        elif name=='Leverage_Studentized_Quantile_Res':

            data = super().plot_lev_stud_quantile()
        
        else:

            data = super().plot_cooks_dis_quantile()

        return data

    def log_get_perfomance(self, name):

        data = None

        if name=='ROC Curve':

            data = super().roc_curve_analytics()

        elif name=='Confusion Matrix':

            data = super().confusion_matrix_plot()

        return data

    def log_get_prediction(self):  

        NAME = st.sidebar.text_input("CUSTOMER NAME")
        AGE = st.sidebar.slider("AGE", 0,100)
        CHILDREN = st.sidebar.slider("CHILDREN", 0, 10)
        PERS_H = st.sidebar.slider("PERS_H", 0, 10)
        TMADD = st.sidebar.slider("TMADD", 0, 1000)
        TMJOB1 = st.sidebar.slider("TMJOB1", 0, 1000)
        TEL = st.sidebar.slider("TEL", 1, 10)
        NMBLOAN = st.sidebar.slider("NMBLOAN", 0, 10)
        FINLOAN = st.sidebar.slider("FINLOAN", 0, 10)
        INCOME = st.sidebar.slider("INCOME", 1, 1000000,100)
        EC_CARD = st.sidebar.slider("EC_CARD", 1, 10,1)
        INC = st.sidebar.slider("INC", 1, 1000000,100)
        INC1 = st.sidebar.slider("INC1", 1, 10,1)
        BUREAU = st.sidebar.slider("BUREAU", 1, 10,1)
        LOCATION = st.sidebar.slider("LOCATION", 1, 10,1)
        LOANS = st.sidebar.slider("LOANS", 1, 10,1)
        REGN = st.sidebar.slider("REGN", 1, 10,1)
        DIV = st.sidebar.slider("DIV", 1, 10,1)
        CASH = st.sidebar.slider("CASH", 1, 1000000,100)        

        # Categorical features
        
        TITLE = st.sidebar.selectbox("TITLE", options=['H','R'])
        STATUS = st.sidebar.selectbox("STATUS",options=['V','U','G','E','T','W'])

        PRODUCT = st.sidebar.selectbox('PRODUCT',options=['Radio_TV_Hifi','Furniture_Carpet','Dept_Store_Mail'
                                                          ,'Leisure','Cars','OT']) # dropped Radio

        RESID = st.sidebar.selectbox('RESID',options=['Lease','Owner']) # dropped Owner

        NAT = st.sidebar.selectbox('NAT',options=['German', 'Turkish','RS', 'Greek' ,'Yugoslav',
                                                  'Italian','Other_European','Spanish_Portugue']) #dropped Yugoslavia

        PROF = st.sidebar.selectbox('PROF',options=['Others','Civil_Service_M' ,'Self_employed_pe',
                                                    'Food_Building_Ca','Chemical_Industr','Pensioner' ,'Sea_Vojage_Gast',
                                                    'State_Steel_Ind,','Military_Service']) # dropped State_Steel_Ind

        CAR = st.sidebar.selectbox('CAR',options=['Car', 'Without_Vehicle', 'Car_and_Motor_bi']) # dropped Without_Vehicle

        CARDS = st.sidebar.selectbox("CARDS",options=['Cheque_card' ,'no_credit_cards', 'Mastercard_Euroc', 'VISA_mybank'
                                                      ,'VISA_Others','Other_credit_car', 'American_Express']) # dropped cheque card 

        button_clicked = st.sidebar.button('Submit')                                                    
       
        if button_clicked:

            V, U, G, E, T = 0,0,0,0,0    

            if STATUS == 'V':
                V=1
            elif STATUS == 'U':
                U=1
            elif STATUS == 'G':
                G=1
            elif STATUS == 'E':
                E=1
            elif STATUS=='T':
                T=1
            else:
                V, U, G, E, T = 0,0,0,0,0    
   

            H = 0    

            if TITLE=='H':
                H = 1
            else:
                H=0
            
            
            Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0    

            if PRODUCT=='Furniture_Carpet':
                Furniture_Carpet=1
            elif PRODUCT=='Dept_Store_Mail':
                Dept_Store_Mail=1
            elif PRODUCT=='Leisure':
                Leisure=1
            elif PRODUCT=='Cars':
                Cars=1
            elif PRODUCT=='OT':
                OT=1
            else:
                Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0    

            
            Lease = 0    

            if RESID=='Lease':
                Lease=1    

            else:
                Lease=0
        
            
            German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0    

            if NAT=='German':
                German=1
            elif NAT=='Turkish':
                Turkish=1        
            elif NAT=='RS':
                RS=1
            elif NAT=='Greek':
                Greek=1
            elif NAT=='Italian':
                Italian=1
            elif NAT=='Other_European':
                Other_European=1
            elif NAT=='Spanish_Portugue':
                Spanish_Portugue=1
            else:
                German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0    

            
            Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
            ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0    

            if PROF=='Others':
                Others=1
            elif PROF=='Civil_Service_M':
                Civil_Service_M=1
            elif PROF=='Self_employed_pe':
                Self_employed_pe=1
            elif PROF=='Food_Building_Ca':
                Food_Building_Ca=1
            elif PROF=='Chemical_Industr':
                Chemical_Industr=1
            elif PROF=='Pensioner':
                Pensioner=1
            elif PROF=='Sea_Vojage_Gast':
                Sea_Vojage_Gast=1
            elif PROF=='Military_Service':
                Military_Service=1
            else:
                Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
                ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0    

            
            Car,Car_and_Motor_bi= 0,0    

            if CAR=='Car':
                Car=1
            elif CAR=='Car_and_Motor_bi':
                Car_and_Motor_bi=1
            else:
                Car,Car_and_Motor_bi= 0,0    

            
            Cheque_card, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0    

            if CARDS=='no_credit_cards':
                no_credit_cards=1
            elif CARDS=='Mastercard_Euroc':
                Mastercard_Euroc=1
            elif CARDS == 'VISA_mybank':
                VISA_mybank=1
            elif CARDS=='VISA_Others':
                VISA_Others=1
            elif CARDS=='Other_credit_car':
                Other_credit_car=1
            elif CARDS=='American_Express':
                American_Express=1
            else:
                Cheque_card, Mastercard_Euroc, VISA_mybank,VISA_Others\
                ,Other_credit_car, American_Express = 0,0,0,0,0,0    

            inputs1 = [H, E, G, T, U, V, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Lease, German, Greek, 
                      Italian, Other_European, RS, Spanish_Portugue, Turkish, Chemical_Industr, Civil_Service_M, 
                      Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe, Car, 
                      Car_and_Motor_bi, American_Express, Cheque_card, Mastercard_Euroc, Other_credit_car, VISA_Others, VISA_mybank]
            
            inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU
                      ,LOCATION, LOANS, REGN, DIV, CASH]    

            list_ = inputs2 + inputs1
            inputs = np.array(list_).reshape(1,-1)
            answer = np.array(data.loaded_model.predict(inputs.reshape(1,-1))) 

            st.subheader('Customer {} probability of default is: {}'.format(NAME , prediction))
            st.success('Successfully executed the model')
            
# ------------------------------------------------------main function (entry point) ------------------------------------------------------

def main(custom_rcParams, x_test, y_test, df_nomiss_cat, type_,
         df_loan_float, target, ind_var, threshold, randomstate):

    basestreamlit = BaseStreamlit("Probabilty of Default Prediction",'data.png'
                                  ,"Various Perfomance Plots", ('Logistic', 'Decision'))
    logistic = Logistic(custom_rcParams, x_test, y_test, threshold)
    decision = DecisionStream(custom_rcParams, df_nomiss_cat, "machine", y_test,
                             df_loan_float, df_loan_float["GB"], threshold, randomstate)
    classifier_name = basestreamlit.classifier_name


    if classifier_name=='Logistic':

        logistic.log_get_dataset(data_stream.x_train)
        diagnostics_name=st.sidebar.selectbox('Select Diagnostic', ('Quantile Res','Breush_Pagan_Test','Normal_Residual_Test',
                                              'Durbin_Watson_Test','Partial_Plots','Leverage_Studentized_Quantile_Res',
                                              'Cooks_Distance_Quantile_Res'))
        figure = logistic.log_get_diagnostics(diagnostics_name, ind_var)

        if diagnostics_name == "Breush_Pagan_Test":
            pass

        elif diagnostics_name == "Quantile Res" or "Normal_Residual_Test" or\
                                 "Durbin_Watson_Test" or "Partial_Plots" or "Leverage_Studentized_Quantile_Res" or\
                                 "Cooks_Distance_Quantile_Res":
            st.pyplot(figure)

        elif diagnostics_name == "Breush_Pagan_Test":
            pass

        perfomance_name = st.sidebar.selectbox('Select Perfomance', ('ROC Curve', 'Confusion Matrix'))
        shape = logistic.log_get_perfomance(perfomance_name)
        st.pyplot(shape)

        logistic.log_get_prediction()

    else:

        diagnostics_name=st.sidebar.selectbox('Select Graphs', ('Cross Validation Alpha', 'Confusion Matrix', 'Tree Plot'))
        decision.dec_get_dataset(data_stream.x_train)
        figure = decision.dec_get_perfomance(diagnostics_name,
                                             data_stream.x_test_orig, data_stream.y_test_orig, data_stream.ccpalpha,
                                             data_stream.threshold_1, data_stream.threshold_2)
        st.pyplot(figure)
        decision.dec_get_prediction()

main(data_stream.custom_rcParams, data_stream.x_test, data_stream.y_test, data_stream.imputer_cat,
     "machine", data_stream.df_loan_float, data_stream.df_loan_float["GB"], data_stream.ind_var, 
     data_stream.threshold, data_stream.randomstate)

# -----------------------------------------------------------------Testing---------------------------------------------------------