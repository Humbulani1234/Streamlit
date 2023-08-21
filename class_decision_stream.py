
import streamlit as st 
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from math import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score

from class_traintest import OneHotEncoding
from class_base import Base
from pd_download import data_cleaning
from class_missing_values import ImputationCat
from class_decision_tree import DecisionTree
 
# ---------------------------------------------------------------DecisionStream--------------------------------------------------

class DecisionStream(DecisionTree):

    def dec_get_dataset(self, data):

        st.dataframe(data)
        st.write('Shape of independent variables training dataframe:', data.shape)

    def dec_get_perfomance(self, name, x_test_orig, y_test_orig,
                           ccpalpha, threshold_1, threshold_2):

        data = None

        if name=='Cross Validation Alpha':

            data = super().cross_validate_alphas(ccpalpha)[1]

        elif name=='Confusion Matrix':

            data = super().dt_pruned_confmatrix(ccpalpha, threshold_1, threshold_2, x_test_orig, y_test_orig)

        else:

            data = super().dt_pruned_tree(ccpalpha, threshold_1, threshold_2)

        return data

    def dec_get_prediction(self):  

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
        
            TITLE = form.cleaned_data.get("TITLE")
            R,H = 0,0
            if TITLE == 'H':
                H=1
                # list_.append(H)
            else:
                R=0
                # list_.append(H)
            #input_ = [H]
            #
            STATUS = form.cleaned_data.get("STATUS")

            W,V, U, G, E, T = 0,0,0,0,0,0    

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
                W = 0 

            PRODUCT = form.cleaned_data.get("PRODUCT") 

            Radio_TV_Hifi, Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT = 0,0,0,0,0,0    

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
                Radio_TV_Hifi = 0   

            RESID = form.cleaned_data.get("RESID")

            Owner,Lease = 0,0    

            if RESID=='Lease':
                Lease=1    

            else:
                Owner=0

            NAT = form.cleaned_data.get("NAT")

            Yugoslav,German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0,0    

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
                Yugoslav = 1 

            PROF = form.cleaned_data.get("PROF")  

            State_Steel_Ind,Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
            ,Pensioner ,Sea_Vojage_Gast, Military_Service = 0,0,0,0,0,0,0,0,0    

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
                State_Steel_Ind = 1 

            CAR = form.cleaned_data.get("CAR")   

            Without_Vehicle,Car,Car_and_Motor_bi= 0,0,0    

            if CAR=='Car':
                Car=1
            elif CAR=='Car_and_Motor_bi':
                Car_and_Motor_bi=1
            else:
                Without_Vehicle= 1    

            Cheque_card,no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0,0  

            CARDS = form.cleaned_data.get("CARDS")  

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
                Cheque_card = 1  


            inputs1 = [H, R, E, G, T, U, V, W, Cars, Dept_Store_Mail, Furniture_Carpet, Leisure, OT, Radio_TV_Hifi, Lease, Owner  
            , German, Greek, Italian, Other_European, RS, Spanish_Portugue, Turkish, Yugoslav, Chemical_Industr,  Civil_Service_M 
            , Food_Building_Ca, Military_Service, Others, Pensioner, Sea_Vojage_Gast, Self_employed_pe, State_Steel_Ind  
            , Car, Car_and_Motor_bi, Without_Vehicle, American_Express, Cheque_card, Mastercard_Euroc, Other_credit_car, VISA_Others  
            , VISA_mybank, no_credit_cards]
            
            inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS\
            , REGN, DIV, CASH]    

            list_ = inputs2 + inputs1

            inputs = np.array([list_]).reshape(1,-1)           
            answer = d.dt_pruned_tree(0, inputs, x_test, y_test, ccpalpha, threshold_1, threshold_2)[2]

            st.subheader('Customer {} probability of default is: {}'.format(NAME , prediction))
            st.success('Successfully executed the model')  