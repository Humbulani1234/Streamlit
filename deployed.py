
import streamlit as st 
import train_test
import Model_Perf
import ED
import matplotlib.pyplot as plt
import GLM_Bino
import warnings
import missing_adhoc
import pandas as pd
from PIL import Image
import clustering
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import Diagnostics
from scipy.stats import norm
import pylab
import statsmodels.stats.diagnostic as sd
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import scipy
from scipy import stats
from math import *
from sklearn.tree import DecisionTreeClassifier
import train_test1
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import Decision_tree


import logging

st.set_option('deprecation.showPyplotGlobalUse', False)

pd.set_option("display.width", 3000)
pd.set_option("display.max_columns", 3000)
pd.set_option("display.max_rows", 3000)
pd.set_option("display.float_format", lambda x: "%.0f" %x)
warnings.filterwarnings("ignore")

#=========Widgets and Titles=============================================================================

st.title("Probability of Default Prediction")
image = Image.open('pngegg.png')
st.image(image,use_column_width=True)
st.subheader("This model will predict the probability of default for a customer")
classifier_name = st.sidebar.selectbox('Select classifier', ('Logistic', 'Decision'),key=30)

#===================================Datasets============================================================================

def get_dataset(name):
    
    if name=='Logistic_KGB':
        data = train_test.X_train
        st.dataframe(data)
        st.write('Shape of dataframe:', data.shape)

    elif name=='Cluster':
        data = clustering.K_Prototypes_Clustering(clustering.dataframe, clustering.cluster_no\
            , categorical=clustering.categorical_columns)
        st.dataframe(data)
        st.write('Shape of dataframe:', data.shape)

    else:
        data = train_test1.X_train
        st.dataframe(data)
        st.write('Shape of dataframe:', data.shape)

#=================================================Diagnostics Statistics==============================================

def get_dataset2(name):
    
    if name=='Quantile Res':

        st.write('Quantile Residuals',Diagnostics.Quantile_Residuals(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47))
        Diagnostics.Plot_Residuals(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)
        st.pyplot()

    elif name=='Breush_Pagan_Test':

        st.write('Breush_Pagan_Test',Diagnostics.Breush_Pagan_Test(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47))
        st.pyplot()

    elif name=='Normal_Residual_Test':

        st.write('Normal_Residual_Test',Diagnostics.Normal_Residual_Test(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47))
        Diagnostics.Normal_Residual_Test(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)
        st.pyplot()

    elif name=='Durbin_Watson_Test':

        st.write('Durbin_Watson_Test',Diagnostics.Durbin_Watson_Test(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47))
        st.pyplot()

    elif name=='Partial_Plots':

        Diagnostics.Partial_Plots(GLM_Bino.GLM_Binomial_fit, train_test.X_test["AGE"],train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)
        st.pyplot()

    elif name=='Leverage_Studentized_Quantile_Res':

        Diagnostics.Leverage_Studentized_Quantile_Res(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)
        st.pyplot()
    
    else:

        Diagnostics.Cooks_Distance_Quantile_Res(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train, threshold=0.47)
        st.pyplot()

#===================================================Data Visualization===================================

def get_data(name):

    if name=='Cross_tab':

        missing_adhoc.Categorical_missingness_Crosstab_Plot(ED.df_loan_categorical["RESID"], ED.df_loan_float["GB"])
        st.pyplot()

    elif name=='confusion':

        st.markdown('##### FN, TN, TP, FP Plot')
        confusion = Model_Perf.a[0]
        FN = confusion[1][0]
        TN = confusion[0][0]
        TP = confusion[1][1]
        FP = confusion[0][1]
        
        plt.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP],width=0.15)
        plt.xticks(fontsize=7)
        plt.xlabel("",fontsize=7)
        plt.yticks(fontsize=7)
        plt.ylabel("",fontsize=7)
        st.pyplot()

    elif name=='Clustering':

        clustering.K_prototype_plot(clustering.dataframes['CHILDREN'], clustering.dataframes['GB'], clustering.dataframes)
        st.pyplot()

    elif name=='ROC':

        Model_Perf.ROC_Curve_Analytics(GLM_Bino.GLM_Binomial_fit, train_test.X_test\
            ,train_test.Y_test, train_test.X_train, train_test.Y_train)
        st.pyplot()

    elif name=='confusion_matrix':

        Model_Perf.Confusion_matrix_plot(GLM_Bino.GLM_Binomial_fit, train_test.X_test, train_test.Y_test, train_test.X_train\
              ,train_test.Y_train, threshold=0.5)
        st.pyplot()

    else:
        missing_adhoc.Categorical_missingness_Pivot_Plot(ED.df_loan_categorical["RESID"], ED.df_loan_float["GB"])
        st.pyplot()


def get_data2(name):

    if name=='Cross_Validate_Alphas':

        Decision_tree.Cross_Validate_Alphas(Decision_tree.DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42\
            , ccpalpha=0)
        st.pyplot()

    elif name=='Ideal_Alpha':

        Decision_tree.Ideal_Alpha(Decision_tree.DT_Classification_fit, train_test1.X_train, train_test1.Y_train, threshold_1=0.0019\
            , threshold_2=0.0021, randomstate=42, ccpalpha=0)
        st.pyplot()

    elif name=='Confusion_matrix_plot_DT':

        Decision_tree.Confusion_matrix_plot_DT(Decision_tree.DT_Classification_fit, train_test1.X_train, train_test1.Y_train\
        ,train_test1.X_test, train_test1.Y_test, randomstate=42, ccpalpha=Decision_tree.ideal_ccp_alpha)
        st.pyplot()

    else:

        Decision_tree.Plot_DT(Decision_tree.DT_Classification_fit, train_test1.X_train, train_test1.Y_train, randomstate=42\
            , ccpalpha=Decision_tree.ideal_ccp_alpha)
        st.pyplot()

# ====================================Logistic Prediction=============================================================

def logistic_fea():

    NAME = st.sidebar.text_input("Customer", key=90)
    AGE = st.sidebar.slider("AGE", 0,100,key=10)
    CHILDREN = st.sidebar.slider("CHILDREN", 0, 10,key=11)
    PERS_H = st.sidebar.slider("PERS_H", 0, 10,key=12)
    TMADD = st.sidebar.slider("TMADD", 0, 1000,key=13)
    TMJOB1 = st.sidebar.slider("TMJOB1", 0, 1000,key=14)
    TEL = st.sidebar.slider("TEL", 1, 10,key=15)
    NMBLOAN = st.sidebar.slider("NMBLOAN", 0, 10,key=16)
    FINLOAN = st.sidebar.slider("FINLOAN", 0, 10,key=17)
    INCOME = st.sidebar.slider("INCOME", 1, 1000000,100,key=18)
    EC_CARD = st.sidebar.slider("EC_CARD", 1, 10,1,key=19)
    INC = st.sidebar.slider("INC", 1, 1000000,100,key=20)
    INC1 = st.sidebar.slider("INC1", 1, 10,1,key=21)
    BUREAU = st.sidebar.slider("BUREAU", 1, 10,1,key=22)
    LOCATION = st.sidebar.slider("LOCATION", 1, 10,1,key=23)
    LOANS = st.sidebar.slider("LOANS", 1, 10,1,key=24)
    REGN = st.sidebar.slider("REGN", 1, 10,1,key=25)
    DIV = st.sidebar.slider("DIV", 1, 10,1,key=26)
    CASH = st.sidebar.slider("CASH", 1, 1000000,100,key=27)        

    # Categorical
    
    TITLE = st.sidebar.selectbox("TITLE", options=['H','R'], key=2)
    STATUS = st.sidebar.selectbox("STATUS",options=['V','U','G','E','T','W'], key=3)
    PRODUCT = st.sidebar.selectbox('PRODUCT',options=['Radio_TV_Hifi','Furniture_Carpet', 'Dept_Store_Mail', 'Leisure','Cars', 'OT'], key=4) # dropped Radio
    RESID = st.sidebar.selectbox('RESID',options=['Lease','Owner'], key=5) # dropped Owner
    NAT = st.sidebar.selectbox('NAT',options=['German', 'Turkish', 'RS', 'Greek' ,'Yugoslav', 'Italian', 'Other_European','Spanish_Portugue'], key=6) #dropped Yugoslavia
    PROF = st.sidebar.selectbox('PROF',options=['Others','Civil_Service_M' ,'Self_employed_pe', 'Food_Building_Ca','Chemical_Industr'\
    , 'Pensioner' ,'Sea_Vojage_Gast', 'State_Steel_Ind,','Military_Service'], key=7) # dropped State_Steel_Ind
    CAR = st.sidebar.selectbox('CAR',options=['Car', 'Without_Vehicle', 'Car_and_Motor_bi'], key=8) # dropped Without_Vehicle
    CARDS = st.sidebar.selectbox("CARDS",options=['Cheque_card' ,'no_credit_cards', 'Mastercard_Euroc', 'VISA_mybank','VISA_Others'\
    , 'Other_credit_car', 'American_Express'], key=9) # dropped cheque card    


    button_clicked = st.button('Submit', key=28)    

    def update_variables():    
 
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

            #     

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

            
            no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
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
                no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
                ,Other_credit_car, American_Express = 0,0,0,0,0,0    


            
            inputs1 = [H,V, U, G, E, T,Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT,Lease,German, Turkish, RS, Greek ,Italian\
            , Other_European, Spanish_Portugue,Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
            , Pensioner ,Sea_Vojage_Gast, Military_Service,Car,Car_and_Motor_bi,no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
            , Other_credit_car, American_Express]
            
            inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS\
            , REGN, DIV, CASH]    

            list_ = inputs2 + inputs1

            inputs = pd.Series(list_)  
            
            prediction = Model_Perf.Prediction(GLM_Bino.GLM_Binomial_fit,inputs, train_test.X_train, train_test.Y_train)    

            st.subheader('Customer {} probability of default is: {}'.format(NAME , prediction))
            st.success('Successfully executed the model')
        

    update_variables()

def decision_fea():

    NAME = st.sidebar.text_input("Customer", key=90)
    AGE = st.sidebar.slider("AGE", 0,100,key=10)
    CHILDREN = st.sidebar.slider("CHILDREN", 0, 10,key=11)
    PERS_H = st.sidebar.slider("PERS_H", 0, 10,key=12)
    TMADD = st.sidebar.slider("TMADD", 0, 1000,key=13)
    TMJOB1 = st.sidebar.slider("TMJOB1", 0, 1000,key=14)
    TEL = st.sidebar.slider("TEL", 1, 10,key=15)
    NMBLOAN = st.sidebar.slider("NMBLOAN", 0, 10,key=16)
    FINLOAN = st.sidebar.slider("FINLOAN", 0, 10,key=17)
    INCOME = st.sidebar.slider("INCOME", 1, 1000000,100,key=18)
    EC_CARD = st.sidebar.slider("EC_CARD", 1, 10,1,key=19)
    INC = st.sidebar.slider("INC", 1, 1000000,100,key=20)
    INC1 = st.sidebar.slider("INC1", 1, 10,1,key=21)
    BUREAU = st.sidebar.slider("BUREAU", 1, 10,1,key=22)
    LOCATION = st.sidebar.slider("LOCATION", 1, 10,1,key=23)
    LOANS = st.sidebar.slider("LOANS", 1, 10,1,key=24)
    REGN = st.sidebar.slider("REGN", 1, 10,1,key=25)
    DIV = st.sidebar.slider("DIV", 1, 10,1,key=26)
    CASH = st.sidebar.slider("CASH", 1, 1000000,100,key=27)        

    # Categorical
    
    TITLE = st.sidebar.selectbox("TITLE", options=['H','R'], key=2)
    STATUS = st.sidebar.selectbox("STATUS",options=['V','U','G','E','T','W'], key=3)
    PRODUCT = st.sidebar.selectbox('PRODUCT',options=['Radio_TV_Hifi','Furniture_Carpet', 'Dept_Store_Mail', 'Leisure','Cars', 'OT'], key=4) # dropped Radio
    RESID = st.sidebar.selectbox('RESID',options=['Lease','Owner'], key=5) # dropped Owner
    NAT = st.sidebar.selectbox('NAT',options=['German', 'Turkish', 'RS', 'Greek' ,'Yugoslav', 'Italian', 'Other_European','Spanish_Portugue'], key=6) #dropped Yugoslavia
    PROF = st.sidebar.selectbox('PROF',options=['Others','Civil_Service_M' ,'Self_employed_pe', 'Food_Building_Ca','Chemical_Industr'\
    , 'Pensioner' ,'Sea_Vojage_Gast', 'State_Steel_Ind,','Military_Service'], key=7) # dropped State_Steel_Ind
    CAR = st.sidebar.selectbox('CAR',options=['Car', 'Without_Vehicle', 'Car_and_Motor_bi'], key=8) # dropped Without_Vehicle
    CARDS = st.sidebar.selectbox("CARDS",options=['Cheque_card' ,'no_credit_cards', 'Mastercard_Euroc', 'VISA_mybank','VISA_Others'\
    , 'Other_credit_car', 'American_Express'], key=9) # dropped cheque card    


    button_clicked = st.button('Submit', key=28)    

    def update_variables2():    
    
        if button_clicked:    

            W, V, U, G, E, T = 0,0,0,0,0,0    

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
                W=1  
   

            R, H = 0,0    

            if TITLE=='H':
                H = 1
            else:
                R=1
            
            
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
                Radio_TV_Hifi = 1   

            
            Owner, Lease = 0,0    

            if RESID=='Lease':
                Lease=1    

            else:
                Owner=1
            
            
            Yugoslav, German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue = 0,0,0,0,0,0,0,0    

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

            
            
            State_Steel_Ind, Others, Civil_Service_M , Self_employed_pe, Food_Building_Ca, Chemical_Industr\
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

            
            Without_Vehicle, Car,Car_and_Motor_bi= 0,0,0    

            if CAR=='Car':
                Car=1
            elif CAR=='Car_and_Motor_bi':
                Car_and_Motor_bi=1
            else:
                Without_Vehicle= 1    

            
            Cheque_card, no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others\
            ,Other_credit_car, American_Express = 0,0,0,0,0,0,0    

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

            
            inputs1 = [R, H, W, V, U, G, E, T,Radio_TV_Hifi, Furniture_Carpet, Dept_Store_Mail, Leisure,Cars, OT,Owner, Lease\
            ,Yugoslav, German, Turkish, RS, Greek ,Italian, Other_European, Spanish_Portugue,Others, Civil_Service_M ,State_Steel_Ind, Self_employed_pe\
            , Food_Building_Ca, Chemical_Industr, Pensioner ,Sea_Vojage_Gast, Military_Service,Without_Vehicle, Car,Car_and_Motor_bi\
            ,Cheque_card, no_credit_cards, Mastercard_Euroc, VISA_mybank,VISA_Others, Other_credit_car, American_Express]
            
            inputs2 = [CHILDREN, PERS_H, AGE, TMADD, TMJOB1, TEL, NMBLOAN, FINLOAN, INCOME, EC_CARD, INC, INC1, BUREAU, LOCATION, LOANS\
            , REGN, DIV, CASH]    

            list_ = inputs2 + inputs1

            inputs = pd.Series(list_)  
            
            prediction = Decision_tree.Predict_binary_DT(Decision_tree.DT_Classification_fit, inputs.values.reshape(1,-1), train_test1.Y_test\
                         , train_test1.X_train, train_test1.Y_train, randomstate=42, ccpalpha=Decision_tree.ideal_ccp_alpha)    

            st.subheader('Customer will default if value is 1 and not if 0: {}'.format( prediction))
            st.success('Successfully executed the model')
        

    update_variables2()

def get_classifier(name):
    
    if name=='Logistic':

        dataset_name = st.sidebar.selectbox('Select dataset', ('Logistic_KGB', 'Decision_KGB','Cluster'),key=29)
        visualization_name=st.sidebar.selectbox('Select Visuals', ('confusion','Cross_tab','Pivot','Clustering','ROC'\
        ,'confusion_matrix'), key=31)
        diagnostics_name=st.sidebar.selectbox('Select Diagnostic', ('Quantile Res','Breush_Pagan_Test','Normal_Residual_Test'\
        ,'Durbin_Watson_Test','Partial_Plots','Leverage_Studentized_Quantile_Res','Cooks_Distance_Quantile_Res'), key=32)
        get_dataset(dataset_name)
        get_dataset2(diagnostics_name)
        get_data(visualization_name)
        logistic_fea()

    else:

        dataset_name = st.sidebar.selectbox('Select dataset', ('Logistic_KGB', 'Decision_KGB','Cluster'),key=29)
        decision_name = st.sidebar.selectbox('Select Decision Tree',('Cross_Validate_Alphas','Ideal_Alpha','Confusion_matrix_plot_DT'\
        ,'Plot_DT'))
        get_dataset(dataset_name)
        get_data2(decision_name)
        decision_fea()

get_classifier(classifier_name)
