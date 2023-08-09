# ============================================
# Point Biserial Test for Binary vs Numerical
# ============================================

# =====
# Plot
# =====

def Point_Biserial_Plot(independent, target):
    
    sns.set_theme(style="ticks", color_codes = True)
    data = pd.concat([independent, target], axis=1)
    
    return sns.catplot(x = independent, y = target, kind="box", data = data)    

# =========
# The Test
# =========

def Point_Biserial_Test_Binary(independent, target):
    
    '''Point Biserial Test for Binary vs Numerical varaibales'''
    
    df_loan_float_po_bi = pd.concat([independent, target], axis=1)
    df_loan_float_po_bi_GB = df_loan_float_po_bi.groupby(target.name)
    
    df_0 = df_loan_float_po_bi_GB.get_group(0)
    k = df_0[independent.name].mean()
    
    df_1 = df_loan_float_po_bi_GB.get_group(1)
    j = df_1[independent.name].mean()
    
    standard_dev_AGE = df_loan_float_po_bi[independent.name].std()
    mean_AGE = df_loan_float_po_bi[independent.name].mean()

    proportion_GB_1 = df_1[target.name].count()/df_loan_float_po_bi[target.name].count()
    proportion_GB_0 = 1-proportion_GB_1

    r_po_bi = ((j-k)/standard_dev_AGE)*sqrt((independent.shape[0]*proportion_GB_0)\
               *(1-proportion_GB_0)/(independent.shape[0]-1))

    #Test statistic

    t_po_bi = r_po_bi/sqrt((1-r_po_bi**2))
    
    return t_po_bi, 

Point_Biserial_test = Point_Biserial_Test_Binary(independent=df_loan["AGE"], target = df_loan["GB"])
Point_Biserial_test

# =========================
# python point_biserial API
# =========================

r = scipy.stats.pointbiserialr(df_loan["AGE"], df_loan["GB"])
print(r)

# =====================================================
# Categorical vs Categorical Chi-square test and plots
# =====================================================

# ======
# Plots
# ======

def Categorical_Crosstab_Plot(independent, target):
    
    '''Plot cross tab'''

    h = pd.crosstab(target,independent, normalize="columns")
    bar = plt.bar(target, independent)
    return plt.show(), h

Plot_categorical_cross_tab = Categorical_Crosstab_Plot(independent, target)
#print(Plot_categorical_cross_tab)

def Categorical_Pivot_Plot(independent, target):
      
    '''Categorical Plot for greater than 2 categories'''
    
    df = pd.concat([independent, target], axis=1) 
    df_pivot = pd.pivot_table(df, index=independent.name, columns=target.name, aggfunc=len, fill_value=0)\
                                                                       .apply(lambda x: x/float(x.sum()))
    
    return df_pivot.plot(kind="bar"), df_pivot

Categorical_Pivot_Plot = Categorical_Pivot_Plot(independent, target)
#print(Categorical_Pivot_Plot)

# ======
# Tests
# ======

def Chi_Square_Categorical_Test(independent, target):
    
    '''Chi Square test for Categorical Variables'''

    h_chi = pd.crosstab(independent, target)
    chi_val, p_val, dof, expected = chi2_contingency(h_chi)
    return chi_val, p_val

Chi_Square_Cat_Test = Chi_Square_Categorical_Test(independent, target)

# =================================================
# Pearson correlation test for Numerical variables
# =================================================

# =====
# Plot
# =====

def Scatter_Plot(independent, target):
    
    '''Scatter plot between numerical variables'''
    
    scatter = plt.scatter(target, independent)
    return plt.show()   

# =====
# Test
# =====

def Pearson_Correlation_Test(independent, target):
    
    '''Pearson correlation test function'''
    
    pearson_coef, p_value = stats.pearsonr(independent, target)
    return pearson_coef, p_val

# ================================
# Multicollinearity investigation
# ================================

# ================================================
# INPUT FLOAT DATAFRAME OF INDEPENDENT VARAIABLES
# ================================================

def Correlation_Plot(dataframe):
    
    '''Independent variables correlation plot'''

    return dataframe.corr()

Corr_Plot = Correlation_Plot(dataframe)

def Correlation_Value_Greater(corr_threshold, dataframe):
    
    dataframe_corr = Correlation_Plot(dataframe)
    g = []

    for i in range(dataframe_corr.shape[0]):
        for j in range(dataframe_corr.shape[0]):
            
            if (dataframe_corr.iloc[i,j]>corr) or (dataframe_corr.iloc[i,j]<-corr):
                g.append(dataframe_corr.iloc[i,j])
    
    return g        

print(Correlation_Value_Greater(corr_threshold, dataframe))

def get_Indexes(dataframe, value):
    
    list_of_pos = []
    result = dataframe.isin([value])
    series_obj = result.any()
    column_names = list(series_obj[series_obj == True].index)
    
    for col in column_names:    
        rows = list(result[col][result[col] == True].index)
        
        for row in rows:
            list_of_pos.append((row,col))
    
    return list_of_pos

print(get_Indexes(dataframe, value))

def get_Variables_for_Corr_Greater(corr_thereshold, dataframe, value):
       
    dataframe_corr = Correlation_Plot(dataframe)
    g_1 = Correlation_Value_Greater(corr_thereshold, dataframe)
    list_of_pos_1 = get_Indexes(dataframe, value)

    u = []

    for i in g_1:
        t = getIndexes(dataframe_corr, i)
        u.append([item for item in t if item[0]!=item[1]])
        
    return u

print(get_Variables_for_Corr_Greater(corr_thereshold, dataframe, value))

# =========
# VIF Test
# =========

def VIF_value(dataframe, target):
    
    '''Calculate variance inflation factor'''

    ols = statsmodels.regression.linear_model.OLS(target, dataframe.drop(labels=[target.name]), axis=1)
    res_ols = ols.fit()              
    VIF = 1/(1-res_ols.rsquared_adj**2)
    
    return VIF

print(VIF_value(dataframe, target))