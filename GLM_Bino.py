
#if __name__ == '__main__':

import statsmodels.api as sm
import train_test
import scipy.stats
import pickle

# GLM FIT-BINOMIAL
# =================

def GLM_Binomial_fit(X_train, Y_train):
    
    '''GLM Binomial fit'''
    
    glm_binom = sm.GLM(Y_train.values.reshape(-1,1), X_train, family=sm.families.Binomial())   
    res = glm_binom.fit()
    
    return res.summary(), res

#m = (GLM_Binomial_fit(train_test.X_train, train_test.Y_train))[1]

#print(m.predict(train_test.X_test))
#print(dir(m))

#p_val = scipy.stats.chi2.pdf(2356, 1957)
#print(p_val)

#pickle.dump(GLM_Binomial_fit, open('model.pkl','wb'))