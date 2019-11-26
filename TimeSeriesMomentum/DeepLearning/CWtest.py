import pandas as pd
import statsmodels.api as sm
#import statsmodels.formula.api as smf
from scipy.stats import norm
#==============================================================================
# Define a function to calculate CW test
#==============================================================================
def CWtest(r_real, r_hat, r_bar,lag):
    f=(r_real-r_bar)**2-((r_real-r_hat)**2-(r_bar-r_hat)**2)
    f = f.astype('float') # BUG1
    f=pd.DataFrame(f,columns=['f'])
    f=sm.add_constant(f)
    f=f.dropna()
    result = sm.OLS(f['f'],f['const']).fit(cov_type='HAC',cov_kwds={'maxlags':lag})
    tstats = result.tvalues[0]
    pval = 1-norm.cdf(tstats)
    return tstats, pval
