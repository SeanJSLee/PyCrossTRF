print('cross_trf')

import pandas as pd
import numpy as np
from itertools import chain

# OLS estimation
# import patsy as pat
from patsy import dmatrices
# import statsmodels.api as sm
from statsmodels.api import OLS

# sklearn scalers
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

# import warnings
from warnings import simplefilter

from statsmodels.tools.sm_exceptions import ValueWarning
simplefilter('ignore', ValueWarning)

from pandas.errors import SettingWithCopyWarning
simplefilter(action="ignore", category=SettingWithCopyWarning)




# ctrf class
#   normalization
#   estimation : base
#   find MRT: mim(mortality) temp
#   estimation : cTRFs
#   calculate h-block cv


class CTRF :

    def __init__(self, df=pd.DataFrame(), y=str, r=str, order=dict(), t_inverval=1000, **kwargs):
        """
        Initialize the Ctrf class.

        Parameters:
        df (pd.DataFrame): DataFrame containing dependent, temperature, covariates, and date variables.
        y (str): Dependent variable.
        r (str): Temperature variable.
        x (list): List of covariate variables.
        date (str): Date variable.
        order (dict): Order for temperature and covariates. Default is {'base': {'p': 4, 'q': 1}}.
        n (int): Number of prediction points.
        kwargs: Additional arguments.
        """
        self.y = y
        self.r = r
        # all covariates in the 'order'
        covariates = list(order.keys())
        # remove temp in the list
        covariates.remove(self.r)
        self.x = covariates.copy()
        

        self.df = df[[y]+[r]+self.x].copy().reset_index(drop=True)
        
        # self.date = date
        self.order = order
        # self.n = n

        # save trf model (statsmodels)
        self.trf = {}
        # save ctrf model (statsmodels)
        self.ctrf = {}

        # store scale information whild normalization.
        self.scaler = {}
        self.temp_s_interval = t_inverval
        self.mmt_r = None
        self.mmt_s = None
        # regression result
        self.reg_res_trf = {}
        self.reg_res_ctrf = {}
        # regression spec
        self.spec_trf = ''
        self.spec_ctrf = ''
        # df for regression
        self.df_reg = pd.DataFrame()
        # self.df_temp_arry = pd.DataFrame({'place_holder': [1.0] * self.n})
        self.df_ctrf = pd.DataFrame()
        # df for recovering trf and ctrf
        self.df_pred_trf = pd.DataFrame()
        self.df_pred_ctrf = pd.DataFrame()






    # construct covariates while normalizing (standardizing)
    # Covariates builder
    def build_regression_df(self, method={'trf', 'ctrf'}, verbose=False):
        # check 'df_reg' is empty, if it is empty, copy 'y' and transform 'temp' as 'base'

        # build 'y'
        self.df_reg = self.df[[self.y]].copy()
        # normalize 'temp'
        # temp_r = self.df[self.r]
        temp_s = pd.DataFrame(self.normalizer( var=self.r, method=self.order[self.r]['scale']), columns=[self.r])
        # temp_s = self.df[self.r]
        # construct df_reg
        if method == 'trf' :
            self.df_reg = self._transform(temp_s=temp_s, method='trf', verbose=verbose)

        # Estimate TRF in order to get MMT.
 

        # if mmt_normalization :
            # 'base_' is temperature.
            # Covariates are interected with transformed 'temp'
            # 'base_00' => Intercept.
            # copy y
            
        # based on order information build maximum order combination
        
        # return temp_s
      
        # # 
        # if verbose :a
        return self.df_reg


    def build_reg_spec(self, method={'trf', 'ctrf'}, verbose=False):
        if method == 'trf'  :
            order = {self.r:self.order[self.r]}
        elif method == 'ctrf' :
            order = self.order
        # 
        spec = f'{self.y} ~ '
        # iterate over order's variable key
        for covariate in order.keys() :
            # iterate over the p and q order
            # p=0 represents intercept terms in the trf.
            for p in range(0, order[covariate]['p']+1) :
                # leave out 'intercept' in the specification
                if (covariate == self.r) &  (p == 0) :
                    spec += ''
                elif (covariate == self.r) &  (p == 1) :
                    spec += f'{covariate}_{p}0'
                else :
                    spec += f' + {covariate}_{p}0'
            for q in range(1, order[covariate]['q']+1) :
                spec += f' + {covariate}_0{q}c + {covariate}_0{q}s'
        if method == 'trf'  :
            self.spec_trf = spec
        elif method == 'ctrf' :
            self.spec_ctrf = spec
        return spec




    def estimate_trf(self, verbose=False):
        # 
        # build data for estimation
        self.df_reg = self.df[[self.y]].copy()
        temp_s = pd.DataFrame(self.normalizer( var=self.r, method=self.order[self.r]['scale']), columns=[self.r])
        self.df_reg = self._transform(temp_s=temp_s, method='trf', verbose=verbose)
        #
        spec = self.build_reg_spec(method='trf')
        print(spec)
        # 
        # run OLS
        y, x = dmatrices(self.spec_trf, self.df_reg, return_type='dataframe')
        # 
        self.reg_res_trf = OLS(y,x).fit()
        # generate prediction table.
        pred_temp_s = pd.DataFrame(np.arange(0,1,(1/self.temp_s_interval)), columns=[self.r])
        df_pred = self._transform(temp_s=pred_temp_s, method='trf_pred')
        trf_s = self.reg_res_trf.predict(df_pred)
        self.mmt_s = np.argmin(trf_s) / self.temp_s_interval
        # update mmt
        if verbose :
            print(self.reg_res_trf.summary())
            print(self.mmt_s)
        return trf_s



    def estimate_ctrf(self, verbose=False):
        # estimate trf first to find MMT.
        self.estimate_trf()
        print (f'MMT S: {self.mmt_s}')
        # build ctrf specification
        spec = self.build_reg_spec(method='ctrf')
        print(spec)
        # # 
        # # construct df for regression
        # temp_s = pd.DataFrame(self.normalizer( var=self.r, method=self.order[self.r]['scale']), columns=[self.r])
        # print(self.x)
        # df_reg = self._transform(temp_s=temp_s, covariate= self.x, method='ctrf')
        # run OLS
        # y, x = dmatrices(self.spec_ctrf, self.df_reg, return_type='dataframe')
        # # 
        # # gen prediction talbe with covariate
        # # need covariate order and name.
        # print('tt')
        # # save ols result
        # self.reg_res_ctrf = OLS(y,x).fit()
        # return df_reg





    # def run_ols(self, model={'trf','ctrf'} ):
    #     #
    #     if model == 'trf' :
    #         order = {self.r:self.order[self.r]}
    #         # covariate = 1
    #     elif model == 'ctrf' :
    #         order = self.order
    #     # 
    #     spec = f'{self.y} ~ '
    #     # iterate over order's variable key
    #     for covariate in order.keys() :
    #         # iterate over the p and q order
    #         # p=0 represents intercept terms in the trf.
    #         for p in range(0, order[covariate]['p']+1) :
    #             if (covariate == self.r) &  (p == 0) :
    #                 spec += ''
    #             else :
    #                 spec += f' + {covariate}_{p}0'
    #         for q in range(1, order[covariate]['q']+1) :
    #             spec += f' + {covariate}_0{q}c + {covariate}_0{q}s'
    #     print(spec)
    #     # 
    #     # run OLS
    #     y, x = dmatrices(spec, self.df_reg, return_type='dataframe')
       
    #     # 

    #     if model == 'trf':
    #         self.reg_res_trf = OLS(y,x).fit()
    #         # generate prediction table.
    #         pred_temp_s = pd.DataFrame(np.arange(0,1,(1/self.temp_s_interval)), columns=[self.r])
    #         df_pred = self._transform(temp_s=pred_temp_s, method='trf_pred')
    #         trf_s = self.reg_res_trf.predict(df_pred)
    #         self.mmt_s = np.argmin(trf_s) / self.temp_s_interval
    #         print(self.reg_res_trf.summary())
    #         print(self.mmt_s)
    #         return trf_s
        
    #     elif model == 'ctrf' :
    #         # gen prediction talbe with covariate
    #         # need covariate order and name.
    #         print('tt')
    #         # save ols result
    #         self.reg_res_ctrf = OLS(y,x).fit()

        




    """
    Normalize the temperature and covariate variables.

    Parameters:
    show_df (bool): If True, print and return the DataFrame with normalized values.

    Returns:
    pd.DataFrame: DataFrame with normalized values if show_df is True.
    """
    def normalizer(self, var=str, method={'minmax','linear','standard','quantile'},temp=None, mmt=None, show_df=False, verbose=False) :
        df = self.df
        standarized = None      # innitialization
        # MMT is not provided, ordinary normalization or standardization
        if not mmt :
            # scikit-lean 'minmax_scale' based code. Transform X var to unit inverval.
            if method == 'minmax':
                print('Minmax normalization')
                # standarized = ( df[var] - df[var].min() ) / (df[var].max() - df[var].min())
                mt = MinMaxScaler()
                standarized = mt.fit_transform(df[[var]])
                if verbose :
                    print(f'min: {mt.data_min_.mean():2f}, max: {mt.data_max_.mean():2f}')
                    

            # run L1 normalization - make a vectro to a unit norm.    
            elif method == 'linear':
                # shifter that start from 0
                shifter = -df[var].min()
                scaler = df[var].max() - df[var].min()
                standarized = (df[var] + shifter) / scaler
                if verbose :
                    print(f'min: {df[var].min()}, max: {df[var].max()}')
                # lt = Normalizer(norm='l1')
                # standarized = lt.fit_transform(df[[var]])
                # if verbose :
                #     print(lt.n_features_in_)

            # ordinary standardization
            elif method == 'standard' :
                # standarized = (df[var] - df[var].mean()) / np.std(df[var],ddof=1)
                st = StandardScaler()
                standarized = st.fit_transform(df[[var]])
                if verbose :
                    print(f'E({var}): {st.mean_[0]:.2f}, SD({var}): {np.sqrt(st.var_[0]):.2f}')

            elif method == 'quantile' :
                print('quantile')
                # quantile transformer
                qt = QuantileTransformer(n_quantiles=100)
                standarized = qt.fit_transform(df[[var]])
                if verbose :
                    print(qt.n_quantiles_)
                    print(qt.quantiles_)
                    # print(qt.references_)
        # When MMT is provided, standardize while centering at MMT.
        elif mmt :
            if method == 'minmax':
                print('minmax with MMT: TBD')
            
            elif method == 'linear':
                # shifter that start from 0 (the same as the TRF w/o MMT)
                print('linear w/ mmt')
                shifter = -df[var].min()
                scaler = df[var].max() - df[var].min()
                standarized = (df[var] + shifter) / scaler
                if verbose :
                    print(f'min: {df[var].min()}, max: {df[var].max()}')

            elif method == 'standard':
                print('Standardize centering at MMT - set sample mean of X as X value at the MMT')
                # slice interval near the MMT, find closest two temp's location
                temp_minimax = df.loc[df[temp] >= mmt][temp].min()
                temp_maximin = df.loc[df[temp] <= mmt][temp].max()

                cov_minimax = df.loc[df[temp]==temp_minimax][var].mean()
                cov_maximin = df.loc[df[temp]==temp_maximin][var].mean()

                if temp_minimax == temp_maximin :
                    weighted_avg = cov_minimax
                else :
                    weighted_avg = np.average([cov_minimax, cov_maximin], 
                                        #  inverse distance weight.
                                          weights=[temp_minimax/(abs(temp_maximin - temp_minimax)),
                                                   temp_maximin/(abs(temp_maximin - temp_minimax))])
                # sample variace
                sigma = (((df[var] - weighted_avg)**2).sum() / (len(df[var]-1))) ** .5
                # standariazed covariate vector
                standarized = (df[var] - weighted_avg) / sigma
                if verbose :
                    print(f'E({var}): {weighted_avg:.2f}, SD({var}): {sigma:.2f}, MMT:{mmt:.2f}, Cov Minimax: {cov_minimax:.2f}, Cov Maximin: {cov_maximin:.2f}')
            
            elif method == 'quantile':
                print('quantile with MMT: TBD')
        if verbose :
            return df, standarized
        else :
            return standarized


    def transform(self, type={'trf', 'ctrf'}, prediction=False, order=None):
        if type == 'trf':
            vars = [self.r]
            if not order : order = {self.r:self.order[self.r]}
            # covariate = 1
            # do powering
            # df = self.normalizer(var= self.r, method=order[self.r]['scale'])
        
        elif type == 'ctrf' :
            vars = [self.r] + self.x
            if not order : order = self.order
            # for var in vars :
            
            #     df[:,f'{var}'] = self.normalizer(var=var, method=self.order[var]['scale'], mmt=self.mmt_r)

        if not prediction :df = self.df[[self.y]].copy()
        elif   prediction :df = pd.DataFrame()

        for var in vars :
            # 
            temp_s = pd.DataFrame(self.normalizer(var= self.r, method=order[self.r]['scale']))
            # print(temp_s)
            if var == self.r : covariate = 1
            else : covariate = self.normalizer(var=var, method=self.order[var]['scale'], mmt=self.mmt_r)
            
            # 
            df_var = self.pq_powering(order[var]['p'], order[var]['q'], temp_s=temp_s, label=var, covariate_s=covariate)
            df = pd.concat([df, df_var], axis=0, ignore_index=True)
        return df, df_var




    def pq_powering(self, order_p, order_q, temp_s, label, covariate_s=None):
        print(order_p, order_q, label)
        print(temp_s)
        print(covariate_s)
        if not covariate_s : covariate_s = 1
        df = pd.DataFrame()
        for p in range(0, order_p+1 ) :
            # the first constant column name as "intercept"
            if (p == 0) & (covariate_s == 1):
                df.loc[:,f'Intercept'] = covariate_s * np.power(temp_s, p) 
            else:
                df.loc[:,f'{label}_{p}0'] = covariate_s * np.power(temp_s, p)
        # trignometric functions.
        for q in range(1, order_q+1 ) :
            df.loc[:,f'{label}_0{q}c'] = covariate_s * np.cos(temp_s * 2 * q *np.pi)
            df.loc[:,f'{label}_0{q}s'] = covariate_s * np.sin(temp_s * 2 * q *np.pi)
        return df
         


        






        # Construct DataFrame for regression while transforming data.
    def _transform(self, temp_s=pd.DataFrame(), covariate=None, method={'trf', 'trf_pred','ctrf', 'ctrf_pred'}, verbose=False) :
        # Initialize 'base' at original data which is a column of 1.
        # make easy to transform - 'base' is not in the data but in the pq order.
        # 'base' pretent as a dummy for temp itself in interation term.
        # Covariates are interected with transformed 'temp'
        # 'base_00' => Intercept.
        # instead of 'base' use order key
        if  method == 'trf':
            order = {self.r:self.order[self.r]}
            covariate = 1
            df = self.df_reg.copy()
        elif method == 'trf_pred' :
            order = {self.r:self.order[self.r]}
            covariate = 1
            df = pd.DataFrame()
        elif method == 'ctrf' :
            print('tt')
            order = self.order.copy().pop(self.r)
            # normalization - covaraite
        elif method == 'ctrf_pred' :
            order = self.order.copy().pop(self.r)
            df = temp_s.copy()
        # # 
        for var in order.keys() :
            # print(var)
            # poly power
            for p in range(0, order[var]['p']+1 ) :
                # self.df_reg.loc[:,f'{var}_{p}0'] = self.df[var] * np.power(self.df[self.temp], p)
                if (var == self.r) & (p == 0) :
                    df.loc[:,f'Intercept'] = covariate * np.power(temp_s, p) 
                else:
                    df.loc[:,f'{var}_{p}0'] = covariate * np.power(temp_s, p)
            # trignometric functions.
            for q in range(1, order[var]['q']+1 ) :
                # self.df_reg.loc[:,f'{var}_0{q}c'] = self.df[var] * np.cos(self.df[self.temp] * 2 * q *np.pi)
                df.loc[:,f'{var}_0{q}c'] = covariate * np.cos(temp_s * 2 * q *np.pi)
                # self.df_reg.loc[:,f'{var}_0{q}s'] = self.df[var] * np.sin(self.df[self.temp] * 2 * q *np.pi)
                df.loc[:,f'{var}_0{q}s'] = covariate * np.sin(temp_s * 2 * q *np.pi)
                # print(q)
        return df




















# #######################################
# estm_ctrf[fips] = ctrf( df = df_base.loc[df_base.fips == fips],
#                         y = 'daily_death100k',
#                         temp = 'TMEAN',
#                         # set empty to run 'TRF'
#                         x = [],
#                         date = 'date', 
#                         order = pq_order)
# estm_ctrf[fips]._normalize(show_df=True)
# estm_ctrf[fips]._transform(show_df=True)
# estm_ctrf[fips]._run_ols(show_df=True)
# estm_ctrf[fips]._gen_temp_array(show_df=True)
# estm_ctrf[fips]._recover_ctrf_base()    
# #######################################


















































