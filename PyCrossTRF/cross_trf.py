import pandas as pd
import numpy as np
from itertools import chain
from patsy import dmatrices
from statsmodels.api import OLS
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ValueWarning
from pandas.errors import SettingWithCopyWarning

# import h_block cross validation
# import h_block_cv

simplefilter('ignore', ValueWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)

class CTRF:
    def __init__(self, df: pd.DataFrame, dep: str, temp_r: str, pq_order: dict, covariates: dict, t_interval: int = 1000, **kwargs):
        """
        Initialize the Ctrf class.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing dependent, temperature, covariates, and date variables.
        - dep (str): Dependent variable (e.g., 'mortality').
        - temp_r (str): Temperature variable (e.g., 'temperature' in Fahrenheit or Celsius).
        - pq_order (dict): Order for temperature and covariates. Default is {'base': {'p': 4, 'q': 1}}.
        - covariates (dict): List of covariate variables with their scaling methods (e.g., {'time':'linear', 'income':'standard'}).
        - t_interval (int): Number of prediction points.
        - kwargs: Additional arguments.
        """
        self.df = df.copy().reset_index(drop=True)
        self.y = dep
        self.r = temp_r
        self.order = pq_order
        self.scaler = covariates.copy()
        self.temp_s_interval = t_interval

        additional_covariates = list(covariates.keys())
        additional_covariates.remove(self.r)
        self.x = additional_covariates.copy()
        
        self.df = self.df[[self.y] + [self.r] + self.x]
        
        self.trf = {}
        self.ctrf = {}
        self.mmt_r = None
        self.mmt_s = None
        
        self.reg_res_trf = {}
        self.reg_res_ctrf = {}
        self.spec_trf = ''
        self.spec_ctrf = ''
        self.df_reg_trf = pd.DataFrame()
        self.df_reg_ctrf = pd.DataFrame()
        self.df_pred_trf = pd.DataFrame()
        self.df_pred_ctrf = pd.DataFrame()




    def build_reg_spec(self, method={'trf', 'ctrf'}, verbose=False):
        """
        Build OLS specification.
        """
        order = self.order
        # 
        spec = f'{self.y} ~ '
        # iterate over order's variable key
        if method == 'trf' : covariates = [self.r]
        elif method == 'ctrf' : covariates = [self.r] + self.x
        # 
        for covariate in covariates :
            # iterate over the p and q order
            # p=0 represents intercept terms in the trf.
            for p in range(0, order['p']+1) :
                # leave out 'intercept' in the specification
                if (covariate == self.r) &  (p == 0) :
                    spec += ''
                elif (covariate == self.r) &  (p == 1) :
                    spec += f'{covariate}_{p}0'
                else :
                    spec += f' + {covariate}_{p}0'
            for q in range(1, order['q']+1) :
                spec += f' + {covariate}_0{q}c + {covariate}_0{q}s'
        # 
        if method == 'trf'  :
            self.spec_trf = spec
        elif method == 'ctrf' :
            self.spec_ctrf = spec
        # 
        if verbose : return spec

    def build_covariates_list(self, label_cov=[], label_temp=None, verbose=False):
        '''
        Build covariate list based on p,q order for variable selection in prediction
        '''
        if not label_temp : label_temp = self.r
        order = self.order
        prediction_cov = [label_temp] + label_cov
        # 
        lst_covariate = []
        for covariate in prediction_cov :
            for p in range(0, order['p']+1) :
                # leave out 'intercept' in the specification
                if (covariate == self.r) &  (p == 0) :
                    lst_covariate += ['Intercept']
                elif (covariate == self.r) &  (p == 1) :
                    lst_covariate += [f'{covariate}_{p}0']
                else :
                    lst_covariate += [f'{covariate}_{p}0']
            for q in range(1, order['q']+1) :
                lst_covariate += [f'{covariate}_0{q}c', f'{covariate}_0{q}s']
        return lst_covariate


    def pq_powering(self, order_p, order_q, temp_s, label, covariate_s=None, verbose=False):
        '''
        tranform 'temp_s' based on 'p', 'q' order.
        if 'covaraite_s' provided, multiplying it.
        '''
        if covariate_s is None : 
            covariate_s = 1
        df = pd.DataFrame()
        for p in range(0, order_p+1 ) :
            # the first constant column name as "intercept"
            if (p == 0) & (label == self.r):
            # if (p == 0) & (len(covariate_s) == 1):
                df.loc[:,f'Intercept'] = covariate_s * np.power(temp_s, p) 
            else:
                df.loc[:,f'{label}_{p}0'] = covariate_s * np.power(temp_s, p)
        # trignometric functions.
        for q in range(1, order_q+1 ) :
            df.loc[:,f'{label}_0{q}c'] = covariate_s * np.cos(temp_s * 2 * q *np.pi)
            df.loc[:,f'{label}_0{q}s'] = covariate_s * np.sin(temp_s * 2 * q *np.pi)
        # 
        if verbose :
            print(order_p, order_q, label)
            print(temp_s)
            print(covariate_s)
        # 
        return df



    def pram_check(self, check=None):
        '''
        class parameter check.
        '''
        print(self.spec_trf)



    def build_regression_df_trf(self, verbose=False):
        """
        Build regression df corresponding to the OLS specification.
        """
        # df=pd.DataFrame()
        temp_s = pd.DataFrame(self.normalizer(var= self.r, method=self.scaler[self.r]['scale']))
        df_var = self.pq_powering(self.order['p'], self.order['q'], temp_s=temp_s, label=self.r)
        df = pd.concat([self.df[[self.y]].copy(), df_var], axis=1, ignore_index=False)
        self.df_reg_trf = df
        # 
        if verbose : return df


    def build_prediction_df_trf(self, verbose=False):
        '''
        Create a prediction df - similar as the regression df except replacing temp_s with normalized value
        '''
        temp_pred = pd.DataFrame(np.arange(0,1,(1/self.temp_s_interval)), columns=[self.r])
        df = self.pq_powering(self.order['p'], self.order['q'], temp_s=temp_pred, label=self.r)
        self.df_pred_trf = df 
        # 
        if verbose : return df








    def estimate_trf(self, verbose=False):
        print(self.scaler)
        self.build_regression_df_trf()
        print(self.df_reg_trf)
        #
        self.build_reg_spec(method='trf')
        # 
        # run OLS
        y, x = dmatrices(self.spec_trf, self.df_reg_trf, return_type='dataframe')
        # 
        self.reg_res_trf = OLS(y,x).fit()
        # generate prediction table.
        # pred_temp_s = pd.DataFrame(np.arange(0,1,(1/self.temp_s_interval)), columns=[self.r])
        # df_pred = self._transform(temp_s=pred_temp_s, method='trf_pred')
        self.build_prediction_df_trf(verbose=verbose)
        trf_s = self.reg_res_trf.predict(self.df_pred_trf)
        self.mmt_s = np.argmin(trf_s) / self.temp_s_interval
        # update mmt
        if verbose :
            print(self.reg_res_trf.summary())
            print(self.mmt_s)
            print(np.argmin(trf_s))




    def build_regression_df_ctrf(self, verbose=False):
        """
        Build regression df corresponding to the OLS specification.
        """
        # df=pd.DataFrame()
        temp_s = pd.DataFrame(self.normalizer(var= self.r, method=self.scaler[self.r]['scale']))
        df_var = self.pq_powering(self.order['p'], self.order['q'], temp_s=temp_s, label=self.r)
        df = pd.concat([self.df[[self.y]].copy(), df_var], axis=1, ignore_index=False)
        # 
        # adding additional covariates
        for cov in self.x :
            cov_s = pd.DataFrame(self.normalizer(var=cov, method=self.scaler[cov]['scale'], temp=self.r,mmt=self.mmt_r))
            df_cov = self.pq_powering(self.order['p'], self.order['q'], temp_s=temp_s, label=cov, covariate_s=cov_s)
            df = pd.concat([df, df_cov], axis=1)
        # 
        self.df_reg_ctrf = df
        # 
        if verbose : return df
        # 



    def build_prediction_df_ctrf(self, covariate={'base':0}, verbose=False):
        '''
        Create a prediction df - similar as the regression df except replacing temp_s with normalized value
        '''
        temp_pred = pd.DataFrame(np.arange(0,1,(1/self.temp_s_interval)), columns=[self.r])
        df = self.pq_powering(self.order['p'], self.order['q'], temp_s=temp_pred, label=self.r)
        self.df_pred_ctrf = df
        # 
        print('baseTRF')
        if verbose : print(self.reg_res_ctrf.params[df.columns])
        label_covariates = self.build_covariates_list()
        pred_y = self.df_pred_ctrf[label_covariates].dot(self.reg_res_ctrf.params[label_covariates])
        # 
        if covariate.keys() != {'base':0}.keys() :
            print(f'{covariate}')
            label_covariates = self.build_covariates_list(label_cov=list(covariate.keys()))
            if verbose : print(label_covariates)
            df_pred_combined = self.df_pred_ctrf.copy()
            for ctrf_cov in list(covariate.keys()) :
                print(ctrf_cov)
                df_cov = self.pq_powering(self.order['p'], self.order['q'], temp_s=temp_pred, label=ctrf_cov,covariate_s=covariate[ctrf_cov])
                df_pred_combined = pd.concat([df_pred_combined, df_cov], axis=1)
            pred_y_ctrf = df_pred_combined[label_covariates].dot(self.reg_res_ctrf.params[label_covariates])
            return pred_y_ctrf   
            
        # 
        # if verbose : return df, pred_y
        # else: return pred_y



    def estimate_ctrf(self, verbose=False):
        # estimate trf first to find MMT.
        self.estimate_trf()
        print (f'MMT S: {self.mmt_s}')
        # build ctrf specification
        self.build_reg_spec(method='ctrf', verbose=verbose)
        # print(spec)
        # # 
        # construct df for regression
        self.build_regression_df_ctrf(verbose=verbose)
        # run OLS
        y, x = dmatrices(self.spec_ctrf, self.df_reg_ctrf, return_type='dataframe')
        # 
        # gen prediction talbe with covariate
        # need covariate order and name.
        # print('tt')
        # save ols result
        self.reg_res_ctrf = OLS(y,x).fit()
        if verbose : print(self.reg_res_ctrf.summary())
        #









    def normalizer(self, var=str, method={'minmax','linear','time','standard','quantile'},temp=None, mmt=None, show_df=False, verbose=False) :
        """
        Normalize the temperature and covariate variables.

        Parameters:
        show_df (bool): If True, print and return the DataFrame with normalized values.

        Returns:
        pd.DataFrame: DataFrame with normalized values if show_df is True.
        """
        df = self.df
        standarized = None      # innitialization
        # MMT is not provided, ordinary normalization or standardization
        if not mmt :
            # scikit-lean 'minmax_scale' based code. Transform X var to unit inverval.
            if method == 'minmax':
                print('Minmax normalization')
                # standarized = ( df[var] - df[var].min() ) / (df[var].max() - df[var].min())
                mms = MinMaxScaler()
                standarized = mms.fit_transform(df[[var]])
                if verbose :
                    print(f'min: {mms.data_min_.mean():2f}, max: {mms.data_max_.mean():2f}')
                    

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

            elif method == 'time':
                # use index
                print('time w/o mmt')
                scaler = df[var].index.max() 
                standarized = df[var].index / scaler
                if verbose :
                    print(f'min: {df[var].min()}, max: {df[var].max()}')

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
            elif method == 'time':
                # use index
                print('time w/ mmt')
                scaler = df[var].index.max() 
                standarized = df[var].index / scaler
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



    # def h_block_cv(self):
        


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


















































