import pandas as pd
import numpy as np
from typing import Any
# from itertools import chain
# from patsy import dmatrices
from statsmodels.api import OLS
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ValueWarning
from pandas.errors import SettingWithCopyWarning


from .utils import ctrf_utils

simplefilter('ignore', ValueWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)

class CTRF:
    def __init__(self, df: pd.DataFrame, dep: str, temp_r: str, pq_order: dict, cov_scale: dict, std_interval: int = 1000):
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
        self.y : pd.Series  = df[dep].copy()            # pd.Series
        self.r : pd.Series  = df[temp_r].copy()         #   same
        self.x_raw  = {}
        for cov_x in [cov for cov in cov_scale.keys() if cov != self.r.name] :
            self.x_raw[f'{cov_x}'] = df[cov_x].copy()

        self.pq_order = pq_order
        self.cov_scale = cov_scale
        self.std_interval = std_interval
        

        self.s = pd.Series()        # normalizaed temperature
        self.s_pred = pd.Series()     # temp range to recover TRF and CTRFs
        self.x_s = {}

        self.Xs = pd.DataFrame()            # p,q powered temp
        self.Xs_ctrf = pd.DataFrame()       # p,q powered temp and covariates
        
        self.mmt_r : float = None           
        self.MMT_s : float = None                   # MMT in the way of normlized as normalization method.
        
        self.reg_res_trf = {}              #               CTRF
        self.reg_res_ctrf = {}              #               CTRF

        self.Xs_pred = pd.DataFrame()   # p,q powered temp based on 'self.temp_pred' and p, q order
        self.Xs_ctrf_pred = pd.DataFrame()  #       the same for CTRF.
        self.normlized_vars = {}
        
        self.recovered_trf = pd.DataFrame()
        self.recovered_ctrf = {}




    def pre_processing(self, prep_for:str = {'trf','ctrf'}, verbose=False):
        if prep_for == 'trf':
            # estimation variable, normalization etc.
            self.s = ctrf_utils().normalizer(x = self.r, 
                                             method = self.cov_scale[self.r.name]['scale'])
            
            self.Xs = ctrf_utils().gen_df_xs(temp_s=self.s, pq_order=self.pq_order)
            # 
            # gen data for recover TRF and MMT.
            self.s_pred     = pd.Series(np.arange(0, 1, 1/self.std_interval), name=self.r.name)
            self.Xs_pred    = ctrf_utils().pq_powering(temp_s=self.s_pred, pq_order=self.pq_order)
            # 
            if verbose : print('self.xs \n',self.s, '\nself.Xs_pred \n', self.Xs_pred)

        elif prep_for == 'ctrf':
            # construct ctrf dataframe
            self.Xs_ctrf = self.Xs
            # adding normalizaed variable for ctrf
            for cov_x in self.x_raw.keys() :
                # normalization
                self.x_s[f'{cov_x}'] = ctrf_utils().normalizer(
                                                    x=self.x_raw[cov_x], 
                                                    method=self.cov_scale[cov_x]['scale'],
                                                    temp= self.s,
                                                    MMT=self.MMT_s
                                                )
                # gen variable with the pq powered the temp var.
                self.Xs_ctrf = pd.concat([  self.Xs_ctrf,
                                            ctrf_utils().pq_powering(   
                                                temp_s= self.s, 
                                                pq_order= self.pq_order, 
                                                covariate_s= self.x_s[f'{cov_x}']
                                            )
                                        ],
                                            axis=1
                                        )
            #
            if verbose : print(self.Xs_ctrf)

        if verbose : print(f'pre-processing done - {prep_for}')



    
    def estimate_trf(self, verbose=False):
        # prep for estimation
        self.pre_processing(prep_for='trf', verbose=verbose)

        # ys = self.df[[self.y]]
        
        # self.xs = ctrf_utils().gen_df_xs(temp_s=self.temp_s, pq_order=self.order)
        # 
        # Run OLS
        self.reg_res_trf = OLS(self.y,self.Xs).fit()
        # 
        # generate prediction table.
        # 
        self.Xs_pred = ctrf_utils().pq_powering(temp_s=self.s_pred, pq_order=self.pq_order)
        # 
        if verbose : print (self.Xs_pred)
        # 
        # find MMT
        self.recovered_trf = self.reg_res_trf.predict(self.Xs_pred)
        # update MMT
        self.MMT_s = np.argmin(self.recovered_trf) / self.std_interval    # need to use saved scale to recover it
        if verbose : print(f'MMT: {self.MMT_s} \n',self.reg_res_trf.summary())




    def estimate_ctrf(self, verbose=False):
        # estimate trf first to find MMT.
        if self.MMT_s is None :
            print('TRF re-estimate')
            self.estimate_trf()
        
        self.pre_processing(prep_for='ctrf', verbose=verbose)

        # run OLS
        # save ols result
        self.reg_res_ctrf = OLS(self.y, self.Xs_ctrf).fit()

        if verbose : print(self.reg_res_ctrf.summary(), '\n', self.reg_res_ctrf.params)
        #


                


# class for recovering CTRFs
class CTRF_recover :
    def __init__(self) -> None:
        pass




    def recover_ctrf(  self, ctrf_pred_lst : list, s_pred : pd.Series, coef : pd.Series, pq_order : dict, verbose=False ):
        '''
        generate list of dict that contatining information to recover CTRF
        ctrf_pred_lst = [{'time':[.5, 1.0]},{'income':[-1.0, 0.0, 3.0]},{'age':[-2.0, 1.0, 3.0]}]

        such as : {
                    'temp': {'multiple': [1.0]}, 
                    'time': {'multiple': [0.5, 1.0]},
                    'income': {'multiple': [0.5, 1.0]}
                    }
        '''
        # 'base CTRf'
        ctrf = {}
        ctrf['base'] = self.calc_ctrf(s_pred=s_pred, coef=coef, pq_order=pq_order)

        # for other covariates
        for elm in ctrf_pred_lst :
            cov = list(elm.keys())[0]
            # iter over muliplication value.
            for cov_multiple in elm[cov] :
                ctrf[f'{cov} {cov_multiple:.4f}'] = self.calc_ctrf(s_pred, coef, pq_order, cov, cov_multiple)
                if verbose : print(cov, cov_multiple)

        return ctrf
            
            




    def calc_ctrf(  self, s_pred : pd.Series, coef : pd.Series, pq_order:dict, cov:str = None, muliple:float = 1.0):
        
        # gen prediction df for ctrf
        # if ctrf is not provided, estimate the baseCTRF.
        if cov is not None or cov == s_pred.name :
            ctrf_pred = pd.Series( [muliple]*len(s_pred.index), name=cov )
        else : ctrf_pred = None
        
        # gen df for recovering ctrf.
        df = ctrf_utils().pq_powering(temp_s=s_pred,
                                        pq_order=pq_order,
                                        covariate_s=ctrf_pred)
        
        # recover ctrf based on normalized temp.
        ctrf_recov = df.dot(coef[df.columns])

        return ctrf_recov
                























































