import os
import pandas as pd
import numpy as np
from statsmodels.api import OLS

from typing import Dict, Optional, Union, Literal, Tuple, Any

from .utils import Normalizer, pq_powering, gen_df_xs

# from statsmodels.tools.sm_exceptions import ValueWarning
# from warnings import simplefilter
# from pandas.errors import SettingWithCopyWarning
# simplefilter('ignore', ValueWarning)
# simplefilter(action="ignore", category=SettingWithCopyWarning)

class CTRF:
    def __init__(self, 
                df          : pd.DataFrame, 
                dep         : str, 
                temp_r      : str,
                covariates  : Optional[list[str]] = None, 
                covariates_logged  : Optional[list[str]] = None, 
                pq_order    : Dict[str,int] = {'p':4, 'q':2}, 
                scale       : Dict[str,str] = {'temp_q':'raw'}, 
                std_interval: int = 1000,
                time_id     : str = 'date',
                cross_id    : str = 'fips',
                 ):
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
        # 
        if not covariates:
            self.model:Literal['trf','ctrf'] = 'trf'
        else :
            self.model:Literal['trf','ctrf'] = 'ctrf'
        # 
        self.df = df
        self.y : pd.Series  = df[dep]            # pd.Series
        self.r : pd.Series  = df[temp_r]         #   same
        self.x_raw  = {}
        # if self.model == 'ctrf':
        if covariates is None :
            self.covariates = {}
        else :
            self.covariates = {k:None for k in covariates}             # {'original name':'normalized name'} 
            for cov in covariates : # pyright: ignore[reportOptionalIterable]
                self.x_raw[f'{cov}'] = df[cov]
        self.covariates_logged = covariates_logged
        self.time_id    = time_id        # date variable
        self.cross_id   = cross_id            # id variable (cross sectional id)
        self.pq_order   = pq_order
        self.scale      = scale
        self.std_interval = std_interval           
        # 
        # reserved object to organize
        self.s = pd.Series()            # normalizaed temperature
        self.s_pred = pd.Series()       # temp range to recover TRF and CTRFs
        self.s_cov = {}                   # 'ctrf' normalized covariates
        # 
        self.Xs = pd.DataFrame()        # p,q powered temp
        self.Xs_ctrf = pd.DataFrame()   # p,q powered temp and covariates
        # date
        self.X_date = pd.DataFrame()
        
        self.mmt_r : Optional[float] = None           
        self.mmt_s : Optional[float] = None            # MMT in the way of normlized as normalization method.
        
        self.reg_res_trf = {}           #  TRF regression result
        self.reg_res_ctrf = {}          # CTRF regression result
        
        self.reg_spec_var_trf = []      # regression specification (slicing dataframe)
        self.reg_spec_var_ctrf = []


        self.Xs_pred = pd.DataFrame()   # p,q powered temp based on 'self.temp_pred' and p, q order
        self.Xs_ctrf_pred = pd.DataFrame()  #       the same for CTRF.
        self.normlized_vars = {}
        
        self.recovered_trf = pd.DataFrame()
        self.recovered_ctrf = pd.DataFrame()
        print(f'''
                    {self.model.upper()} model initiated with:
                    * Dep var: {dep}
                    * Temp var: {temp_r}, normalization: {scale[temp_r]}
                                ''')




   
    def estimate_trf(self, verbose:bool=False, regres:bool=False):
        # estimation variable, normalization etc.
        method:str          = self.scale[self.r.name]   # type: ignore
        temp_var_name:str   = self.r.name               # type: ignore
        # 
        if method == 'raw' :
            self.s = self.r.copy()
        else :
            transformed_df = Normalizer().normalizer(trans_var = temp_var_name,
                                                    method     = method,        # type: ignore
                                                    df         = self.df,
                                                    cross_id   = self.cross_id )
            self.df = pd.concat([self.df,transformed_df], axis=1)
            self.s  = self.df[f'{self.r.name}_{method}']
        # 
        # construct regressors based on 'order' specification
        self.Xs:pd.DataFrame = gen_df_xs(
                                        temp_s = self.s, 
                                        pq_order = self.pq_order)
        # 
        # Run OLS
        self.reg_res_trf = OLS(self.y,
                               self.Xs).fit()
        # 
        if verbose or regres :
            print(self.reg_res_trf.summary())
        # generate prediction table.
        # gen data for recover TRF and MMT.
        self.s_pred :pd.Series    = pd.Series(np.arange(0, 1, 1/self.std_interval), name = self.r.name)
        self.Xs_pred:pd.DataFrame = pq_powering(temp_s=self.s_pred, pq_order = self.pq_order)
        if verbose : print (self.Xs_pred)
        # 
        # find  'MMT'
        self.recovered_trf  = self.reg_res_trf.predict(self.Xs_pred)    # fitted y; recovered TRF
        self.mmt_s          = np.divide(np.argmin(self.recovered_trf), #type:ignore - index of the minimum 'TRF'
                                                self.std_interval)     # 'interval' to convert 'index' to 'temp_s'
        if verbose : print(f'MMT: {self.mmt_s} \n',self.reg_res_trf.summary())





    def estimate_ctrf(self, verbose:bool = False, regres:bool = False):
        # estimate trf first, to find MMT.
        self.estimate_trf()
        # 
        self.preprocess_ctrf()
        if verbose : print(self.Xs_ctrf)
        # 
        # run OLS
        self.reg_res_ctrf = OLS(self.y, self.Xs_ctrf[self.reg_spec_var_ctrf]).fit()
        if verbose or regres : print(self.reg_res_ctrf.summary())
        # 
        # update MMT in ctrf
        self.recovered_ctrf = recover_ctrf(
                                    ctrf_pred_lst=[],
                                    s_pred  = self.s_pred,
                                    coef    = self.reg_res_ctrf.params,
                                    pq_order= self.pq_order,
                                    verbose = False
                                    )
        # print(self.recovered_ctrf)
        # self.recovered_ctrf
        self.mmt_s = float(np.argmin(self.recovered_ctrf['base']) / self.std_interval)
        # 
        if verbose : print(self.reg_res_ctrf.summary(), '\n', self.reg_res_ctrf.params)
        #



    def preprocess_ctrf(self,
                        ):
        # 
        if self.covariates is None or len(self.covariates.keys()) < 1 :
            raise ValueError('CTRF model needs covariates')
        Xs_ctrf = self.Xs.copy()
        reg_spec_var_ctrf = Xs_ctrf.columns.to_list()
        # adding normalizaed variable for ctrf
        for cov in self.covariates.keys() :  # pyright: ignore[reportOptionalIterable]
            # update key.
            method  = self.scale[cov]
            df      = self.df[[cov]]
            
            normalized_covariate:pd.DataFrame  # contains transformer
            if method == 'linear' :
                print(f'lienar: {cov} {method}')
                self.covariates[cov] = cov_label = f'{cov}_{method}' # type: ignore
                normalized_covariate = Normalizer().normalizer(
                                                    trans_var = cov, 
                                                    method    = method,  # type: ignore
                                                    df        = df,
                                                    cross_id  = None    # not grouping to make covariates explain differences.
                                                )
                transformer = normalized_covariate[[f'{cov}_{method}_transformer']]

            elif method == 'raw':
                self.covariates[cov] = cov_label = f'{cov}_{method}' # type: ignore
                normalized_covariate = df.rename(columns={cov:cov_label}).copy()
                transformer = pd.DataFrame({f'{cov}_{method}_transformer':['.']*len(self.df)})
            
            elif method == 'standard' :
                self.covariates[cov] = cov_label = f'{cov}_{method}_centered' # type: ignore
                '''CENTERING here'''
                if self.covariates_logged is not None:
                    log_converted = cov in self.covariates_logged      # type: ignore
                else: 
                    log_converted = False
                normalized_covariate = Normalizer().centering_cov(
                                                    df            = df,
                                                    trans_var     = cov, 
                                                    mmt_s         = self.mmt_s, # type: ignore
                                                    s            = self.s,
                                                    log_converted = log_converted,
                                                    method        = method,     # type: ignore
                                                    cross_id      = None        # not grouping to make covariates explain differences.
                                                )
                transformer = normalized_covariate[[f'{cov}_{method}_centered_transformer']]

            else :
                print(f'else: {cov} {method}')
                self.covariates[cov] = cov_label = f'{cov}_{method}'        # type: ignore
                normalized_covariate = Normalizer().normalizer(
                                                    trans_var = cov, 
                                                    method    = method,     # type: ignore
                                                    df        = df,
                                                    cross_id  = None        # not grouping to make covariates explain differences.
                                                )
                transformer = normalized_covariate[[f'{cov}_{method}_transformer']]
                # 

            self.s_cov[cov] = normalized_covariate
            pq_powered_cov = pq_powering(   
                                            temp_s      = self.s, 
                                            pq_order    = self.pq_order, 
                                            covariate_s = normalized_covariate[cov_label] )
            reg_spec_var_ctrf.extend(pq_powered_cov.columns.to_list())
            # 
            Xs_ctrf = pd.concat([Xs_ctrf, transformer, pq_powered_cov],
                                        axis=1 )
            # 
        self.Xs_ctrf = Xs_ctrf
        self.reg_spec_var_ctrf = reg_spec_var_ctrf
        return Xs_ctrf, reg_spec_var_ctrf





def recover_ctrf(   s_pred : pd.Series, 
                    coef : pd.Series, 
                    pq_order : dict, 
                    ctrf_pred_lst : Optional[list] = None,
                    verbose=False ):
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
    ctrf['base'] = calc_ctrf(s_pred=s_pred, coef=coef, pq_order=pq_order)

    # for other covariates
    if ctrf_pred_lst is not None :
        for elm in ctrf_pred_lst :
            cov = list(elm.keys())[0]
            # iter over muliplication value.
            for cov_multiple in elm[cov] :
                ctrf[f'{cov} {cov_multiple:.4f}'] = calc_ctrf(s_pred, coef, pq_order, cov, cov_multiple)
                if verbose : print(cov, cov_multiple)

    return ctrf
        


def calc_ctrf( s_pred : pd.Series, 
              coef : pd.Series, 
              pq_order:dict, cov:Optional[str] = None, 
              muliple:float = 1.0):
    
    # gen prediction df for ctrf
    # if ctrf is not provided, estimate the baseCTRF.
    if cov is not None or cov == s_pred.name :
        ctrf_pred = pd.Series( [muliple]*len(s_pred.index), name=cov )
    else : ctrf_pred = None
    
    # gen df for recovering ctrf.
    df = pq_powering(temp_s=s_pred,
                                    pq_order=pq_order,
                                    covariate_s=ctrf_pred) # type: ignore
    
    # recover ctrf based on normalized temp.
    ctrf_recov = df.dot(coef[df.columns])

    return ctrf_recov


