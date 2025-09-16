# import warnings
import pandas as pd
import numpy as np
from statsmodels.api import OLS
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.ar_model import AutoRegResultsWrapper

from sklearn.preprocessing import QuantileTransformer

import psutil
import os

from joblib import Parallel, delayed
import torch
from typing import Dict, Optional, Union, Literal, Tuple, Any




from .cross_trf import CTRF


class SieveBootstrap:
     # Column Name Constants
    _ETA_HAT = 'etahat'
    _EPSILON_HAT = 'epsilonhat'
    _EPSILON_STAR = 'epsilonstar'
    _ETA_STAR = 'etastar'
    _Y_HAT = 'yhat'
    _Y_STAR = 'ystar'
    _T = 'T'


    def __init__(self,  model : CTRF         , 
                        covariates : list = [],   
                        maxL : int  = 24,
                        B : int               =999,
                        device : Literal['cpu','cuda'] = 'cuda',
                        multiprocessing : Literal[None, 'cpu', 'gpu'] = 'cpu',
                        # ar_order_selection:Literal['aic','bic','hqic', None] = None
                        ):
        """
        TRF model
        1. Use model, generate fitted residual self._ETA_HAT
        2. append 'time' and 'cross' ids
        3. use fitted residual self._ETA_HAT, esimate AR model for each 'cross' id
        4. apply BIC to get the AR parameters - multiple of 12.
        5. obtain self._EPSILON_HAT from AR model
        6. demean self._EPSILON_HAT to optain self._EPSILON_STAR
        7. Use the AR model, generate bootstraped self._ETA_STAR with self._EPSILON_STAR
        8. use bootstraped self._ETA_STAR, construct 'y star'
        9. find 0.025, and 0.975 tau equvalant 'y star'
        10. use them, estimate TRF, or CTRF -> confidence band

        CTRF model
        * covariates selection process.

        Parameter(s):
            CTRF: CTRF model object from CTRF class.
        """
        self.dep      : str = model.y.name      #type:ignore
        self.indep    : str = model.r.name      #type:ignore
        self.time_id  : str = model.time_id
        self.cross_id : str = model.cross_id
        # self.df         = model.df[[self.dep, self.indep, self.time_id, self.cross_id]].copy()
        self.CTRF : CTRF    = model              # CTRF model object from CTRF class
        self.B              = B                 # number of bootstrap samples
        # 

        self.cross_ids:list = list(model.df[model.cross_id].unique().copy()) # list of cross-sectional id

        self.maxT : int  # maximum time point update

        # optimal AR order selection
        self.ar_select :dict[str,int]
        self.maxL : int = maxL
        # 
        self.device = device
        self.muliprocessing = multiprocessing
        print('pooled, AR preserved, non-contemporary')


    def run_trf(self, update_lags:bool = False
                ) -> None :
        '''
        For TRF model:
            obtain demeaned whitenoise '_ETA_STAR'
            bootstrapping:
                * within group
                * allowing between group (contemprarily) (update in later)
        '''
        multiprocesssing:bool = self.muliprocessing is not None

        # TRF model generate dataframe
        self.CTRF_reg_res : RegressionResultsWrapper = self.CTRF.reg_res_trf  # type: ignore
        # p,q powered normalized temperature
        self.CTRF_df_Xs      = self.CTRF.Xs.copy()
        self.df = pd.concat([   self.CTRF.df[[self.cross_id, self.time_id]],
                        self.CTRF.y.rename('y'),
                        pd.Series(self.CTRF_reg_res.predict(), name=self._Y_HAT), # yhat
                        pd.Series(self.CTRF_reg_res.resid, name=self._ETA_HAT), # eta
                    ], axis=1)
        # assign time index
        self._assign_time_index()
        self.maxT = self.df[self._T].max()  # maximum time point update
        # update maxL
        if update_lags :
            self.ar_order_selection(maxlag=52, ar_order_selection='bic')
        # obtain whitenoise, and AR structure.
        self.obtain_error_terms_and_ar_structure(multiprocess=multiprocesssing)
        # bootstrap _EPSILON_STAR then construct _ETA_STAR, and _Y_STAR.
        self.bootstrapping(max_iter = self.B, cpu = multiprocesssing)
        # Find 0.025, and 0.975 thresholds _Y_STAR.
        # self.find_thresholds()        # not using
        # return self.df_bootstrapped_ystar
    

    def run_ctrf(self, covariates:Optional[dict[str,float]] = None, update_lags:bool = False):
        multiprocesssing:bool = self.muliprocessing is not None

        self.CTRF_reg_res : RegressionResultsWrapper = self.CTRF.reg_res_ctrf    # type: ignore
        self.CTRF_df_Xs   = self.CTRF.Xs_ctrf.copy()

        self.df = pd.concat([   self.CTRF.df[[self.CTRF.cross_id, self.CTRF.time_id]],
                                    self.CTRF.y.rename('y'),
                                ], axis=1)
        # baseTRF fitted y
        reg_params = self.CTRF_reg_res.params
        name_temp = self.CTRF.s.name
        params_basetrf = ['Intercept'] + reg_params.filter(like=name_temp).index.to_list().copy()
        # self.df[f'{self._Y_HAT}_base'] = self.CTRF_df_Xs[params_basetrf].dot(reg_params[params_basetrf])
        self.df[self._Y_HAT] = self.CTRF_df_Xs[params_basetrf].dot(reg_params[params_basetrf])
        # for covariates, iter over covariates, and mulpiply corresponding covariates value.
        if covariates :
            for cov in list(covariates.keys()):
                cov_val : float = covariates[cov]
                params_cov = reg_params.filter(like=cov).index.to_list().copy()
                # self.df[f'{self._Y_HAT}_{cov}'] = cov_val * self.CTRF_df_Xs[params_cov].dot(reg_params[params_cov])
                # self.df[self._Y_HAT] = self.df[self._Y_HAT] + self.df[f'{self._Y_HAT}_{cov}']
                self.df[self._Y_HAT] = (self.df[self._Y_HAT] 
                                        + cov_val * self.CTRF_df_Xs[params_cov].dot(reg_params[params_cov]))
        # generate etahat
        self.df[self._ETA_HAT] = self.df['y'] - self.df[self._Y_HAT]

        # assign time index
        self._assign_time_index()
        self.maxT = self.df[self._T].max()  # maximum time point update
        # update maxL
        if update_lags :
            self.ar_order_selection(maxlag=52, ar_order_selection='bic')
        
        # obtain whitenoise, and AR structure.
        self.obtain_error_terms_and_ar_structure(multiprocess=multiprocesssing)
        # bootstrap _EPSILON_STAR then construct _ETA_STAR, and _Y_STAR.
        self.bootstrapping(max_iter = self.B, cpu = multiprocesssing)
        # Find 0.025, and 0.975 thresholds _Y_STAR.
        self.find_thresholds()
        return self.df_bs_res_y_ths






    def run_ar_model_i(self, df:pd.DataFrame, cid:str, endog:str,
                       ) -> Tuple[pd.DataFrame, AutoRegResultsWrapper] :
        dfi = df[[self.cross_id, self.time_id, self._T, endog]].loc[df[self.cross_id]==cid].copy()
        ari = AutoReg(dfi[endog],
                     lags=self.maxL,
                     trend='n',
                     seasonal=False).fit()
        dfi[self._EPSILON_HAT] = pd.Series(ari.resid, name=self._EPSILON_HAT)
        dfi[self._EPSILON_STAR] = dfi[self._EPSILON_HAT] - dfi[self._EPSILON_HAT].mean()
        return dfi, ari 
    

    

    def obtain_error_terms_and_ar_structure(self, multiprocess:bool = False):
        if not multiprocess :
            df_errors = []
            ar_res = {}
            for cid in self.cross_ids :
                dfi, ari = self.run_ar_model_i(df = self.df,
                                            cid = cid, 
                                            endog= self._ETA_HAT)
                df_errors.append(dfi)
                ar_res[cid] = ari
            df_errors = pd.concat(df_errors, axis=0)
            # 
            self.df = self.df.merge(df_errors[[self._EPSILON_STAR]], left_index=True, right_index=True)
            self.ar_res = ar_res
            # return df_errors, ar_res

        else:
            results = Parallel(n_jobs=-1)(
                delayed(self.run_ar_model_i)(
                    df=self.df, cid=cid, endog=self._ETA_HAT
                    )
                for cid in self.cross_ids
                )
            df_errors_list, ar_res_dict = zip(*results)
            df_errors = pd.concat(df_errors_list, axis=0)
            ar_res = dict(zip(self.cross_ids, ar_res_dict))
            # 
            self.df = self.df.merge(df_errors[[self._EPSILON_STAR]], left_index=True, right_index=True)
            self.ar_res = ar_res
            # return df_errors, ar_res



    def bootstrap_ystar_i(self, df:pd.DataFrame, cid:str
                          ) -> pd.DataFrame :
        dfi = df[[self.cross_id, self.time_id, self._T, 'y', self._Y_HAT, self._ETA_HAT, self._EPSILON_STAR]].loc[df[self.cross_id]==cid].copy()
        dfi[self._ETA_STAR] = dfi[self._ETA_HAT]
        dfi.loc[dfi[self._T] >= self.maxL, self._ETA_STAR] = pd.NA

        # bootstrapping epsilconstar
        rnd_index_gen = np.random.default_rng(seed=None)
        epsilonstar = dfi[self._EPSILON_STAR].dropna()
        rnd_index = rnd_index_gen.choice(epsilonstar.index, size=epsilonstar.__len__())
        dfi.loc[dfi[self._T] >= self.maxL, 'bs_epsilonstar'] = pd.Series(epsilonstar[rnd_index].values, index=epsilonstar.index)

        # 
        ar_params = self.ar_res[cid].params

        def _lagged_srr(srr:pd.Series, t:int, maxL:int, newindex:list):
            picker = list(srr.index)[t-maxL:t][::-1]
            values = srr.loc[picker].values
            return pd.Series(values, index=newindex, name=srr.name)
        
        for t in dfi[self._T] :
            if t >= self.maxL :
                dfi.loc[dfi[self._T] == t, self._ETA_STAR] = (dfi.loc[dfi[self._T] == t, 'bs_epsilonstar']
                                                     + _lagged_srr(dfi[self._ETA_STAR], t=t, maxL=self.maxL, newindex=list(ar_params.index)).dot(ar_params))

        dfi.loc[dfi[self._T] >= self.maxL,self._Y_STAR] = dfi[self._Y_HAT] + dfi[self._ETA_STAR]

        return dfi

    
    def bootstrap_ystar(self, bs_iteration:int = 1
                        ) -> pd.DataFrame :
        bootstraped_ystar = []
        for cid in self.cross_ids :
            bsi = self.bootstrap_ystar_i(df=self.df, cid=cid)[[self.cross_id, self.time_id, self._T, self._Y_STAR]].rename(columns={self._Y_STAR:f'ystar{bs_iteration}'})
            bootstraped_ystar.append(bsi)
        return pd.concat(bootstraped_ystar, axis=0)

    

    def bootstrapping(self, max_iter:Optional[int] = None, cpu:bool=False):
        # 
        if not max_iter :
            max_iter = self.B + 1
        # 
        if not cpu :
            bootstrapped_res = {}
            for i in range(1, max_iter + 1) :
                bootstrapped_res[i] = self.bootstrap_ystar(bs_iteration=i)
            self.bootstrapped_res = bootstrapped_res
            # 
        else :
            p_core_indices = list(range(12)) # Assuming first 12 cores are P-cores
            p = psutil.Process(os.getpid())
            p.cpu_affinity(p_core_indices)
            # 
            results = Parallel(n_jobs=-1)(
                delayed(self.bootstrap_ystar)(bs_iteration=i)
                for i in range(1, max_iter+1)
            )
            # Collect into dict
            self.bootstrapped_res = {i+1: res for i, res in enumerate(results)}




    def combined_bs_result(self, verbose:bool = False):
        # Extract just the ystar columns, ensuring they are aligned
        ystar_dfs = [res.set_index([self.cross_id, self.time_id, self._T]) for res in self.bootstrapped_res.values()]  # type: ignore
        
        # Concatenate all ystar DataFrames along the column axis at once
        all_ystars = pd.concat(ystar_dfs, axis=1)

        # Merge this single, wide DataFrame with the base DataFrame
        base_df = self.df[[self.cross_id, self.time_id, self._T, 'y']].set_index([self.cross_id, self.time_id, self._T]).rename(columns={'y':f'{self._Y_STAR}0'})
        self.df_bootstrapped_ystar = base_df.join(all_ystars).reset_index()

        if verbose:
            return self.df_bootstrapped_ystar



    def find_thresholds(self):  

        self.combined_bs_result()

        df = self.df_bootstrapped_ystar.dropna(ignore_index=True)
        ystar_cols = df.filter(like=self._Y_STAR).columns

        q025 = np.quantile(df[ystar_cols].values, 0.025, axis=1)
        q975 = np.quantile(df[ystar_cols].values, 0.975, axis=1)

        df['y_025'] = q025
        df['y_975'] = q975

        self.df_bs_res_y_ths = df[[self.cross_id, self.time_id, self._T, 'y_025', 'y_975']]

        return self.df_bs_res_y_ths








    def ar_order_selection(self, 
                           maxlag:int = 52, 
                           ar_order_selection:Literal['aic','bic','hqic'] = 'bic',
                           verbos:bool=False):
        '''
        Use 'statsmodel' ar selector
        '''
        cross_ids = self.cross_ids
        df = self.df
        aic = []
        bic = []
        hqic = []
        for id in cross_ids :
            eta_hat = df.loc[df[self.cross_id]==id][self._ETA_HAT].copy()
            aic.append(ar_select_order(eta_hat, maxlag=maxlag, ic='aic',trend='n', seasonal=False).ar_lags)
            bic.append(ar_select_order(eta_hat, maxlag=maxlag, ic='bic',trend='n', seasonal=False).ar_lags)
            hqic.append(ar_select_order(eta_hat, maxlag=maxlag, ic='hqic',trend='n', seasonal=False).ar_lags)

        df_ar_sel = pd.concat([
                        pd.Series(cross_ids, name=self.cross_id),
                        pd.Series(aic, name='aic'),
                        pd.Series(bic, name='bic'),
                        pd.Series(hqic, name='hqic')], axis=1)
        
        df_ar_sel['max_aic'] = df_ar_sel['aic'].apply(max)
        df_ar_sel['max_bic'] = df_ar_sel['bic'].apply(max)
        df_ar_sel['max_hqic'] = df_ar_sel['hqic'].apply(max)
        
        self.ar_select = {'aic' : int(df_ar_sel['max_aic'].max()),
                          'bic' : int(df_ar_sel['max_bic'].max()),
                          'hqic': int(df_ar_sel['max_hqic'].max()),}
        print(self.ar_select)

        annual_converted_optimal_lag = (self.ar_select[ar_order_selection] + 12 - 1) // 12
        self.maxL : int      = annual_converted_optimal_lag * 12          # maximum lag for AR process

        if verbos :
            return df_ar_sel



    # assign T to the dataframe
    def _assign_time_index(self):
        df_T = self.df[[self.CTRF.time_id]].drop_duplicates().copy()
        df_T[self._T] = (df_T[self.CTRF.time_id].rank() -1).astype(int)
        self.df[self._T] = self.df.drop(columns=self._T, errors='ignore').merge(df_T, on=self.CTRF.time_id, how='left')[self._T] 


    
