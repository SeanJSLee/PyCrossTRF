# import warnings
import sys
import os
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from statsmodels.tsa.ar_model import ar_select_order

import psutil
import os

from joblib import Parallel, delayed
import torch
from typing import Dict, Optional, Union, Literal, Tuple, Any

# Assuming CTRF is defined in this module or imported correctly
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

    def __init__(self,  model : CTRF      , 
                         covariates : list = [],  
                         maxL : int  = 24,
                         B : int           = 999,
                        #  device : Literal['cpu','cuda'] = 'cuda',
                         multiprocessing : Literal[None, 'cpu', 'torch','cuda'] = 'cpu',
                         # ar_order_selection:Literal['aic','bic','hqic', None] = None
                         seed : int = 9099
                         ):
        
        self.dep        : str = model.y.name      #type:ignore
        self.indep      : str = model.r.name      #type:ignore
        self.time_id    : str = model.time_id
        self.cross_id   : str = model.cross_id
        self.CTRF       : CTRF  = model           # CTRF model object from CTRF class
        self.B          = B                     # number of bootstrap samples
        self.cross_ids  : list = list(model.df[model.cross_id].unique()) # list of cross-sectional id
        self.maxT       : int   # maximum time point update
        self.ar_select  : dict[str,int]
        self.maxL       : int = maxL
        
        if multiprocessing == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("CUDA is available. Using GPU for bootstrapping.")
        elif multiprocessing == 'torch':
            self.device = torch.device('cpu')
        #     # if device == 'cuda':
        #     #     print("CUDA not available. Falling back to CPU.")
        
        self.muliprocessing = multiprocessing




    def bootstrapping(self, 
                      max_iter: Optional[int] = None, 
                      contemporaneously:bool = False,
                      multiprocessing : Literal[None, 'cpu', 'gpu'] = None
                      ):
        if not max_iter:
            max_iter = self.B
        
        if (multiprocessing == 'cuda') | (multiprocessing == 'torch'):
            print(f"--- Running PyTorch Bootstrapping on {self.device} for {max_iter} iterations ---")
            if contemporaneously :
                print('contemp')
                self.bootstrapping_pytorch_contemporaneous(max_iter=max_iter)
            else:
                self.bootstrapping_pytorch(max_iter=max_iter)

        # elif multiprocessing == 'torch' :
        #     print(f"--- Running PyTorch Bootstrapping on Torch for {max_iter} iterations ---")
        #     self.bootstrapping_pytorch(max_iter=max_iter)

        elif multiprocessing == 'cpu' :
            print(f"--- Running CPU Bootstrapping with Parallel processing ---")
            try:
                p_core_indices = list(range(12)) 
                p = psutil.Process(os.getpid())
                p.cpu_affinity(p_core_indices)
            except (AttributeError, ValueError, psutil.Error) as e:
                print(f"Could not set CPU affinity: {e}")

            results = Parallel(n_jobs=-1)(
                delayed(self.bootstrap_ystar)(bs_iteration=i, contemporaneously=contemporaneously)
                for i in range(1, max_iter + 1)
            )
            self.bootstrapped_res = {i + 1: res for i, res in enumerate(results)}
        else:
            print(f"--- Running Original Single-threaded CPU Bootstrapping ---")
            bootstrapped_res = {}
            for i in range(1, max_iter + 1):
                bootstrapped_res[i] = self.bootstrap_ystar(bs_iteration=i, contemporaneously=contemporaneously)
            self.bootstrapped_res = bootstrapped_res
            
    def bootstrapping_pytorch(self, max_iter: int):
        B = max_iter
        N = len(self.cross_ids)
        T = self.maxT
        L = self.maxL

        # --- 1. Data Preparation: Pandas to PyTorch Tensors ---
        df_pivot_yhat = self.df.pivot(index=self.cross_id, columns=self._T, values=self._Y_HAT)
        df_pivot_etahat = self.df.pivot(index=self.cross_id, columns=self._T, values=self._ETA_HAT)
        df_pivot_epsilonstar = self.df.pivot(index=self.cross_id, columns=self._T, values=self._EPSILON_STAR)
        
        df_pivot_yhat = df_pivot_yhat.loc[self.cross_ids]
        df_pivot_etahat = df_pivot_etahat.loc[self.cross_ids]
        df_pivot_epsilonstar = df_pivot_epsilonstar.loc[self.cross_ids]
        
        # FIX: Coerce DataFrames to numeric type before tensor conversion.
        df_pivot_yhat = df_pivot_yhat.apply(pd.to_numeric, errors='coerce')
        df_pivot_etahat = df_pivot_etahat.apply(pd.to_numeric, errors='coerce')

        yhat_tensor = torch.tensor(df_pivot_yhat.values, dtype=torch.float32, device=self.device)
        etahat_tensor = torch.tensor(df_pivot_etahat.values, dtype=torch.float32, device=self.device)
        
        ar_params_list = [torch.tensor(self.ar_res[cid].params.values, dtype=torch.float32) for cid in self.cross_ids]
        ar_params_tensor = torch.stack(ar_params_list).to(self.device)

        # --- 2. Resampling Epsilon Star ---
        bs_epsilon_tensor = torch.full((B, N, T), float('nan'), dtype=torch.float32, device=self.device)
        for i, cid in enumerate(self.cross_ids):
            valid_epsilons = torch.tensor(df_pivot_epsilonstar.loc[cid].dropna().values, dtype=torch.float32, device=self.device)
            num_valid = len(valid_epsilons)
            num_to_sample = T - L
            if num_valid > 0 and num_to_sample > 0:
                random_indices = torch.randint(0, num_valid, size=(B, num_to_sample), device=self.device)
                bs_epsilon_tensor[:, i, L:] = valid_epsilons[random_indices]

        # --- 3. Core Bootstrap Loop (Vectorized) ---
        yhat_b = yhat_tensor.expand(B, -1, -1)
        etastar_b = etahat_tensor.expand(B, -1, -1).clone()
        
        for t in range(L, T):
            lagged_etastar = etastar_b[:, :, t-L:t]
            ar_term = torch.sum(lagged_etastar.flip(dims=[2]) * ar_params_tensor.unsqueeze(0), dim=2)
            etastar_b[:, :, t] = ar_term + bs_epsilon_tensor[:, :, t]

        ystar_b = yhat_b + etastar_b

        # --- 4. Format Results: PyTorch Tensors back to Pandas ---
        ystar_b_cpu = ystar_b.cpu().numpy()
        self.bootstrapped_res = {}
        
        for i in range(B):
            ystar_series = pd.Series(ystar_b_cpu[i, :, :].flatten(), name=f'ystar{i+1}')
            df_ystar = df_pivot_yhat.stack(future_stack=True).to_frame().drop(columns=0) # type: ignore
            df_ystar[f'ystar{i+1}'] = ystar_series.values
            df_ystar = df_ystar.reset_index().rename(columns={'level_1': self._T})
            final_df = df_ystar[[self.cross_id, self._T, f'ystar{i+1}']]
            final_df = final_df.merge(self.df[[self.cross_id, self.time_id, self._T]].drop_duplicates(), on=[self.cross_id, self._T])
            self.bootstrapped_res[i+1] = final_df[[self.cross_id, self.time_id, self._T, f'ystar{i+1}']]


    # 
    def bootstrapping_pytorch_contemporaneous(self, max_iter: int, seed:Optional[int]=None):
        B = int(max_iter)
        N = len(self.cross_ids)
        T = int(self.maxT)  # your code sets maxT = df[T].max() + 1
        L = int(self.maxL)

        # device = getattr(self, 'device', torch.device('cpu'))
        device = self.device
        dtype = torch.float32

        # -------------------------------
        # 1) Prepare panel tensors (N x T)
        # -------------------------------
        # Ensure consistent T columns 0..T-1 for all pivots
        cols = pd.Index(range(T), name=self._T)

        df_pivot_yhat = (
            self.df.pivot(index=self.cross_id, columns=self._T, values=self._Y_HAT)
                .reindex(index=self.cross_ids, columns=cols)
                .apply(pd.to_numeric, errors='coerce')
        )
        df_pivot_etahat = (
            self.df.pivot(index=self.cross_id, columns=self._T, values=self._ETA_HAT)
                .reindex(index=self.cross_ids, columns=cols)
                .apply(pd.to_numeric, errors='coerce')
        )
        df_pivot_epsilonstar = (
            self.df.pivot(index=self.cross_id, columns=self._T, values=self._EPSILON_STAR)
                .reindex(index=self.cross_ids, columns=cols)
                .apply(pd.to_numeric, errors='coerce')
        )

        # Move to torch
        yhat_tensor = torch.tensor(df_pivot_yhat.values, dtype=dtype, device=device)   # [N, T]
        etahat_tensor = torch.tensor(df_pivot_etahat.values, dtype=dtype, device=device) # [N, T]
        eps_tensor = torch.tensor(df_pivot_epsilonstar.values, dtype=dtype, device=device)  # [N, T], may have NaNs

        # AR params per cid (assumed length L from statsmodels AutoReg with lags=L)
        ar_params_list = [torch.tensor(self.ar_res[cid].params.values, dtype=dtype) for cid in self.cross_ids]
        ar_params_tensor = torch.stack(ar_params_list, dim=0).to(device)  # [N, L]

        # -------------------------------
        # 2) Build contemporaneous epsilon* draws: [B, N, T]
        # -------------------------------
        bs_epsilon = torch.zeros((B, N, T), dtype=dtype, device=device)
        bs_epsilon[:, :, :L] = 0  # no bootstrapping in the first L steps

        # Optional determinism (if you stored a seed in __init__ as seed)
        gen = torch.Generator(device=device)
        if hasattr(self, 'seed') and isinstance(seed, (int, np.integer)):
            gen.manual_seed(int(seed))

        for t in range(L, T):
            # Pool across cross-sections at time t (valid eps only)
            col = eps_tensor[:, t]  # [N]
            mask = ~torch.isnan(col)
            pool = col[mask]
            K = pool.numel()
            if K == 0:
                # No available epsilon* at this t; follow your CPU fallback (zeros)
                bs_epsilon[:, :, t] = 0
                continue
            # Draw with replacement for every (b, n)
            idx = torch.randint(low=0, high=K, size=(B, N), generator=gen, device=device)
            # Index into pool to get shape [B, N]
            bs_epsilon[:, :, t] = pool[idx]

        # -------------------------------
        # 3) AR recursion for eta* and build y*
        # -------------------------------
        # Broadcast to [B, N, T]
        yhat_b = yhat_tensor.expand(B, -1, -1)  # read-only view
        etastar = etahat_tensor.expand(B, -1, -1).clone()  # we will overwrite t >= L

        for t in range(L, T):
            # Collect last L lags at time t for each series
            lagged = etastar[:, :, t - L:t]              # [B, N, L]
            # Reverse lag dimension to align with params ordering [phi1, ..., phiL]
            lagged_rev = torch.flip(lagged, dims=[2])    # [B, N, L]
            # Multiply by AR params per cid and sum over L
            ar_term = (lagged_rev * ar_params_tensor.unsqueeze(0)).sum(dim=2)  # [B, N]
            etastar[:, :, t] = ar_term + bs_epsilon[:, :, t]

        ystar = yhat_b + etastar  # [B, N, T]

        # -------------------------------
        # 4) Convert back to pandas for each draw (match your layout)
        # -------------------------------
        ystar_cpu = ystar.detach().cpu().numpy()  # [B, N, T]

        # Precompute (cid, T) → time_id mapping
        time_map_df = (
            self.df[[self.cross_id, self._T, self.time_id]]
            .drop_duplicates()
            .sort_values([self.cross_id, self._T])
        )

        self.bootstrapped_res = {}
        midx = pd.MultiIndex.from_product([self.cross_ids, range(T)], names=[self.cross_id, self._T])

        for i in range(B):
            flat = ystar_cpu[i].reshape(N * T)
            df_i = pd.DataFrame({f'{self._Y_STAR}{i+1}': flat}, index=midx).reset_index()
            df_i = df_i.merge(time_map_df, on=[self.cross_id, self._T], how='left')
            df_i = df_i[[self.cross_id, self.time_id, self._T, f'{self._Y_STAR}{i+1}']]
            self.bootstrapped_res[i + 1] = df_i

        # Done.
    # 



    ##########################################################
    # containing core, iteration across county.
    ##########################################################
    def bootstrap_ystar(self, 
                        bs_iteration: int = 1,
                        contemporaneously:bool = False,
                        seed:Optional[int] = None
                        ) -> pd.DataFrame:
        if contemporaneously :
            bootstraped_ystar = [
                self.bootstrap_ystar_i_contemp(df=self.df, 
                                    cid=cid,
                                    seed=seed
                                    ).rename(columns={self._Y_STAR: f'ystar{bs_iteration}'})
                for cid in self.cross_ids
            ]
        else:
            bootstraped_ystar = [
                self.bootstrap_ystar_i(df=self.df, 
                                    cid=cid,
                                    seed=seed
                                    ).rename(columns={self._Y_STAR: f'ystar{bs_iteration}'})
                for cid in self.cross_ids
            ]
        return pd.concat(bootstraped_ystar, axis=0)[[self.cross_id, self.time_id, self._T, f'ystar{bs_iteration}']]

    ##########################################################
    # bootstrapping core - within county
    ##########################################################
    def bootstrap_ystar_i(self, 
                          df: pd.DataFrame, 
                          cid: str,
                          seed:Optional[int] = None,
                          demeaning:bool = False
                          ) -> pd.DataFrame:
        
        dfi = df[[self.cross_id, 
                  self.time_id, 
                  self._T, 
                  'y', 
                  self._Y_HAT, 
                  self._ETA_HAT, 
                  self._EPSILON_STAR]].loc[df[self.cross_id] == cid].copy()
        dfi[self._ETA_STAR] = dfi[self._ETA_HAT]
        dfi.loc[dfi[self._T] >= self.maxL, self._ETA_STAR] = pd.NA

        # randomizer
        rng = np.random.default_rng(seed=seed)
        # randomly select epsilon star for bootstrap, length = T-max.L
        epsilonstar = dfi.loc[dfi['T']>= self.maxL][self._EPSILON_STAR]
        # demean within cross section for additional safty
        if demeaning :
            print('before demean',epsilonstar.mean())
            epsilonstar = epsilonstar - epsilonstar.mean()
            print('after demean',epsilonstar.mean())
        if not epsilonstar.empty:
            # From index (T) > maxL, generating bootstrapping error.
            # generate random index for within county bootstrapping. 
            random_indices = rng.choice(epsilonstar.index, size=len(epsilonstar))
            # assign randomized values
            # print(dfi.loc[dfi[self._T] >= self.maxL])
            # print(epsilonstar.loc[random_indices].values)
            dfi.loc[dfi[self._T] >= self.maxL, 'bs_epsilonstar'] = epsilonstar.loc[random_indices].values
        else:
            # T from 0 to maxL, assign 0.
            dfi.loc[dfi[self._T] >= self.maxL, 'bs_epsilonstar'] = 0

        # get AR parameters from AR estimation
        ar_params = self.ar_res[cid].params

        # generating lagged series
        def _lagged_srr(srr: pd.Series, t: int, maxL: int, newindex: list):
            values = srr.iloc[t - maxL:t].values[::-1]
            return pd.Series(values, index=newindex, name=srr.name)
        
        # prep dataframe.
        dfi = dfi.sort_values(by=self._T).reset_index(drop=True)
        # iter over 'T' >= maxL.
        for t_val in dfi[self._T][dfi[self._T] >= self.maxL]:
            current_pd_index = t_val  # type: ignore
            # generate lagged epsilson star.
            lagged_component = _lagged_srr(dfi[self._ETA_STAR], t=t_val, maxL=self.maxL, newindex=list(ar_params.index)).dot(ar_params)
            # assign bootstrapped error.
            dfi.loc[current_pd_index, self._ETA_STAR] = dfi.loc[current_pd_index, 'bs_epsilonstar'] + lagged_component

        # generate bootstrapped y.
        dfi.loc[dfi[self._T] >= self.maxL, self._Y_STAR] = dfi[self._Y_HAT] + dfi[self._ETA_STAR]
        return dfi
    


    ##########################################################
    # bootstrapping core - contemporaneusly
    ##########################################################
    def bootstrap_ystar_i_contemp(self, 
                          df: pd.DataFrame, 
                          cid: str,
                          seed:Optional[int] = None,
                          ) -> pd.DataFrame:
        
        dfi = df[[self.cross_id, 
                  self.time_id, 
                  self._T, 
                  'y', 
                  self._Y_HAT, 
                  self._ETA_HAT, 
                  self._EPSILON_STAR]].loc[df[self.cross_id] == cid].copy()
        dfi[self._ETA_STAR] = dfi[self._ETA_HAT]
        dfi.loc[dfi[self._T] >= self.maxL, self._ETA_STAR] = pd.NA

        # randomizer
        # rng = np.random.default_rng(seed=seed)
        # randomly select epsilon star for bootstrap, length = T-max.L
        # contemporaneously
        # epsilonstar = pd.Series(0, index=range(0,self.maxT + 1), name=self._EPSILON_STAR)
        epsilonstar = dfi.loc[dfi['T']>= self.maxL][['T',self._EPSILON_STAR]]
        epsilonstar[self._EPSILON_STAR] = float
        # print(epsilonstar)
        for T in range(0, self.maxT) :
            # print(T)
            # pick error aross cross sections.
            if T >= self.maxL :
                # index at T
                idx = df[['T']].loc[df['T']==T].index
                rndomized_idx = np.random.choice(idx)
                # print(rndomized_idx)
                epsilonstar.loc[epsilonstar['T'] == T, self._EPSILON_STAR] = (
                    df.loc[rndomized_idx][self._EPSILON_STAR]
                )
        epsilonstar = epsilonstar[self._EPSILON_STAR]
        # epsilonstar = dfi.loc[dfi['T']>= self.maxL][self._EPSILON_STAR]

        if not epsilonstar.empty:
            # From index (T) > maxL, generating bootstrapping error.
            # generate random index for within county bootstrapping. 
            # random_indices = rng.choice(epsilonstar.index, size=len(epsilonstar))
            # assign randomized values
            # print(dfi.loc[dfi[self._T] >= self.maxL])
            # print(epsilonstar.loc[random_indices].values)
            dfi.loc[dfi[self._T] >= self.maxL, 'bs_epsilonstar'] = epsilonstar
        else:
            # T from 0 to maxL, assign 0.
            dfi.loc[dfi[self._T] >= self.maxL, 'bs_epsilonstar'] = 0

        # get AR parameters from AR estimation
        ar_params = self.ar_res[cid].params

        # generating lagged series
        def _lagged_srr(srr: pd.Series, t: int, maxL: int, newindex: list):
            values = srr.iloc[t - maxL:t].values[::-1]
            return pd.Series(values, index=newindex, name=srr.name)
        
        # prep dataframe.
        dfi = dfi.sort_values(by=self._T).reset_index(drop=True)
        # iter over 'T' >= maxL.
        for t_val in dfi[self._T][dfi[self._T] >= self.maxL]:
            current_pd_index = t_val  # type: ignore
            # generate lagged epsilson star.
            lagged_component = _lagged_srr(dfi[self._ETA_STAR], t=t_val, maxL=self.maxL, newindex=list(ar_params.index)).dot(ar_params)
            # assign bootstrapped error.
            dfi.loc[current_pd_index, self._ETA_STAR] = dfi.loc[current_pd_index, 'bs_epsilonstar'] + lagged_component

        # generate bootstrapped y.
        dfi.loc[dfi[self._T] >= self.maxL, self._Y_STAR] = dfi[self._Y_HAT] + dfi[self._ETA_STAR]
        return dfi


    ##########################################################
    # TRF model 
    ##########################################################
    def run_trf(self, 
                ar_each_cid:bool = True,
                contemporaneously:bool = False,
                update_lags:bool = False
                ) -> pd.DataFrame:
        if ar_each_cid:
            if contemporaneously:
                print('TRf model, Each cross-section has  AR preserved, contemporaneously')
            else :
                print('TRf model, Each cross-section has  AR preserved, non-contemporaneously')
        else:
            if contemporaneously:
                print('TRf model, Single AR structure, contemporaneously')
            else :
                print('TRf model, Single AR structure, non-contemporaneously')

        # Use 'CTRF' object, get data, least square params, and simplyfying 'actual' value as 'y', fitted as 'y_hat'.
        self.CTRF_reg_res : RegressionResultsWrapper = self.CTRF.reg_res_trf # type: ignore
        self.CTRF_df_Xs = self.CTRF.Xs.copy()
        self.df = pd.concat([
            self.CTRF.df[[self.cross_id, self.time_id]],
            self.CTRF.y.rename('y'),
            pd.Series(self.CTRF_reg_res.predict(), name=self._Y_HAT),
            pd.Series(self.CTRF_reg_res.resid, name=self._ETA_HAT),
            ], 
            axis=1)
        # self.df = self.df.copy()
        
        # Set time index, 'T
        self._assign_time_index()
        self.maxT = self.df[self._T].max() + 1
        
        # Do the optimal lag selection, defalt skip.
        if update_lags:
            self.ar_order_selection(maxlag=52, ar_order_selection='bic')
            
        # get AR structure.
        self.obtain_error_terms_and_ar_structure(ar_each_cid=ar_each_cid)

        # main boostrapping process.
        self.bootstrapping(max_iter=self.B, 
                            contemporaneously=contemporaneously,
                            multiprocessing=self.muliprocessing) # type: ignore

        # combine result.
        self.combined_bs_result()
        self.build_confidence_interval()
        # self.find_thresholds()
        return self.df_bs_res_y_ths



    def run_ctrf(self, 
                    ar_each_cid:bool = True,
                    contemporaneously:bool = False,
                    covariates:Optional[dict[str,float]] = None, 
                    update_lags:bool = False):
        # 
        if ar_each_cid:
            if contemporaneously:
                print('cTRf model, Each cross-section has  AR preserved, contemporaneously')
            else :
                print('cTRf model, Each cross-section has  AR preserved, non-contemporaneously')
        else:
            if contemporaneously:
                print('cTRf model, Single AR structure, contemporaneously')
            else :
                print('cTRf model, Single AR structure, non-contemporaneously')
        # 
        self.CTRF_reg_res : RegressionResultsWrapper = self.CTRF.reg_res_ctrf # type: ignore
        self.CTRF_df_Xs = self.CTRF.Xs_ctrf.copy()

        self.df = pd.concat([
            self.CTRF.df[[self.CTRF.cross_id, self.CTRF.time_id]],
            self.CTRF.y.rename('y'),
            # pd.Series(self.CTRF_reg_res.predict(), name=self._Y_HAT),
            # pd.Series(self.CTRF_reg_res.resid, name=self._ETA_HAT),
            ], 
            axis=1)
        
        reg_params = self.CTRF_reg_res.params
        name_temp = self.CTRF.s.name
        params_basetrf = ['Intercept'] + reg_params.filter(like=name_temp).index.to_list().copy()
        # base TRF fitted outcome
        self.df[self._Y_HAT] = self.CTRF_df_Xs[params_basetrf].dot(reg_params[params_basetrf])
        # for CTRF, add covariates.
        if covariates:
            for cov, cov_val in covariates.items():
                params_cov = reg_params.filter(like=cov).index.to_list().copy()
                self.df[self._Y_HAT] += (cov_val * self.CTRF_df_Xs[params_cov]).dot(reg_params[params_cov])
        
        self.df[self._ETA_HAT] = self.df['y'] - self.df[self._Y_HAT]
        self.df = self.df.copy()

        self._assign_time_index()
        self.maxT = self.df[self._T].max() + 1
        
        if update_lags:
            self.ar_order_selection(maxlag=52, ar_order_selection='bic')
        
        self.obtain_error_terms_and_ar_structure(ar_each_cid=ar_each_cid)
        self.bootstrapping(max_iter=self.B, 
                           contemporaneously=contemporaneously,
                           multiprocessing=self.muliprocessing)  # type: ignore
        # self.find_thresholds() 
        # return self.df_bs_res_y_ths
        # combine result.
        self.combined_bs_result()
        # self.build_confidence_interval()
        # for ctrf, use 'covariate' and 'scale' to generate fitted value.
        # self.find_thresholds()
        # return self.df_bs_res_y_ths
    


    def run_ar_model_i(self, 
                        df:pd.DataFrame,
                        endog:str,
                        cid:Optional[str] = None
                        ) -> Tuple[pd.DataFrame, AutoRegResultsWrapper]:
        if cid is not None :
            dfi = df.loc[df[self.cross_id] == cid, [self.cross_id, self.time_id, self._T, endog]].copy()
        elif cid is None :
            dfi = df[[self.cross_id, self.time_id, self._T, endog]].copy()

        ari = AutoReg(dfi[endog], lags=self.maxL, trend='n', seasonal=False).fit()
        dfi[self._EPSILON_HAT] = ari.resid
        dfi[self._EPSILON_STAR] = dfi[self._EPSILON_HAT] - dfi[self._EPSILON_HAT].mean()
        return dfi, ari 
    



    def obtain_error_terms_and_ar_structure(self,
                                            ar_each_cid:bool = True):
        df_errors = []
        ar_res = {}
        # 
        # run ar model for each cid
        if ar_each_cid == True :
            for cid in self.cross_ids:
                dfi, ari = self.run_ar_model_i(df=self.df, endog=self._ETA_HAT, cid=cid)
            # else :
            #     dfi, ari = self.run_ar_model_i(df=self.df, endog=self._ETA_HAT)
                df_errors.append(dfi)
                ar_res[cid] = ari
                df_errors_concat = pd.concat(df_errors, axis=0)

        # run single ar model
        #   * assign ar_res for each cid
        #   * eta_hat return in a single dfi.
        else :
            dfi, ari = self.run_ar_model_i(df=self.df, endog=self._ETA_HAT)
            for cid in self.cross_ids:
                ar_res[cid] = ari
                df_errors_concat = dfi
        self.df = self.df.merge(df_errors_concat[[self._EPSILON_STAR]], left_index=True, right_index=True) # type: ignore
        self.ar_res = ar_res




    def combined_bs_result(self, verbose:bool = False):
        ystar_dfs = [res.set_index([self.cross_id, self.time_id, self._T]) for res in self.bootstrapped_res.values()]  # type: ignore
        all_ystars = pd.concat(ystar_dfs, axis=1)
        base_df = self.df[[self.cross_id, self.time_id, self._T, 'y']].set_index([self.cross_id, self.time_id, self._T]).rename(columns={'y': f'ystar0'})
        self.df_bootstrapped_ystar = base_df.join(all_ystars).reset_index()
        if verbose:
            return self.df_bootstrapped_ystar
        

    
    def build_confidence_interval(self):
        #  
        # bootstrapping model
        model = self.CTRF
        df_bs_res = self.df_bootstrapped_ystar
        bs_trf = {}
        bs_fitted_trf_res = {}
        bootstrapped_outcomes = df_bs_res.filter(like='ystar').columns
        for idx, i_star in enumerate(bootstrapped_outcomes) :
            with suppress_output():
                bs_trf[idx] = CTRF(df = df_bs_res[['fips','date','T',i_star]].merge(model.df, on=['fips','date']),
                                    dep= i_star,
                                    covariates= model.covariates,
                                    temp_r=model.r.name,
                                    pq_order=model.pq_order,
                                    scale=model.scale)
                bs_trf[idx].estimate_trf()
                bs_fitted_trf_res[idx] = bs_trf[idx].recovered_trf
        # 
        bootstrapped_outcomes = pd.concat(bs_fitted_trf_res, axis=1)
        df_res = pd.DataFrame(index=bootstrapped_outcomes.index)
        df_res[model.r.name] = df_res.index / df_res.index.max()
        df_res['ystar_0.025'] = bootstrapped_outcomes.quantile(0.025, axis=1)
        df_res['ystar_0.975'] = bootstrapped_outcomes.quantile(0.975, axis=1)
        self.df_bs_res_y_ths = df_res
        # return self.df_bs_res_y_ths



    # def find_thresholds(self):  
    #     self.combined_bs_result()
    #     df = self.df_bootstrapped_ystar.dropna(ignore_index=True)
    #     ystar_cols = df.filter(like=self._Y_STAR).columns
    #     df[ystar_cols] = df[ystar_cols].astype(np.float64)
    #     q025 = np.quantile(df[ystar_cols].values, 0.025, axis=1)
    #     q975 = np.quantile(df[ystar_cols].values, 0.975, axis=1)
    #     df['y_025'] = q025
    #     df['y_975'] = q975
    #     self.df_bs_res_y_ths = df[[self.cross_id, self.time_id, self._T, 'y_025', 'y_975']]
    #     return self.df_bs_res_y_ths
    



    def ar_order_selection(self, maxlag:int = 52, ar_order_selection:Literal['aic','bic','hqic'] = 'bic', verbos:bool=False):
        cross_ids = self.cross_ids
        df = self.df
        crit_values = {'aic': [], 'bic': [], 'hqic': []}
        for cid in cross_ids:
            eta_hat = df.loc[df[self.cross_id] == cid, self._ETA_HAT].copy() # type: ignore
            for crit in crit_values.keys():
                sel_res = ar_select_order(eta_hat, maxlag=maxlag, ic=crit, trend='n', seasonal=False)
                crit_values[crit].append(sel_res.ar_lags if sel_res.ar_lags else [0])

        df_ar_sel = pd.DataFrame({self.cross_id: cross_ids, **crit_values})
        for crit in crit_values.keys():
            df_ar_sel[f'max_{crit}'] = df_ar_sel[crit].apply(max)
        
        self.ar_select = {crit: int(df_ar_sel[f'max_{crit}'].max()) for crit in crit_values.keys()}
        print(self.ar_select)
        annual_converted_optimal_lag = (self.ar_select[ar_order_selection] + 11) // 12
        self.maxL: int = annual_converted_optimal_lag * 12
        if verbos:
            return df_ar_sel
        



    def _assign_time_index(self):
        unique_times = sorted(self.df[self.time_id].unique())
        time_map = {time: i for i, time in enumerate(unique_times)}
        self.df[self._T] = self.df[self.time_id].map(time_map)


class suppress_output:
    """A context manager to suppress stdout (console output)."""
    def __enter__(self):
        # Save the original stdout
        self._original_stdout = sys.stdout
        # Redirect stdout to the null device ('/dev/null' on Unix, 'NUL' on Windows)
        # 'w' means write mode
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original stdout and close the temporary null file
        sys.stdout.close()
        sys.stdout = self._original_stdout
        # Return False to propagate any exceptions
        return False