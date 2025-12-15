# import warnings
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
    """
    Implements a sieve bootstrap procedure for time-series panel data, 
    with options for CPU and GPU (CUDA) execution.
    """
    # Column Name Constants
    _ETA_HAT = 'etahat'
    _EPSILON_HAT = 'epsilonhat'
    _EPSILON_STAR = 'epsilonstar'
    _ETA_STAR = 'etastar'
    _Y_HAT = 'yhat'
    _Y_STAR = 'ystar'
    _T = 'T'


    def __init__(self,
                 model: CTRF,
                 maxL: int = 24,
                 B: int = 999,
                 multiprocessing: Literal['cpu', 'gpu'] = 'gpu',
                 device: str = 'cuda'
                 ):
        """
        Initializes the SieveBootstrap procedure.

        Args:
            model (CTRF): A fitted model object from the CTRF class.
            maxL (int): The maximum lag order for the autoregressive model.
            B (int): The number of bootstrap replications.
            multiprocessing (Literal['cpu', 'gpu']): The computation backend. 
                                                     'gpu' for PyTorch/CUDA, 'cpu' for joblib.
            device (str): The torch device to use for 'gpu' mode (e.g., 'cuda', 'cuda:0').
        """
        self.CTRF: CTRF = model
        self.dep: str = model.y.name  #type:ignore
        self.indep: str = model.r.name  #type:ignore
        self.time_id: str = model.time_id
        self.cross_id: str = model.cross_id
        
        self.maxL = maxL
        self.B = B
        self.multiprocessing = multiprocessing
        
        # Setup device for PyTorch
        if self.multiprocessing == 'gpu':
            if not torch.cuda.is_available():
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)
            print(f"Using device: {self.device}")
        
        self.cross_ids: list = sorted(list(model.df[self.cross_id].unique()))
        self.n_cross: int = len(self.cross_ids)

        # Extract regression results
        if model.model == 'trf':
            self.CTRF_reg_res: RegressionResultsWrapper = model.reg_res_trf  # type: ignore
        else: # 'ctrf'
            self.CTRF_reg_res: RegressionResultsWrapper = model.reg_res_ctrf  # type: ignore
        
        # Prepare initial DataFrame
        self.df = pd.concat([
            model.df[[self.cross_id, self.time_id]],
            model.y.rename('y'),
            pd.Series(self.CTRF_reg_res.predict(), name=self._Y_HAT),
            pd.Series(self.CTRF_reg_res.resid, name=self._ETA_HAT),
        ], axis=1).sort_values(by=[self.cross_id, self.time_id]).reset_index(drop=True)

        self._assign_time_index()
        self.n_time: int = self.df[self._T].max() + 1
        
        self.ar_res: Dict[str, AutoRegResultsWrapper] = {}
        self.bootstrapped_ystar_tensor: Optional[torch.Tensor] = None

    def _assign_time_index(self):
        """Assigns a zero-based integer time index 'T' to the dataframe."""
        df_T = self.df[[self.time_id]].drop_duplicates().sort_values(by=self.time_id)
        df_T[self._T] = range(len(df_T))
        self.df = self.df.merge(df_T, on=self.time_id, how='left')

    def run_ar_model_i(self, cid: str) -> Tuple[pd.Series, AutoRegResultsWrapper]:
        """Fits an AR model for a single cross-section."""
        dfi = self.df.loc[self.df[self.cross_id] == cid]
        ari = AutoReg(dfi[self._ETA_HAT], lags=self.maxL, trend='n').fit()
        
        epsilon_hat = pd.Series(ari.resid, index=dfi.index)
        epsilon_star = epsilon_hat - epsilon_hat.mean()
        
        return epsilon_star, ari

    def obtain_error_terms_and_ar_structure(self, use_parallel: bool = True):
        """
        Fits AR models to residuals for each cross-section to obtain innovations.
        
        Args:
            use_parallel (bool): Whether to use joblib for parallel processing on CPU.
        """
        print("Fitting AR models to residuals...")
        if use_parallel and self.multiprocessing == 'cpu':
            results = Parallel(n_jobs=-1)(
                delayed(self.run_ar_model_i)(cid) for cid in self.cross_ids
            )
            epsilon_stars, ar_results_list = zip(*results)
            self.df[self._EPSILON_STAR] = pd.concat(epsilon_stars)
            self.ar_res = dict(zip(self.cross_ids, ar_results_list))
        else: # Serial execution
            all_epsilon_stars = []
            for cid in self.cross_ids:
                epsilon_star, ari = self.run_ar_model_i(cid)
                all_epsilon_stars.append(epsilon_star)
                self.ar_res[cid] = ari
            self.df[self._EPSILON_STAR] = pd.concat(all_epsilon_stars)
        
        # Prepare tensors if using GPU
        if self.multiprocessing == 'gpu':
            self._prepare_tensors()

    def _prepare_tensors(self):
        """Converts data to PyTorch tensors and moves them to the specified device."""
        print("Preparing data and moving to GPU...")
        # Reshape dataframe to (n_cross, n_time) for easy tensor conversion
        panel_df = self.df.pivot(index=self.cross_id, columns=self._T)
        
        self.yhat_t = torch.tensor(panel_df[self._Y_HAT].values, 
                                     dtype=torch.float32, device=self.device)
        self.etahat_t = torch.tensor(panel_df[self._ETA_HAT].values, 
                                      dtype=torch.float32, device=self.device)
        
        # Epsilon star has NaNs for the first maxL values, handle this
        self.epsilonstar_t = torch.tensor(panel_df[self._EPSILON_STAR].values, 
                                          dtype=torch.float32, device=self.device)
        
        # Collect AR parameters into a tensor
        # Shape: (n_cross, maxL)
        ar_params_list = [self.ar_res[cid].params.values for cid in self.cross_ids]
        self.ar_params_t = torch.tensor(np.array(ar_params_list), 
                                        dtype=torch.float32, device=self.device)

    def bootstrapping(self):
        """Main bootstrapping dispatcher."""
        if self.multiprocessing == 'gpu':
            self.bootstrapping_gpu()
        else:
            self.bootstrapping_cpu()

    def bootstrapping_gpu(self):
        """Performs all B bootstrap simulations in parallel on the GPU."""
        print(f"Starting {self.B} bootstrap simulations on {self.device}...")
        
        # Tensor to store all bootstrap simulations of etastar
        # Shape: (n_cross, n_time, B)
        etastar_sims = torch.empty(self.n_cross, self.n_time, self.B, device=self.device)
        
        # The first maxL values are the original etahat values for all simulations
        etastar_sims[:, :self.maxL, :] = self.etahat_t[:, :self.maxL].unsqueeze(-1)
        
        # For each cross-section, get the valid (non-NaN) epsilonstar values
        valid_epsilons = [self.epsilonstar_t[i, self.maxL:] for i in range(self.n_cross)]

        # Generate random indices for resampling for ALL simulations at once
        # Shape: (n_cross, n_time - maxL, B)
        num_valid_eps = self.n_time - self.maxL
        rand_indices = torch.randint(0, num_valid_eps, 
                                     (self.n_cross, num_valid_eps, self.B), 
                                     device=self.device)
        
        # Resample epsilonstar
        # Shape: (n_cross, n_time - maxL, B)
        bs_epsilonstar = torch.stack([eps[indices] for eps, indices in zip(valid_epsilons, rand_indices)])

        # Loop through time to generate the AR process
        for t in range(self.maxL, self.n_time):
            # Get the previous maxL values of etastar
            # Shape: (n_cross, maxL, B)
            prev_etastars = etastar_sims[:, t-self.maxL:t, :]
            
            # We need to reverse the time dimension for matmul with AR params
            # Shape after permute: (n_cross, B, maxL)
            prev_etastars_rev = torch.flip(prev_etastars, [1]).permute(0, 2, 1)
            
            # Calculate the AR component: AR_params * lagged_etastars
            # (n_cross, 1, maxL) @ (n_cross, maxL, B) -> (n_cross, 1, B)
            ar_component = torch.bmm(self.ar_params_t.unsqueeze(1), prev_etastars_rev.transpose(1, 2))
            
            # Squeeze to (n_cross, B)
            ar_component = ar_component.squeeze(1)

            # etastar[t] = bootstrapped_error[t] + AR_component
            etastar_sims[:, t, :] = bs_epsilonstar[:, t-self.maxL, :] + ar_component
        
        # Calculate ystar for all simulations
        # ystar = yhat + etastar
        self.bootstrapped_ystar_tensor = self.yhat_t.unsqueeze(-1) + etastar_sims
        print("GPU bootstrapping complete.")

    def bootstrapping_cpu(self):
        """Performs bootstrap simulations in parallel on the CPU using joblib."""
        print(f"Starting {self.B} bootstrap simulations on CPU...")
        
        def bootstrap_ystar_iteration(bs_iter: int) -> pd.DataFrame:
            """Generates one full bootstrap sample across all cross-sections."""
            bootstraped_ystar = []
            for cid in self.cross_ids:
                dfi = self.df.loc[self.df[self.cross_id] == cid].copy()
                
                # Resample epsilonstar
                epsilonstar = dfi[self._EPSILON_STAR].dropna()
                bs_epsilonstar = np.random.choice(epsilonstar.values, size=len(dfi), replace=True)
                
                # Simulate etastar
                ar_params = self.ar_res[cid].params.values
                etastar = np.copy(dfi[self._ETA_HAT].values)
                for t in range(self.maxL, len(dfi)):
                    ar_component = np.dot(ar_params, etastar[t-self.maxL:t][::-1])
                    etastar[t] = bs_epsilonstar[t] + ar_component
                
                dfi[self._Y_STAR] = dfi[self._Y_HAT] + etastar
                
                # Keep only necessary columns
                bsi = dfi[[self.cross_id, self.time_id, self._T, self._Y_STAR]].rename(
                    columns={self._Y_STAR: f'ystar{bs_iter}'})
                bootstraped_ystar.append(bsi)
            return pd.concat(bootstraped_ystar, axis=0)

        results = Parallel(n_jobs=-1)(
            delayed(bootstrap_ystar_iteration)(i) for i in range(1, self.B + 1)
        )
        self.bootstrapped_res_dfs = {i + 1: res for i, res in enumerate(results)}
        print("CPU bootstrapping complete.")

    def find_thresholds(self, q_lower: float = 0.025, q_upper: float = 0.975) -> pd.DataFrame:
        """Calculates the quantiles of the bootstrapped ystar distribution."""
        if self.multiprocessing == 'gpu':
            if self.bootstrapped_ystar_tensor is None:
                raise RuntimeError("Bootstrapping has not been run yet.")
            
            print(f"Calculating {q_lower} and {q_upper} quantiles from GPU results...")
            quantiles = torch.quantile(
                self.bootstrapped_ystar_tensor,
                q=torch.tensor([q_lower, q_upper], device=self.device),
                dim=2 # Quantiles over the B dimension
            )
            # quantiles shape: (n_cross, n_time, 2)
            
            # Convert back to DataFrame
            q_lower_vals = quantiles[:, :, 0].cpu().numpy().flatten()
            q_upper_vals = quantiles[:, :, 1].cpu().numpy().flatten()
            
            df_res = self.df[[self.cross_id, self.time_id, self._T]].copy()
            df_res['y_lower'] = q_lower_vals
            df_res['y_upper'] = q_upper_vals
            return df_res

        else: # CPU version
            if not hasattr(self, 'bootstrapped_res_dfs'):
                raise RuntimeError("Bootstrapping has not been run yet.")
            
            print(f"Calculating {q_lower} and {q_upper} quantiles from CPU results...")
            # Combine results into a single DataFrame
            ystar_dfs = [res.set_index([self.cross_id, self.time_id, self._T]) 
                         for res in self.bootstrapped_res_dfs.values()]
            all_ystars = pd.concat(ystar_dfs, axis=1)

            # Calculate quantiles
            df_res = all_ystars.quantile(q=[q_lower, q_upper], axis=1).T
            df_res = df_res.rename(columns={q_lower: 'y_lower', q_upper: 'y_upper'}).reset_index()
            return df_res