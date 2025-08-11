import pandas as pd
import numpy as np
import math
from statsmodels.api import OLS

from typing import Dict, Optional, Union, Literal, Tuple, Any

from joblib import Parallel, delayed
import psutil
import os
import torch

from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ValueWarning
from pandas.errors import SettingWithCopyWarning

# from .utils import pq_powering, gen_df_xs, apply_transform
from .cross_trf import CTRF
from .utils import pq_powering



# Suppress specific warnings
simplefilter('ignore', ValueWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)



'''
'h-block Cross Validation' for TRF model.
    - It shares the basic structure with 'cross_trf.py' and use 'utils.py'.
    - normalized and p,q powered df, select covariates as p, and q, and covariates.
    - 'pq_order' is the maximum order of p and q.
* It tests model's fit (TRF model). 
    The CTRF model is decompose the TRF model with covariates, 
    therefore the CTRF model fittness may proper when it inherit from the TRF model.
'''
class CV_h_block:
    """
    Cross-validation using h-block method to report Prediction Error (PE) over different p and q orders.
    * inherit normalized value and many other parameter from the 'CTRF' class.
    """
    def __init__(self, 
                 model : CTRF,  # inherit 'trf' model from 'CTRF
                 pq_order_max: dict = {'p':6, 'q':4}, 
                ):
        '''
        'df_grouped_index_block' is the key to tracking each 'cross_id' corresponding time at 'i'.
            If I have 4 fips codes and each code have 100 observation over time, 
            this code returns [[0, 100, 200, 400], [1, 101, 201, 401], ..., [99, 199, 299, 399]]
        '''
        time_id    = model.time_id              
        cross_id   = model.cross_id             
        df_index_block = model.df[[time_id, cross_id]].sort_values(by=[time_id,cross_id]).reset_index().copy()
        self.df_grouped_index_block = df_index_block.groupby(time_id, as_index=False).agg(i=('index',list))
        #
        self.pq_order_max = pq_order_max

        self.y : pd.Series  = model.y            
        # self.r : pd.Series  = model.r         
        self.s      = model.s                # normalizaed temperature
        # self.s_pred = model.s_pred           # temp range to recover TRF and CTRFs
        # self.df_pq_powered = pd.DataFrame()
        self.pq_combination = []
        # 
        self.df_pe_res = pd.DataFrame()
        self.pq_order_updated = []
        self.mean_pe_lowest = float()
        # 


      

    # show C.V. results.
    def compute_cv(self, 
                   parallel: Literal['normal', 'cpu', 'torch'] = 'normal',
                   show_res=False, 
                   verbose=False
                   ) -> tuple:
        """
        Compute the cross-validation prediction error for all p and q combinations.
        
        :param verbose: Print detailed information if True.
        """
        pq_combination = self.gen_pq_combination(verbose=verbose)
        print('Toal combinations:', len(pq_combination))

        if parallel == 'torch' :
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using pytorch, device: {device}")

        pe_res = []
        for pq_order in pq_combination:
            # if verbose : 
            #     mse, df_mse = self.compute_prediction_error(pq_order=pq_order, verbose=verbose)
            # else :
            #     mse = self.compute_prediction_error(pq_order=pq_order, verbose=verbose)
            if parallel == 'normal' :
                mse = self.compute_prediction_error(pq_order=pq_order)
            elif parallel == 'cpu' :
                mse = self.compute_prediction_error_cpu(pq_order=pq_order)
            elif parallel == 'torch' :
                mse = self.compute_prediction_error_gpu(pq_order=pq_order)
            if verbose : print(pq_order, mse)
            pe_res.append([pq_order, mse])
        
        self.df_pe_res = pd.DataFrame(pe_res, columns=['pq_order', 'CV'])
        self.pick_lowest_pe(df_pe_res= self.df_pe_res, verbose=verbose)

        if verbose or show_res: print(self.df_pe_res.sort_values('CV'),'\n', self.pq_order_updated)
        # 
        return self.pq_order_updated[0], self.mean_pe_lowest




    def  compute_prediction_error(self, pq_order = {'p':4, 'q':1}, verbose=False) :
        '''
        Need:
            normalized data with p and q powered. 'ctrf_utils.pre_processing' -> df_Xs
                xs, pq powered.
                1
        Blocking serial correlation by i, j selector.
            grouping same i and j for a panel.
        '''
        # normalized dataframe Xs, using ctrf.utils
        Xs = pq_powering(temp_s=self.s,
                         pq_order=pq_order)
        df_grouped_index_block = self.df_grouped_index_block
        n = df_grouped_index_block.index.max()
        squared_errors = []
        for i in df_grouped_index_block.index :
            '''
            This is the core of the h-block cross-validation.
              - selecting i, j to remove serial correlation.
              'selector' slicing blocks.
            '''
            # selector i grabs observation at time i across cross_id by index.
            selector_i = df_grouped_index_block.loc[i]['i']     
            j = self.gen_ij_selector(
                                i = i, 
                                n = n, 
                                h = 6  # 'h' is the critical valaue proposed by burman et al. 
                                )
            # selector j grabs observations where n/h time point away from time i across cross_id by index.
            selector_j = df_grouped_index_block.loc[j].explode('i')['i'].to_list()
            selector_j.sort()
            model = OLS(self.y.loc[selector_j], Xs.loc[selector_j]).fit()
            # 
            df_predicted_i = pd.DataFrame(model.predict(Xs.loc[selector_i]), columns=['Xs_hat'])
            df_predicted_i['y'] = self.y.loc[selector_i]
            df_predicted_i['squared_error'] = (df_predicted_i['y'] - df_predicted_i['Xs_hat']) ** 2
            squared_errors.append(df_predicted_i['squared_error'])
        mse = pd.concat(squared_errors, axis=0).mean()
        if verbose : 
            return mse, pd.concat(squared_errors, axis=0)
        else :
            return mse  # return mse.



    def  compute_prediction_error_cpu(self, pq_order = {'p':4, 'q':1}, verbose=False) :
        '''
        Use joblib, parallel computing the prediction error.
        '''
        # Set the process affinity to use only the P-cores
        p_core_indices = list(range(12)) # Assuming first 12 cores are P-cores
        p = psutil.Process(os.getpid())
        p.cpu_affinity(p_core_indices)

        # normalized dataframe Xs, using ctrf.utils
        Xs = pq_powering(temp_s=self.s,
                         pq_order=pq_order)
        df_grouped_index_block = self.df_grouped_index_block
        n = df_grouped_index_block.index.max()
        
        def calculate_errors_for_block(i, self_y, Xs_data, df_grouped_index_block, n):
            """
            A helper function to perform the calculation for a single iteration (i)
            of the loop, which can be executed in parallel.
            """
            selector_i = df_grouped_index_block.loc[i]['i']
            j = self.gen_ij_selector(i=i, n=n, h=6)
            selector_j = df_grouped_index_block.loc[j].explode('i')['i'].to_list()
            selector_j.sort()
            
            model = OLS(self_y.iloc[selector_j], Xs_data.iloc[selector_j]).fit()
            
            df_predicted_i = pd.DataFrame(model.predict(Xs_data.iloc[selector_i]), columns=['Xs_hat'])
            df_predicted_i['y'] = self_y.iloc[selector_i]
            df_predicted_i['squared_error'] = (df_predicted_i['y'] - df_predicted_i['Xs_hat']) ** 2
            
            return df_predicted_i['squared_error']

        # Parallelize the loop using joblib
        # n_jobs=-1 uses all available CPU cores
        squared_errors = Parallel(n_jobs=-1)(
            delayed(calculate_errors_for_block)(
                i, self.y, Xs, df_grouped_index_block, n
            ) for i in df_grouped_index_block.index
        )
        
        mse = pd.concat(squared_errors, axis=0).mean()
        
        if verbose : 
            return mse, pd.concat(squared_errors, axis=0)
        else :
            return mse

         

    def compute_prediction_error_gpu(self, pq_order = {'p':4, 'q':1}, verbose=False):
        '''
        GPU-accelerated version of compute_prediction_error using PyTorch.
        This version re-implements OLS calculation on the GPU.
        '''
        # Set the process affinity to use only the P-cores
        p_core_indices = list(range(12)) # Assuming first 12 cores are P-cores
        p = psutil.Process(os.getpid())
        p.cpu_affinity(p_core_indices)

        # Check for GPU availability and set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {device}")

        Xs = pq_powering(temp_s=self.s,
                         pq_order=pq_order)
        df_grouped_index_block = self.df_grouped_index_block
        n = df_grouped_index_block.index.max()

        def calculate_errors_for_block_gpu(i, self_y, Xs_data, df_grouped_index_block, n, device):
            """
            Helper function to perform OLS calculation on the GPU for a single block.
            """
            # Select data for training and prediction
            selector_i = df_grouped_index_block.loc[i]['i']
            j = self.gen_ij_selector(i=i, n=n, h=6)
            selector_j = df_grouped_index_block.loc[j].explode('i')['i'].to_list()
            selector_j.sort()

            y = self_y.loc[selector_j].to_numpy()
            X = pd.concat([pd.DataFrame({'Intercept':[1]*len(Xs_data)}),
                           Xs_data.loc[selector_j]],axis=1).to_numpy()
            
            # Convert selected data to PyTorch tensors and move to the GPU
            # y_train   = torch.tensor( self_y.loc[selector_j].values, dtype=torch.float32, device=device)
            # X_train   = torch.tensor(Xs_data.loc[selector_j].values, dtype=torch.float32, device=device)
            # X_predict = torch.tensor(Xs_data.loc[selector_i].values, dtype=torch.float32, device=device)
            y_train   = torch.from_numpy( self_y.loc[selector_j].to_numpy()).to(device=device)
            X_train   = torch.from_numpy(Xs_data.loc[selector_j].to_numpy()).to(device=device)
            X_predict = torch.from_numpy(Xs_data.loc[selector_i].to_numpy()).to(device=device)

            # Re-implement OLS formula using PyTorch tensor operations
            # OLS solution: beta = (X_train^T * X_train)^-1 * X_train^T * y_train
            ols = torch.linalg.lstsq(X_train, y_train)
            beta = ols.solution
            
            # Make predictions
            y_hat = torch.matmul(X_predict, beta)

            # Calculate squared error
            y_true = torch.from_numpy(self_y.loc[selector_i].to_numpy()).to(device=device)
            squared_errors_tensor = (y_true - y_hat) ** 2
            
            # Move the results back to the CPU for pandas
            return squared_errors_tensor.cpu().numpy()

        # Parallelize the loop on the CPU, with each worker using the GPU for its block's calculation
        squared_errors_list = Parallel(n_jobs=-1)(
            delayed(calculate_errors_for_block_gpu)(
                i, self.y, Xs, df_grouped_index_block, n, device
            ) for i in df_grouped_index_block.index
        )
        
        # Concatenate the results from each process
        mse = pd.concat([pd.Series(se) for se in squared_errors_list]).mean()

        if verbose:
            all_errors = pd.concat([pd.Series(se) for se in squared_errors_list])
            return mse, all_errors
        else:
            return mse




    def gen_pq_combination(self, pq_order : Optional[dict] = None, verbose=False) -> list:
        """
        Generate all possible combinations of p and q orders.
        """
        if pq_order is None : pq_order = self.pq_order_max
        pq_combination = []
        for q in range (1, pq_order['q']+1) :
            for p in range(1, pq_order['p']+1) : 
                pq_combination += [{'p':p, 'q':q}]

        self.pq_combination = pq_combination
        if verbose: print(self.pq_combination)
        
        return self.pq_combination





    def gen_ij_selector(self, i: int, n: int, h: int, verbose=False) -> list:
        """
        Generate the list of indices for training set based on current index i and block size h.
        
        :i: Index of the test observation.
        :n: Total number of observations.
        :h: Block size = n/h.
        """
        h_block:int = int(math.ceil(n/h))
        # 
        if i <= h_block:
            lst_selector = list(range(i + h_block + 1, n))
        # 
        elif i > h_block and i < n - h_block:
            lst_selector = list(range(0, h_block + i)) + list(range(i + h_block + 1, n))
        # 
        elif i >= n - h_block:
            lst_selector = list(range(0, i - h_block))
        # 
        else:
            raise ValueError("Invalid state in gen_ij_selector")
        # 
        if verbose: print(i, lst_selector)
        # 
        return lst_selector
    
        


    def pick_lowest_pe(self, df_pe_res : pd.DataFrame, verbose=False) -> list :
        self.mean_pe_lowest = df_pe_res['CV'].min()
        self.pq_order_updated = df_pe_res.loc[df_pe_res['CV'] == self.mean_pe_lowest]['pq_order'].to_list()
        if verbose : print(self.pq_order_updated)

        return self.pq_order_updated








def pe_weights(df: pd.DataFrame, method={'berman', 'uniform'}) -> list:
    """
    This is for a demonstration. It turns out, run OLS with proper selection of observations
    do the same as the Burman et al. (1994). Additionally, the weight that Burman et al. proposed is the same as the uniform weighting.
    Generate weights for the h-block cross-validation based on Burman et al. (1994) or uniform weighting.
    
    :param df: Input DataFrame.
    :param method: Weighting method to use ('berman' or 'uniform').
    """
    n = len(df.index)
    h = np.ceil(n / 6).astype(int)
    weights = []
    
    if method == 'berman':
        for j in range(1, n + 1):
            if 1 <= j <= h:
                weight = 1 / (n - j - h)
                row = [0 if 1 <= i <= j + h else weight for i in range(1, n + 1)]
                
            elif h < j <= n - h:
                weight = 1 / (n - 2 * h - 1)
                row = [0 if j - h <= i <= j + h else weight for i in range(1, n + 1)]

            elif n - h < j <= n:
                weight = 1 / (j - h - 1)
                row = [0 if j - h <= i <= n else weight for i in range(1, n + 1)]

            else : 
                raise ValueError("Invalid weight index")
            
            weights.append(row)

    elif method == 'uniform':
        for j in range(1, n + 1):
            if 1 <= j <= h:
                row = [0 if 1 <= i <= j + h else 1 for i in range(1, n + 1)]

            elif h < j <= n - h:
                row = [0 if j - h <= i <= j + h else 1 for i in range(1, n + 1)]

            elif n - h < j <= n:
                row = [0 if j - h <= i <= n else 1 for i in range(1, n + 1)]

            else : 
                raise ValueError("Invalid weight index")
            
            row = list(np.divide(row, [sum(row)] * len(row)))
            weights.append(row)
    
    return weights
