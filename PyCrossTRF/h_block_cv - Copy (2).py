import pandas as pd
import numpy as np
from statsmodels.api import OLS
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ValueWarning
from pandas.errors import SettingWithCopyWarning
from .utils import ctrf_utils

from joblib import Parallel, delayed
import psutil

# Suppress specific warnings
simplefilter('ignore', ValueWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)



'''
'h-block Cross Validation' for TRF and CTRF model.
    - It shares the basic structure with 'cross_trf.py' and use 'utils.py'.
        For CTRF, normlized p,q powered df is needed.
    - normalized and p,q powered df, select covariates as p, and q, and covariates.
    - 'pq_order' is the maximum order of p and q.
'''
class CV_h_block:
    """
    Cross-validation using h-block method to report Prediction Error (PE) over different p and q orders.
    """
    def __init__(self, df: pd.DataFrame, dep: str, indep: str, pq_order: dict, time_id:str = None, cross_id:str=None, cov_scale:dict = None):
        """
        Initialize the CV_h_block class with the dataset and parameters.
        
        :param df: Input DataFrame containing dependent and independent variables.
        :param dep: Name of the dependent variable.
        :param indep: Name of the independent variable.
        :param pq_order: Dictionary specifying the maximum order of p and q.
        """
        self.time_id = time_id
        if time_id is None :
            df[time_id] = df.index
        if cross_id is None :
            df[cross_id] = '00000'
        # sort
        df = df.sort_values(by=[time_id, cross_id]).reset_index(drop=True).copy()
        df['i'] = df.index
        # 
        self.df_grouped_index_block = df.groupby(time_id, as_index=False).agg(i = ('i', list))
        # 
        self.y : pd.Series = df[dep]
        self.r : pd.Series = df[indep]

        self.pq_order = pq_order

        if cov_scale is None : 
            self.scale_method = 'minimax'
        else :
            self.scale_method = cov_scale[f'{indep}']['scale']
        
        self.df_pq_powered = pd.DataFrame()
        self.pq_combination = []
        # 
        self.df_pe_res = pd.DataFrame()
        self.pq_order_updated = []
        self.mean_pe_lowest = float()



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
        s = ctrf_utils().normalizer(x = self.r, method = 'minmax')
        Xs = ctrf_utils().gen_df_xs(temp_s=s, pq_order = pq_order)
        # print(Xs)

        # i over the time index - the index of the df_grouped_index_block.
        n = self.df_grouped_index_block.index.max()
        squared_errors = []
        for i in self.df_grouped_index_block.index :
            '''
            This is the core of the h-block cross-validation.
              - selecting i, j to remove serial correlation.
              'selector' slicing blocks.
            '''
            selector_i = self.df_grouped_index_block.loc[i]['i']
            selector_j = self.gen_ij_selector(
                                i = i, 
                                n = n, 
                                h = 6  # 'h' is the critical valaue proposed by burman et al. 
                                )
            model = OLS(self.y.iloc[selector_j], Xs.iloc[selector_j]).fit()
            print(model.params)
            df_predicted_i = pd.DataFrame(model.predict(Xs.iloc[selector_i]), columns=['Xs_hat'])
            
            df_predicted_i['y'] = self.y.iloc[selector_i]
            df_predicted_i['squared_error'] = (df_predicted_i['y'] - df_predicted_i['Xs_hat']) ** 2
            squared_errors.append(df_predicted_i['squared_error'])
        mse = pd.concat(squared_errors, axis=0).mean()
        if verbose : 
            return mse, pd.concat(squared_errors, axis=0)
        else :
            return mse  # return mse.

         



    def gen_pq_combination(self, pq_order : dict = None, verbose=False) -> list:
        """
        Generate all possible combinations of p and q orders.
        """
        if pq_order is None : pq_order = self.pq_order
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
        
        :param i: Index of the test observation.
        :param n: Total number of observations.
        :param h: Block size.
        """
        if i <= h:
            lst_selector = list(range(i + h + 1, n))
        # 
        elif i > h and i < n - h:
            lst_selector = list(range(0, i)) + list(range(i + h + 1, n))
        # 
        elif i >= n - h:
            lst_selector = list(range(0, i - h))
        # 
        else:
            raise ValueError("Invalid state in gen_ij_selector")
        # 
        if verbose: print(i, lst_selector)
        # 
        return lst_selector
        
        # a panel data, groupping cross sectional data by time series id.
        #   probably padded by number of groups in index.
        # else :

    

    # # mapping i, j to date value.
    # def mapping_ij_to_date(self, lst_selector:list) -> list:
    #     if self.date_var is None : 
    #         return lst_selector # return in index values so that '.iloc' can be used.
        
    #     else :
    #         # mapping date to index, return index values so that '.iloc' can be used.
    #         # mapping 0,..,i...,n to date values.
    #         df_mapping_date_i = (self.df_date.drop_duplicates().reset_index(drop=True)
    #                                 .reset_index()
    #                                 )
            
    #         self.df_date['i'] = self.df_date.index
             


    
    # def h_block_e_pe(self, ys:pd.Series, xs:pd.DataFrame, crit_segment: int = 6, verbose=False) -> float:
    #     """
    #     Calculate the average prediction error (PE) using the h-block cross-validation method with parallel computation.
    #     Ensure that tasks are not allocated to Intel's 'E' cores.
        
    #     :param ys: Dependent variable data.
    #     :param xs: Independent variable data.
    #     :param crit_segment: Number of segments to determine block size h.
    #     """
    #     if self.date_var is None :
    #         n = ys.index
    #     else :
    #         n = 1
    #     crit_h = np.ceil(n / crit_segment).astype(int)

    #     def compute_pe(i):
    #         selector = self.gen_ij_selector(i=i, n=n, h=crit_h)
    #         model = OLS(ys.iloc[selector], xs.iloc[selector]).fit()
    #         return model.predict(xs.iloc[i]).iloc[0]



    #     try:
    #         # pe_lst = Parallel(n_jobs=len(p_cores))(delayed(compute_pe)(i) for i in n)
    #                 # # Set the environment variable to restrict joblib to use only P cores
    #         original_affinity = psutil.Process().cpu_affinity()
    #         p_cores = [core for core in original_affinity if core < 12]  # Assuming P-cores are even-numbered
    #         psutil.Process().cpu_affinity(p_cores)
    #         pe_lst = Parallel(n_jobs=len(p_cores))(delayed(compute_pe)(i) for i in n)
    #         for i in n:
    #             compute_pe(i)
    #     except Exception as e:
    #         print(e)
    #     finally:
    #     #     # Restore the original CPU affinity
    #         psutil.Process().cpu_affinity(original_affinity)
        
    #     pe_mean = np.mean(pe_lst)
        
    #     if verbose: print(pe_mean, pe_lst)
        
    #     return pe_mean
        



    def pick_lowest_pe(self, df_pe_res : pd.DataFrame, verbose=False) -> list :
        self.mean_pe_lowest = df_pe_res['CV'].min()
        self.pq_order_updated = df_pe_res.loc[df_pe_res['CV'] == self.mean_pe_lowest]['pq_order'].to_list()
        if verbose : print(self.pq_order_updated)

        return self.pq_order_updated



    # show C.V. results.
    def compute_cv(self, show_res=False, verbose=False) -> tuple:
        """
        Compute the cross-validation prediction error for all p and q combinations.
        
        :param verbose: Print detailed information if True.
        """
        pq_combination = self.gen_pq_combination(verbose=verbose)
        pe_res = []
        for pq_order in pq_combination:
            if verbose : 
                mse, df_mse = self.compute_prediction_error(pq_order=pq_order, verbose=verbose)
            else :
                mse = self.compute_prediction_error(pq_order=pq_order, verbose=verbose)
            print(pq_order, mse)
            pe_res.append([pq_order, mse])
        
        self.df_pe_res = pd.DataFrame(pe_res, columns=['pq_order', 'CV'])
        self.pick_lowest_pe(df_pe_res= self.df_pe_res, verbose=verbose)

        if verbose or show_res: print(self.df_pe_res,'\n', self.pq_order_updated)
        # 
        if verbose : 
            return self.pq_order_updated[0], self.mean_pe_lowest, df_mse
        else :
            return self.pq_order_updated[0], self.mean_pe_lowest





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
