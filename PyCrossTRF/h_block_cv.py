import pandas as pd
import numpy as np
from statsmodels.api import OLS
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ValueWarning
from pandas.errors import SettingWithCopyWarning
from .utils import ctrf_utils

# Suppress specific warnings
simplefilter('ignore', ValueWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)

class CV_h_block:
    """
    Cross-validation using h-block method to report Prediction Error (PE) over different p and q orders.
    """

    def __init__(self, df: pd.DataFrame, dep: str, indep: str, pq_order: dict):
        """
        Initialize the CV_h_block class with the dataset and parameters.
        
        :param df: Input DataFrame containing dependent and independent variables.
        :param dep: Name of the dependent variable.
        :param indep: Name of the independent variable.
        :param pq_order: Dictionary specifying the maximum order of p and q.
        """
        self.df = df[[dep, indep]].copy().reset_index(drop=True)
        self.dep = dep
        self.indep = indep
        self.pq_order = pq_order
        self.df_pq_powered = pd.DataFrame()
        self.pq_combination = []
        # 
        self.df_pe_res = pd.DataFrame()
        self.pq_order_updated = []
        self.mean_pe_lowest = float()


    def gen_po_powered_df(self, verbose=False):
        """
        Generate the dataframe for OLS based on the maximum p and q order.
        """
        self.df_pq_powered = pd.concat([
            self.df[[self.dep]],
            ctrf_utils().pq_powering(temp_s=self.df[[self.indep]], pq_order=self.pq_order)
        ], axis=1)
        
        if verbose: print(self.df_pq_powered)
        
        return self.df_pq_powered


    def gen_pq_combination(self, verbose=False) -> list:
        """
        Generate all possible combinations of p and q orders.
        """
        pq_combination = []
        for q in range (1, self.pq_order['q']+1) :
            for p in range(1, self.pq_order['p']+1) : 
                pq_combination += [{'p':p, 'q':q}]

        self.pq_combination = pq_combination
        if verbose: print(self.pq_combination)
        
        return self.pq_combination


    def gen_regressor_lst(self, pq_order: dict, verbose=False) -> list:
        """
        Generate the list of regressors for OLS based on p and q order.
        """
        x_lst = ['Intercept']
        for p in range(1, pq_order['p']+1) : 
            x_lst += [f'{self.indep}_{p}0']
        
        for q in range (1, pq_order['q']+1) :
            x_lst += [f'{self.indep}_0{q}c']
            x_lst += [f'{self.indep}_0{q}s']
        
        if verbose: print(x_lst)
        
        return x_lst
    

    def gen_ij_selector(self, i: int, n: int, h: int, verbose=False) -> list:
        """
        Generate the list of indices for training set based on current index i and block size h.
        
        :param i: Index of the test observation.
        :param n: Total number of observations.
        :param h: Block size.
        """
        if i <= h:
            lst_selector = list(range(i + h + 1, n))

        elif i > h and i < n - h:
            lst_selector = list(range(0, i)) + list(range(i + h + 1, n))

        elif i >= n - h:
            lst_selector = list(range(0, i - h))

        else:
            raise ValueError("Invalid state in gen_ij_selector")
        
        if verbose: print(i, lst_selector)
        
        return lst_selector
    

    def h_block_e_pe(self, ys:pd.DataFrame, xs:pd.DataFrame, crit_segment: int = 6, verbose=False) -> float:
        """
        Calculate the average prediction error (PE) using the h-block cross-validation method.
        
        :param ys: Dependent variable data.
        :param xs: Independent variable data.
        :param crit_segment: Number of segments to determine block size h.
        """
        n = len(ys.index)
        crit_h = np.ceil(n / crit_segment).astype(int)
        pe_lst = []
        
        for i in range(0, n):
            selector = self.gen_ij_selector(i=i, n=n, h=crit_h)
            model = OLS(ys.iloc[selector], xs.iloc[selector]).fit()
            pe = model.predict(xs.iloc[i]).iloc[0]
            pe_lst.append(pe)
        
        pe_mean = np.mean(pe_lst)
        
        if verbose: print(pe_mean, pe_lst)
        
        return pe_mean
        

    def pick_lowest_pe(self, df_pe_res : pd.DataFrame, verbose=False) -> list :
        self.mean_pe_lowest = df_pe_res['CV'].min()
        self.pq_order_updated = df_pe_res.loc[df_pe_res['CV'] == self.mean_pe_lowest]['pq_order'].to_list()
        if verbose : print(self.pq_order_updated)

        return self.pq_order_updated


    def compute_cv(self, show_res=False,verbose=False) -> tuple:
        """
        Compute the cross-validation prediction error for all p and q combinations.
        
        :param verbose: Print detailed information if True.
        """
        pq_combination = self.gen_pq_combination(verbose=verbose)
        df = self.gen_po_powered_df(verbose=verbose)
        dep = self.dep
        
        pe_res = []
        
        for pq_order in pq_combination:
            ys = df[[dep]]
            xs = df[self.gen_regressor_lst(pq_order=pq_order, verbose=verbose)]
            # average prediction error of the give pq_order
            pe_mean = self.h_block_e_pe(ys=ys, xs=xs, crit_segment=6, verbose=verbose)
            pe_res.append([pq_order, pe_mean])
        
        self.df_pe_res = pd.DataFrame(pe_res, columns=['pq_order', 'CV'])
        self.pick_lowest_pe(df_pe_res= self.df_pe_res, verbose=verbose)

        if verbose or show_res: print(self.df_pe_res)
        
        return self.pq_order_updated, self.mean_pe_lowest



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
