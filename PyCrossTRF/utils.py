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

class ctrf_utils:
    def __init__(self):
        pass   


    def pq_powering(self, temp_s : pd.Series, pq_order : dict = {'p':4, 'q':1},  var_name=None, covariate_s:pd.Series = None, verbose=False):
        '''
        tranform 'temp_s' based on 'p', 'q' order.
        if 'covaraite_s' provided, multiplying it.
        '''
        
        if covariate_s is not None : 
            var_name = covariate_s.name
            
        if covariate_s is None :
            covariate_s = pd.Series([1]*len(temp_s.index))

        if var_name is None :
            var_name = temp_s.name
        
        
        df = pd.DataFrame()
        for p in range(0, pq_order['p']+1 ) :
            # the first constant column name as "intercept"
            if (p == 0) & (var_name == temp_s.name):
                df.loc[:,f'Intercept'] = covariate_s.values * np.power(temp_s, p) 
            else:
                df.loc[:,f'{var_name}_{p}0'] = covariate_s.values * np.power(temp_s, p)
        # trignometric functions.
        for q in range(1, pq_order['q']+1 ) :
            df.loc[:,f'{var_name}_0{q}c'] = covariate_s.values * np.cos(temp_s * 2 * q *np.pi)
            df.loc[:,f'{var_name}_0{q}s'] = covariate_s.values * np.sin(temp_s * 2 * q *np.pi)
        # 
        if verbose :
            # print('sssssssssssss',temp_s.columns)
            print(f"p:{pq_order['p']}, q:{pq_order['q']}, colname:{var_name}")
        # 
        return df
    

    def gen_df_xs(self, temp_s : pd.Series, pq_order : dict = {'p':4, 'q':1}, xs_cov : pd.DataFrame = None, verbose=False) :
        """
        Generate pq powered df based on temp and covariates.
        """
        df = self.pq_powering(temp_s=temp_s, pq_order=pq_order)
        if xs_cov is not None :
            for cov in xs_cov.columns : 
                df = pd.concat([df, self.pq_powering(temp_s=temp_s, pq_order=pq_order, covariate_s=xs_cov[[cov]])], axis=1)
        if verbose : print(df)

        # print('lllllllllll',df.columns)

        return df






    def normalizer(self, x : pd.Series, method={'minmax','linear','time','standard','quantile'}, 
                   temp : pd.Series=None, MMT=None, verbose=False) :
        """
        Normalize the temperature and covariate variables.

        Parameters:
        show_df (bool): If True, print and return the DataFrame with normalized values.

        Returns:
        pd.DataFrame: DataFrame with normalized values if show_df is True.
        """
        xs = pd.Series()      # innitialization
        # MMT is not provided, ordinary normalization or standardization
        # if MMT is None :
        # scikit-lean 'minmax_scale' based code. Transform X var to unit inverval.
        if method == 'minmax':
            print('Minmax normalization')
            # xs = ( df[var] - df[var].min() ) / (df[var].max() - df[var].min())
            scaler = MinMaxScaler()
            xs = scaler.fit_transform(pd.DataFrame(x))
            if verbose :
                print(f'min: {scaler.data_min_.mean():2f}, max: {scaler.data_max_.mean():2f}')
                

        # run L1 normalization - make a vectro to a unit norm.    
        elif method == 'linear':
            # shifter that start from 0
            shifter = -x.min()
            scaler = x.max() - x.min()
            xs = (x + shifter) / scaler
            if verbose :
                print(f'min: {x.min()}, max: {x.max()}')
            # lt = Normalizer(norm='l1')
            # xs = lt.fit_transform(df[[var]])
            # if verbose :
            #     print(lt.n_features_in_)

        elif method == 'time':
            # use index
            print('time w/o MMT')
            scaler = len(x.index)
            xs = x.index / scaler
            if verbose :
                print(f'min: {x.min()}, max: {x.max()}')

        # ordinary standardization
        elif method == 'standard' :
            if MMT is None :
                # xs = (df[var] - df[var].mean()) / np.std(df[var],ddof=1)
                scaler = StandardScaler()
                xs = scaler.fit_transform(x)
                if verbose :
                    print(f'E({x.name}): {scaler.mean_[0]:.2f}, SD({x.name}): {np.sqrt(scaler.var_[0]):.2f}')

            # When MMT is provided, standardize while centering at MMT.
            elif MMT is not None :
                print(f'{x.name}Standardize centering at MMT - set sample mean of X as X value at the MMT')
                # slice interval near the MMT, find closest two temp's location
                # temp_minimax = df.loc[df[temp] >= MMT][temp].min()
                temp_minimax = temp[temp >= MMT].min()
                temp_maximin = temp[temp <= MMT].max()
                # temp_maximin = df.loc[df[temp] <= MMT][temp].max()

                # cov_minimax = df.loc[df[temp]==temp_minimax][var].mean()
                # temp locate minimax then get the index, use the index get x's mean.
                cov_minimax = x[temp[temp==temp_minimax].index].mean()
                # cov_maximin = df.loc[df[temp]==temp_maximin][var].mean()
                cov_maximin = x[temp[temp==temp_maximin].index].mean()

                # weigted average based on the inverse distance between the True MMT.
                if temp_minimax == temp_maximin :
                    weighted_avg = cov_minimax
                else :
                    weighted_avg = np.average([cov_minimax, cov_maximin], 
                                        #  inverse distance weight.
                                          weights=[temp_minimax/(abs(temp_maximin - temp_minimax)),
                                                   temp_maximin/(abs(temp_maximin - temp_minimax))])
                # sample variace of x (dof adjusted)
                sigma = (((x - weighted_avg)**2).sum() / (len(x-1))) ** .5
                # standariazed covariate vector
                xs = (x - weighted_avg) / sigma
                if verbose :
                    print(f'E({x.name}): {weighted_avg:.2f}, SD({x.name}): {sigma:.2f}, MMT:{MMT:.2f}, Cov Minimax: {cov_minimax:.2f}, Cov Maximin: {cov_maximin:.2f}')


        elif method == 'quantile' :
            print('quantile')
            # quantile transformer
            scaler = QuantileTransformer(n_quantiles=100)
            xs = scaler.fit_transform(df[[var]])
            if verbose :
                print(scaler.n_quantiles_)
                print(scaler.quantiles_)
                # print(qt.references_)

        
        xs = np.array(xs).flatten()
        xs = pd.Series(xs, name=x.name)
        return  xs