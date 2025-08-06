import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from itertools import chain
from patsy import dmatrices # type: ignore
from statsmodels.api import OLS
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from warnings import simplefilter
from typing import (
    Optional, List, Dict, Any, Tuple, Iterable, Union, Literal
)
from statsmodels.tools.sm_exceptions import ValueWarning
from pandas.errors import SettingWithCopyWarning

from statsmodels.tsa.seasonal import seasonal_decompose

from PyCrossTRF.cross_trf import CTRF

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
            # xs = ( df[var] - df[var].min() ) / (df[var].max() - df[var].min())
            scaler = MinMaxScaler()
            xs = scaler.fit_transform(pd.DataFrame(x))
            if verbose :
                print('Minmax normalization')
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
            scaler = len(x.index)
            xs = x.index / scaler
            if verbose :
                print('time w/o MMT')
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
                if verbose: print(f'{x.name}Standardize centering at MMT - set sample mean of X as X value at the MMT')
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
            # quantile transformer
            scaler = QuantileTransformer(n_quantiles=100)
            xs = scaler.fit_transform(df[[var]])
            if verbose :
                print('quantile')
                print(scaler.n_quantiles_)
                print(scaler.quantiles_)
                # print(qt.references_)

        
        xs = np.array(xs).flatten()
        xs = pd.Series(xs, name=x.name)
        return  xs
    


class Detrending:
    def __init__(
                self, 
                df=pd.DataFrame,
                dt_var = 'daily_death100k',
                id  = 'fips',
                name_date = 'date',
                model:{'additive','multiplicative'}='multiplicative',
                period:int = 12,
                two_sided=False
                ) :
        # 
        self.df = df
        self.dt_var  = dt_var
        self.id = id
        self.name_date = name_date
        self.model = model
        self.period = period 
        self.two_sided = two_sided
        # 
    

    def estm(self, df = None) :
        if df is None :
            df = self.df[self.dt_var]
        # else :
        #     df = df[self.dt_var]
        # ['trend', 'seasonal','resid']
        decomp = seasonal_decompose(df,
                                    model=self.model,
                                    period=self.period,
                                    two_sided=self.two_sided
                                    )
        # 
        res = {}
        res['trend']    = decomp.trend
        res['seasonal'] = decomp.seasonal
        res['resid']    = decomp.resid
        # 
        return res['trend'], res['seasonal'], res['resid']
    
    
    def df_detreding(self) : 
        df = self.df
        id = self.id
        df_agg = pd.DataFrame()
        for _ in df[id].unique() :
            df_i = df.loc[df[id] == _][[id,self.name_date,self.dt_var]].copy()
            df_i = df_i.sort_values(by=self.name_date).reset_index(drop=True)
            try :
                (df_i[f'{self.dt_var}_trend'], 
                    df_i[f'{self.dt_var}_seasonal'], 
                    df_i[f'{self.dt_var}_resid']) = self.estm(df_i[self.dt_var])
            except :
                continue
            df_agg = pd.concat([df_agg, df_i
                                # pd.concat([ df_i,
                                #             df_i[f'{self.dt_var}_trend'],
                                #             df_i[f'{self.dt_var}_seasonal'],
                                #             df_i[f'{self.dt_var}_resid']],
                                #             axis=1)
                                ], axis=0)

        # return pd.concat([self.df[[self.id, self.name_date]],
        #                   df_agg.drop(columns=[self.dt_var], errors='ignore')],
        #                   axis=1)
        return df_agg.drop(columns=[self.dt_var], errors='ignore')
    




class Normalizer:
    def __init__(self, 
                 ctrf_model:Optional[CTRF] = None
                 ) -> None:
        if ctrf_model  :
            self.df         = ctrf_model.df
            # self.cross_id   = ctrf_model.cross_id
            self.transformer_path = ctrf_model.transformer_path
            self.mmt_r      = ctrf_model.mmt_r
            self.mmt_s      = ctrf_model.mmt_s


    def normalizer( self,
                    trans_var: str,
                    method: Literal['raw','minmax','linear','time', 'standard','quantile'],
                    df : Optional[pd.DataFrame] = None, 
                    cross_id : Optional[str] = None ,
                    verbose=False,
                    )  : 
        """
        Normalize the temperature and covariate variables.
        'cross_id' should provide for proper normalization.
        """
        if df is None: df = self.df
        # 
        # 
        def _apply_transform(   method:Literal['minmax','linear','standard','quantile'], 
                                crossid:Optional[str] = None
                                ) -> pd.DataFrame : #type: ignore
            # 
            # normalization ignoring 'cross_id'
            if cross_id is None :
                # 
                if (method == 'minmax') or (method == 'linear'):
                    transformer = MinMaxScaler()
                elif method == 'standard':
                    transformer = StandardScaler()
                elif method == 'quantile' :
                    transformer = QuantileTransformer(
                                    output_distribution='uniform',
                                    subsample=None, # type: ignore
                                    random_state=None)
                # 
                transformed = transformer.fit_transform(df[[trans_var]].rename(columns={trans_var:'nogroup'})).ravel()
                return pd.DataFrame({f'{trans_var}_{method}_transformer':[transformer] * len(df),
                                         f'{trans_var}_{method}':transformed}, 
                                         index=df.index) 
            # 
            elif cross_id :
                # 
                def _transform_minmax(g : pd.DataFrame, method=method, trans_var=trans_var, cross_id=cross_id):
                    # 
                    if (method == 'minmax') or (method == 'linear'):
                        transformer = MinMaxScaler()
                    elif method == 'standard':
                        transformer = StandardScaler()
                    elif method == 'quantile' :
                        transformer = QuantileTransformer(
                                        output_distribution='uniform',
                                        subsample=None, # type: ignore
                                        random_state=None)
                    # 
                    fips = g[cross_id].iloc[0]
                    transformed = transformer.fit_transform(g[[trans_var]].rename(columns={trans_var:fips},errors='ignore')).ravel() # type: ignore
                    return pd.DataFrame({f'{trans_var}_{method}_transformer_{cross_id}':[transformer] * len(g), # type: ignore
                                         f'{trans_var}_{method}':transformed}, 
                                         index=g.index)
                # 
                return df[[trans_var,cross_id]].groupby(cross_id, observed=True, group_keys=False).apply(_transform_minmax)
        # 
        # 

        if method == 'raw':
            xs = df[trans_var]


        elif method in(['minmax','linear','standard','quantile']):
            return _apply_transform(method=method, crossid=cross_id) # type: ignore
        

        elif method == 'time':
            # use index
            scaler = len(x.index)
            xs = x.index / scaler
            if verbose :
                print('time w/o MMT')
                print(f'min: {x.min()}, max: {x.max()}')


        # xs = np.array(xs).flatten()
        # xs = pd.Series(xs, name=x.name)
        # return  xs
    

    def centering(self,
                  method: Literal['raw','minmax','linear','time','quantile']):
        '''
        'centering' is essential process to estimate 'ctrf' model.
        To properly estiamted, first, run the model and find 'mmt(mmt_s)' then
            normalize covariates again to them have 'mean 0' at the 'mmt(mmt_s)'. 
        'temp' variable forced to have [0,1] interval, but covariates are strandardized.
        '''
    