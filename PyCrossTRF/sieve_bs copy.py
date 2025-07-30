import os, pickle, warnings
import pandas as pd
import geopandas as gpd
import numpy as np

from patsy import dmatrices
from statsmodels.api import OLS

import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer

from joblib import Parallel, delayed

warnings.filterwarnings(action='ignore',category=UserWarning)


# 
# Sieve Bootstrap for TRF and CTRF models.
# It takes the data, fitted model from "CTRF" class,
#   then it returns 2.5 and 97.5 percentiles confidence interval.

class SieveBootstrap:
    def __init__(self,  CTRF = None, 
                        df_reg = pd.DataFrame(), 
                        ctrf:str = 'base',
                        B:int=999, 
                        by_cross_section:bool=True):
        """
        Initialize the SieveBootstrap class.

        Parameter(s):
            CTRF: CTRF model object from CTRF class.
        """
        self.df_reg     = df_reg
        self.CTRF       = CTRF               # CTRF model object from CTRF class
        self.ctrf       = ctrf               # CTRF model name
        self.B          = B                  # number of bootstrap samples
        self.by_cross_section = by_cross_section # bootstrapping by cross section
        # 
        self.maxL   = 24                # maximum lag for AR process
        self.maxT   = 0                 # maximum time point

        # regression results - TRF or CTRF
        if self.CTRF.reg_res_ctrf == {} :
            self.CTRF_reg_res = CTRF.reg_res_trf
            # p,q powered normalized temperature
            self.CTRF_df_Xs      = CTRF.Xs.copy()
        else :
            self.CTRF_reg_res = CTRF.reg_res_ctrf
            self.CTRF_df_Xs      = CTRF.Xs_ctrf.copy()

        self.df_eta_hat = {}
        self.df_estimate_rho_hat = {}
        self.reg_res_rho_hat = {}
        self.df_e_hat = {}
        self.df_e_star = {}
        self.B_eta_star_y_star = {}
        self.B_df_y_star = {}




    # generate fitted residuals from the TRF or CTRF "eta_hat"
    def generate_residuals(self):
        """
        Generate "eta_hat" fitted residuals.
        Parameter(s):
        Return(s):
            eta_hat: Fitted residuals.
        """
        var_date    = self.CTRF.var_date
        var_id      = self.CTRF.var_id
        ctrf        = self.ctrf
        # cross-sectional id & time variable that retains the order of the data
        _df_id      = self.df_reg[[var_date, var_id]].copy()

        # generate T = 0,...,T
        _df_T = (_df_id[[var_date]].drop_duplicates()
                                    .sort_values(by=var_date)
                                    .reset_index(drop=True)
                                    .reset_index().rename(columns={'index':'T'})
                                    )
        # update maxT
        self.maxT = _df_T['T'].max()

        # 
        _reg_res = self.CTRF_reg_res
        _df_Xs      = self.CTRF_df_Xs
        # # regression results - TRF or CTRF
        # if self.CTRF.reg_res_ctrf == {} :
        #     _reg_res = self.CTRF.reg_res_trf
        #     # p,q powered normalized temperature
        #     _df_Xs      = self.CTRF.Xs.copy()
        # else :
        #     _reg_res = self.CTRF.reg_res_ctrf
        #     _df_Xs      = self.CTRF.Xs_ctrf.copy()

        # variables for the ctrf model
        _var_pred = list(self.CTRF.Xs.columns)
        if ctrf != 'base' :
            _var_pred = _var_pred + list(self.CTRF.Xs_ctrf.filter(like=ctrf).columns)

        # gen "eta_hat"
        # 'base TRF' or 'base TRF' + 'ctrf'
        print('Vars to generate eta_hat', _var_pred)
        _df_eta_hat = pd.DataFrame(_df_Xs[_var_pred].dot(_reg_res.params[_var_pred]),
                                   columns=['eta_hat'])
        _df_eta_hat = pd.concat([_df_id, _df_eta_hat], axis=1)
        # merge with T
        _df_eta_hat = _df_eta_hat.merge(_df_T, on=var_date, how='left')
        # 
        # save to the dictionary "eta_hat"
        self.df_eta_hat = _df_eta_hat
        


    # generate a new fitted residueal 'e_hat' for each bootstrap sample.
    def generate_e_hat(self):
        # reshape to wide while grouping fips to estimate rho_hat.
        # add fips and t for safe.
        # 24 lags...
        maxL, maxT = self.maxL, self.maxT
        var_date    = self.CTRF.var_date
        var_id      = self.CTRF.var_id
        ctrf        = self.ctrf
        _df_eta = self.df_eta_hat
        # gnereate column names for L = 1,2,...,24
        colnames = [f'L{L}' for L in range(1, maxL + 1)]
        # 
        # iterate over crossectional id (fips), generate each row with lagged eta_hat.
        # Function to process each fips
        def process_fips(fips):
            _df_eta_lagged = None
            for T in range(0 + maxL, maxT + 1)[::-1]:
                # Subset _df_eta with a specific id and time
                _df = _df_eta.loc[(_df_eta['T'] == T) & (_df_eta[var_id] == fips)]

                # Generate each row of lags
                _df_eta_ctrf_L = (
                                _df_eta.loc[
                                        (_df_eta['T'].isin(range(T - maxL, T))) & (_df_eta[var_id] == fips)
                                    ][['eta_hat']]
                                    .set_index(np.array(colnames)).T
                                )

                _df_eta_ctrf_L.set_index(_df.index, inplace=True)
                _df = pd.concat([_df, _df_eta_ctrf_L], axis=1)

                if T == maxT:
                    _df_eta_lagged = _df
                else:
                    _df_eta_lagged = pd.concat([_df_eta_lagged, _df], axis=0)

            return fips, _df_eta_lagged

        # Parallel processing for fips loop
        results = Parallel(n_jobs=-1)(
            delayed(process_fips)(fips) for fips in _df_eta[var_id].unique()
        )
        # Collect results into a dictionary
        df_fips_ctrf = {str(fips): df for fips, df in results}
        
        # Combine results into a single DataFrame
        df_estimate_rho_hat = pd.concat([df_fips_ctrf[f'{fips}'] for fips in _df_eta[var_id].unique()], axis=0)

        # save to the dictionary "e_hat"
        self.df_estimate_rho_hat = df_estimate_rho_hat
        

        # Run OLS
        # Generate model specification without constant term
        spec = 'eta_hat ~ ' + ' + '.join([f'L{_}' for _ in range(1, maxL + 1)]) + ' - 1'
        # 
        yy,xx = dmatrices(spec, df_estimate_rho_hat, return_type='dataframe')
        # 
        reg_res_rho_hat = OLS(yy,xx).fit()
        self.reg_res_rho_hat = reg_res_rho_hat

        # generate "e_hat"
        df_e_hat = pd.DataFrame(reg_res_rho_hat.resid, columns=['e_hat']).reset_index()
        df_e_hat = df_e_hat.merge(self.df_reg[[var_date, var_id]].copy(), how='left', left_on='index', right_index=True)
        # print(df_e_hat)
        self.df_e_hat = df_e_hat



    # bootstrapping "e_star" by cross-sectional id.
    def bootstrapping_e_star(self, parallel:bool=True) :
        df_e_star = self.df_e_hat.copy()
        # demeaned with sample mean of e_hat within fips.
        if self.by_cross_section :
            _df_e_star_grouped_mean = df_e_star.groupby('fips')[['e_hat']].mean().reset_index().rename(columns={'e_hat':'e_hat_mean'})
            df_e_star = df_e_star.merge( _df_e_star_grouped_mean,
                                            on='fips', how='left')
            df_e_star['e_hat'] = df_e_star['e_hat'] - df_e_star['e_hat_mean']
        else :
            # demeaned with sample mean of e_hat.
            df_e_star['e_hat'] = df_e_star['e_hat'] - df_e_star['e_hat'].mean()

        if parallel is not True :
            # bootstrapping "e_star" within cross sections from the demeaned "e_hat".
            for b in range(1,self.B + 1)[:]:
                for _, fips in enumerate(df_e_star['fips'].unique()[:]) :
                    rng = np.random.default_rng(seed=None)
                    idx_cross_section = df_e_star[df_e_star['fips']==fips].index
                    random_idx = rng.choice(idx_cross_section,
                                            size=len(idx_cross_section))

                    # generate bootstrapped "eta_star = fitted residual;e + fitted eta"
                    if _ == 0 :
                        # print(_df)
                        _df = pd.DataFrame()
                        # print(_df)
                        _df = df_e_star[['e_hat']]
                        _df[f'e_star_{b}'] = np.nan
                        _df.loc[idx_cross_section, f'e_star_{b}'] = (_df.loc[idx_cross_section,'e_hat'] 
                                                + df_e_star.loc[random_idx,:'e_hat'].set_index(idx_cross_section)['e_hat'])
                    # 
                    else :
                        _df.loc[idx_cross_section, f'e_star_{b}'] =(_df.loc[idx_cross_section,'e_hat']  
                                                + df_e_star.loc[random_idx,:'e_hat'].set_index(idx_cross_section)['e_hat'])

                    
                    # _df = pd.DataFrame(df_e_star.iloc[random_idx][['e_hat']].reset_index()['e_hat'] + df_e_star['e_hat'],
                    #                     columns=[f'e_star_{_}'])
                df_e_star = pd.concat([df_e_star, _df[[f'e_star_{b}']]], axis=1)

        elif parallel :
            # Bootstrapping "e_star" by cross-sectional id.
            def process_bootstrap(b):
                _df = df_e_star[['e_hat']].copy()
                _df[f'e_star_{b}'] = np.nan

                for _, fips in enumerate(df_e_star['fips'].unique()):
                    # initialize random number generator
                    rng = np.random.default_rng(seed=None)
                    # save the original index of the cross section
                    idx_cross_section = df_e_star[df_e_star['fips'] == fips].index
                    # get andomly resampled index from the random number generator
                    random_idx = rng.choice(idx_cross_section, size=len(idx_cross_section), replace=True)

                    # Generate bootstrapped "e_star_b" from resampled "e_hat".
                    _df.loc[idx_cross_section, f'e_star_{b}'] = list(_df.loc[random_idx, 'e_hat'])#.set_index(idx_cross_section)
                    # _df.loc[idx_cross_section, f'e_star_{b}'] = (
                    #     _df.loc[idx_cross_section, 'e_hat'] +
                    #     df_e_star.loc[random_idx, 'e_hat'].set_index(idx_cross_section)
                    # )

                return _df[[f'e_star_{b}']]

            # Parallel execution for b loop
            results = Parallel(n_jobs=-1)(
                delayed(process_bootstrap)(b) for b in range(1, self.B + 1)
            )

            # Combine results with df_e_star
            for b, result in enumerate(results, start=1):
                df_e_star = pd.concat([df_e_star, result], axis=1)

        self.df_e_star = df_e_star



    # generate "eta_star" - generate from "rho_hat" and "e_star".
    def generate_eta_star(self):
        """"
        eta_star" = "rho"*"eta_star_t-1" + "e_star"
        "rho" is the coefficients of the residual AR model.
        "eta_star_t-1" is the lagged "eta_star". Note, "eta_star_0" to "eta_star_-maxL" is "eta_hat"(fitted residual from TRF or CTRF). 
        """
        # 
        df_e_star = self.df_e_star
        df_estimate_rho_hat = self.df_estimate_rho_hat
        reg_res_rho_hat = self.reg_res_rho_hat
        # 
        def sieve_process_b(b, df_estimate_rho_hat, df_e_star, reg_res_rho_hat, CTRF_model_reg_res, maxL, maxT):
            # result df initilizaed with the data to estiamate "rho".
            df_eta_star_b = df_estimate_rho_hat.copy().drop(columns=['eta_hat'])
            # Fill NaN except the initial point
            df_eta_star_b.loc[df_estimate_rho_hat['T'] != maxL, df_estimate_rho_hat.filter(like="L").columns] = np.nan
            # df_eta_star_b['eta'] = np.nan
            # df_eta_star_b['e_star'] = np.nan
            df_eta_star_b['e_star'] = df_e_star[f'e_star_{b}'].values
            df_eta_star_b['eta_star'] = np.nan
            # fill lagged eta. Once "eta_star" is generated, it is used for the next time point.
            for _, T in enumerate(range(maxL, maxT + 1)):
                # Copy lagged data from previous T
                if T > maxL:
                    # L1
                    df_eta_star_b.loc[df_eta_star_b['T'] == T, 
                                    df_eta_star_b.filter(like='L').columns[0]
                                        ] = df_eta_star_b.loc[df_eta_star_b['T'] == T - 1, # make 1 lag.
                                                            'eta_star'].values
                    # L2 to L24
                    df_eta_star_b.loc[df_eta_star_b['T'] == T, 
                                    df_eta_star_b.filter(like='L').columns[1:]
                                        ] = df_eta_star_b.loc[df_eta_star_b['T'] == T - 1, 
                                                                df_eta_star_b.filter(like='L').columns[:-1]].values

                # Generate eta_star = eta + e_star
                # "eta_hat" from the lagged "eta" and "rho_hat"
                eta_hat = df_eta_star_b.loc[df_eta_star_b['T'] == T].filter(like='L').dot(reg_res_rho_hat.params)
                # "e_star" picked from the bootstrapped "e_star_{b}".
                # e_star = df_e_star.loc[df_e_star['index'].isin(df_eta_star_b.loc[df_eta_star_b['T'] == T].index)][f'e_star_{b}'].values
                # e_star = df_eta_star_b.loc[df_eta_star_b.loc[df_eta_star_b['T'] == T].index]['e_star'].values
                e_star = df_eta_star_b.loc[df_eta_star_b['T'] == T]['e_star']
                eta_star = eta_hat + e_star

                df_eta_star_b.loc[df_eta_star_b['T'] == T, ['eta_hat']] = np.array(eta_hat)
                # df_eta_star_b.loc[df_eta_star_b['T'] == T, ['e_star']] = np.array(e_star)
                df_eta_star_b.loc[df_eta_star_b['T'] == T, ['eta_star']] = np.array(eta_star)

            # Fit y and merge
            df_eta_star_b = df_eta_star_b.merge(pd.DataFrame(CTRF_model_reg_res.predict(), columns=['fitted_y']), 
                                                                how='left', 
                                                                left_index=True, 
                                                                right_index=True)
            df_eta_star_b['y_star'] = df_eta_star_b['fitted_y'] + df_eta_star_b['eta_star']

            return b, df_eta_star_b

        # Parallel execution of the loop
        results = Parallel(n_jobs=-1)(
            delayed(sieve_process_b)(b, 
                               df_estimate_rho_hat = df_estimate_rho_hat, 
                               df_e_star = df_e_star, 
                               reg_res_rho_hat = reg_res_rho_hat, 
                               CTRF_model_reg_res = self.CTRF_reg_res, 
                               maxL = self.maxL, 
                               maxT = self.maxT) for b in range(1, self.B + 1)[:]
        )

        # Collect results into the dictionary
        B_eta_star_y_star = {str(b): df for b, df in results}
        # return B_eta_star_y_star['1']
        self.B_eta_star_y_star = B_eta_star_y_star
        # 


    # generate bootstrapped y_star
    def generate_y_star(self):
        # construct bootstrapped y_star
        # B_df_y_star = df_to_reg[['ln_daily_death100k']].rename(columns={'ln_daily_death100k':'y_star_0'}).copy()
        # B_df_y_star = pd.DataFrame()
        # for b in range(1,self.B + 1)[:]:
        #     # B_df_y_star[f'y_star_{b}'] = B_eta_star_y_star[f'{b}']['y_star']
        #     B_df_y_star = pd.concat([B_df_y_star, self.B_eta_star_y_star[f'{b}'][['y_star']].rename(columns={'y_star':f'y_star_{b}'})])
        # B_df_y_star = self.df_reg[['ln_daily_death100k']].rename(columns={'ln_daily_death100k':'y_star_0'}).copy().merge(B_df_y_star, how='right', left_index=True, right_index=True)
        # B_df_y_star = B_df_y_star.reset_index()
        # # B_df_y_star
        # self.B_df_y_star = B_df_y_star
        B_df_y_star = self.df_reg[['ln_daily_death100k']].rename(columns={'ln_daily_death100k':'y_star_0'}).copy()
        for b in range(1,1000)[:]:
            B_df_y_star = B_df_y_star.merge(self.B_eta_star_y_star[f'{b}'][['y_star']].rename(columns={'y_star':f'y_star_{b}'}),
                                        how='right',
                                        left_index=True,
                                        right_index=True,
                                        )
        B_df_y_star = B_df_y_star.reset_index()
        self.B_df_y_star = B_df_y_star



def process_sieve_bootstrap(CTRF, df_reg, ctrf):
    pass











