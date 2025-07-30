import warnings
import pandas as pd
import numpy as np
from patsy import dmatrices
from statsmodels.api import OLS
from sklearn.preprocessing import QuantileTransformer
from joblib import Parallel, delayed
import torch

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
                        maxL:int = 24,
                        device='cuda',
                        multiprocessing : {None, 'cpu', 'gpu'} = None,
                        ):
        """
        Initialize the SieveBootstrap class.
        Parameter(s):
            CTRF: CTRF model object from CTRF class.
        """
        self.df_reg     = df_reg
        self.CTRF       = CTRF              # CTRF model object from CTRF class
        self.ctrf       = ctrf              # CTRF model name
        self.B          = B                 # number of bootstrap samples
        # 
        self.maxL   = maxL                  # maximum lag for AR process
        self.maxT   = 0                     # maximum time point initialization
        self.lscross_ids = df_reg[self.CTRF.var_id].unique() # list of cross-sectional id

        # regression results - TRF or CTRF
        if self.CTRF.reg_res_ctrf == {} :
            self.CTRF_reg_res = CTRF.reg_res_trf
            # p,q powered normalized temperature
            self.CTRF_df_Xs      = CTRF.Xs.copy()
        else :
            self.CTRF_reg_res = CTRF.reg_res_ctrf
            self.CTRF_df_Xs      = CTRF.Xs_ctrf.copy()
        # 
        # assign time index
        self._assign_time_index()
        self.maxT = self.df_reg['T'].max()  # maximum time point update
        # generate y_hat 
        self.df_reg['y_hat'] = self.CTRF_reg_res.predict()
        # generate fitted resiual of the CTRF model (eta_hat).
        self.df_reg['eta_hat'] = self.CTRF_reg_res.resid
        #
        self.cross_id_col :str = self.CTRF.var_id
        self.date_col  :str = self.CTRF.var_date
        self.dependent_var   :str = self.CTRF.reg_res_trf.model.endog_names
        # 
        self.df_e_hat = {}
        self.B_df_y_star = {}
        self.AR_model_results = {}
        # 
        self.device = device
        self.muliprocessing = multiprocessing

    #######################################################################################
    # main process.
    #######################################################################################
    def sieve_bootstrap_i(self, cross_id:str, 
                          save_results_e_hat:bool = False, ):
        '''
        1. get the fitted residuals of CTRF model (eta_hat)
        2. run AR LS
            get the fitted residuals of AR model (e_hat;WN)
        3. repeat the following B times (999 times)
            bootstrapping e_star from e_hat.
            generate eta_star sequentially from the AR model + e_star.
            construct y_star from y_hat + eta_star.
            * implement multiprocessing : none, CPU, GPU.
        4. find the 2.5 and 97.5 percentiles of 999 y_star and 1 y_actual.
        '''
        # 1. initialize, get the fitted resiudal
        dfi = (self.df_reg.loc[self.df_reg[self.cross_id_col] == cross_id][[
                    self.date_col, self.cross_id_col, self.dependent_var,'y_hat', 'eta_hat', 'T']].copy()
                    .sort_values(by='T',ascending=False).reset_index(drop=True))
        #       demeaning eta_hat (CTRF model fitted residual)
        dfi['eta_hat'] = dfi['eta_hat'] - dfi['eta_hat'].mean()

        # 2. run AR OLS
        df_e_hat_i, AR_result_i = self.run_AR_model(dfi, self.dependent_var, self.maxL)
        #       save results
        if save_results_e_hat:
            self.df_e_hat[cross_id] = df_e_hat_i
            self.AR_model_results[cross_id] = AR_result_i


        # 3. bootstrapping e_star
        if self.muliprocessing == 'gpu' :
            # drop consumed lags by AR model.
            dfi = dfi.loc[dfi['T'] >= self.maxL]
            df_y_star_B = self.bootstrap_e_star_torch_gpu(dfi, df_e_hat_i, cross_id)
        else :
            # initialize bootstrapped results.
            df_y_star_B = {}
            # 'y_star_0' reserved for the actual y.
            df_y_star_B[0] = dfi[['T',self.dependent_var]].rename(columns={self.dependent_var:'y_star_0'})
            df_y_star_B[0] = (df_y_star_B[0].sort_values(by='T', ascending=False)
                                            .reset_index(drop=True)
                                            .drop(columns=['T']))
            #       loop over B times, b=0 means actual y.
            # CPU multiprocessing
            if self.muliprocessing == 'cpu' :
                # b=1,...,B: Compute bootstrap iterations in parallel.
                bootstrap_results = Parallel(n_jobs=-1)(
                    delayed(self.bootstrap_iteration)(dfi, df_e_hat_i, AR_result_i, cross_id)
                    for b in range(1, self.B + 1)
                )
                # Rename columns for each bootstrap result and store in the dictionary.
                for b, df_result in enumerate(bootstrap_results, start=1):
                    df_y_star_B[b] = df_result.rename(columns={'y_star': f'y_star_{b}'})
    
            # No multiprocessing
            else :
                for b in range(1, self.B + 1):
                    df_y_star_B[b] = (self.bootstrap_iteration(dfi, 
                                                            df_e_hat_i, 
                                                            AR_result_i, 
                                                            cross_id,)
                                            .rename(columns={'y_star':f'y_star_{b}'}))
            # concatenate y_star_b for B+1 times (1000 times).
            df_y_star_B = pd.concat([df_y_star_B[b] for b in range(0, self.B + 1)], axis=1)
            df_y_star_B = df_y_star_B.dropna(axis=0)
            # save results
            self.B_df_y_star[cross_id] = df_y_star_B


        # 4. find the 2.5 and 97.5 percentiles of B y_star and 1 y_actual.
        df_bs_result = dfi[[self.date_col, self.cross_id_col, self.dependent_var,'y_hat','T']]
        if self.muliprocessing == 'gpu':
            # Vectorized computation of quantiles for all rows
            tau025_all, tau975_all = self.find_bootstrap_tails_torch_gpu(df_y_star_B)
            # Align df_bs_result to the index of df_y_star_B (since dropna() may have reduced the number of rows)
            df_bs_result = df_bs_result.loc[df_y_star_B.index].copy()
            # Now assign the quantile arrays
            df_bs_result['y_star_tau025'] = tau025_all
            df_bs_result['y_star_tau975'] = tau975_all
        else :
            for idx in df_y_star_B.index:
                tau025, tau975 = self.find_bootstrap_tails(idx, df_y_star_B)
                df_bs_result.loc[idx, 'y_star_tau025'] = tau025
                df_bs_result.loc[idx, 'y_star_tau975'] = tau975

        df_bs_result = df_bs_result.dropna(axis=0)
        return df_bs_result
    #######################################################################################



    #######################################################################################
    # Subpprocesses
    #######################################################################################
    # 2. AR model
    def run_AR_model(self, df, depvar:str, maxL:int):
        # no constant model because 'eta_hat' will be feed in as demeaned.
        ARspec = f'{depvar} ~ ' + ' + '.join([f'L{lags}' for lags in range(1, maxL + 1)]) + ' - 1'
        # transform the dataframe to wide style for AR model.
        df_wide = self._transform_to_wide(df, var_to_tr=depvar)
        y, X = dmatrices(ARspec, df_wide, return_type='dataframe')
        AR_result = OLS(y, X).fit()
        df_e_hat = pd.DataFrame(AR_result.resid, columns=['e_hat'])
        return df_e_hat, AR_result



    # 3 bootstrapping sub process
    def bootstrap_e_star_torch_gpu(self, dfi, df_e_hat_i, cross_id):
        """
        Vectorized version of the bootstrap iterations using the GPU.
            y_star = y_hat + bootstrapped_residuals
        where bootstrapped_residuals are computed by randomly permuting the fitted residuals.
        This function returns a DataFrame with shape (n, B) where n is the number of time points.
        """
        device = self.device
        B = self.B
        maxL = self.maxL
        # Convert the fitted residuals to a GPU tensor.
        # (df_e_hat_i is assumed to be a DataFrame with a column 'e_hat'.)
        e_hat = torch.tensor(df_e_hat_i['e_hat'].values, dtype=torch.float32, device=device)
        n = e_hat.shape[0]

        # Create a batch of B random permutations.
        # Generate a (B, n) tensor of random values and compute argsort along dim=1.
        rand_vals = torch.rand(B, n, device=device)
        perm_indices = torch.argsort(rand_vals, dim=1)
        # Expand e_hat to shape (B, n) and gather the bootstrapped residuals.
        e_hat_expanded = e_hat.unsqueeze(0).expand(B, n)
        e_star_batch = torch.gather(e_hat_expanded, 1, perm_indices)
        
        # # Create the time index values (if needed) – note: here we mimic your T-values.
        # T_values = torch.arange(n, device=device, dtype=torch.int32).flip(0) + maxL

        # Convert y_hat to GPU tensor.
        y_hat = torch.tensor(dfi['y_hat'].values, dtype=torch.float32, device=device)
        # Expand y_hat to shape (B, n)
        y_hat_expanded = y_hat.unsqueeze(0).expand(B, n)
        
        # Compute bootstrapped y_star replicates (for the simplified model).
        y_star_batch = y_hat_expanded + e_star_batch
        # Optionally, if you need to run an AR simulation loop, you would need to recast that loop 
        # into vectorized PyTorch code here.

        # Move the results back to CPU as a NumPy array.
        y_star_np = y_star_batch.cpu().numpy()
        
        # Create a DataFrame: each column corresponds to one bootstrap replicate.
        df_y_star_B = pd.DataFrame(y_star_np.T, columns=[f'y_star_{b}' for b in range(1, B + 1)])
        # If you need to include the actual (observed) y (i.e. b=0), add that column:
        df_actual = dfi[['T', self.dependent_var]].rename(columns={self.dependent_var: 'y_star_0'})
        df_actual = (df_actual.sort_values(by='T', ascending=False)
                                .reset_index(drop=True)
                                .drop(columns=['T']))
        df_y_star_B.insert(0, 'y_star_0', df_actual['y_star_0'])
        
        # Remove any rows that might contain NaN (if needed).
        df_y_star_B = df_y_star_B.dropna(axis=0)
        
        # Optionally, save the results.
        self.B_df_y_star[cross_id] = df_y_star_B
        
        return df_y_star_B





    def bootstrap_iteration(self, dfi, df_e_hat_i, AR_result_i, cross_id):
        # bootstrapping e_star
        df_e_star_ib = self.bootstrapping_e_star(df_e_hat_i)
        # generate eta_star
        df_eta_star_ib = self.gen_eta_star_by_ARmodel(dfi, df_e_star_ib, AR_result_i)
        # generate y_star
        df_y_star_ib = self.generate_y_star_b(df_eta_star_ib, cross_id)[['y_star']]   
        return df_y_star_ib


    def bootstrapping_e_star(self, df_e_hat):
        maxL = self.maxL
        rndgen = np.random.default_rng(seed=None)
        # pick without replacement.
        rnd_idx = rndgen.choice(df_e_hat.index, size=df_e_hat.__len__())
        df_e_star = df_e_hat.loc[rnd_idx].copy().reset_index()
        df_e_star['e_star'] = df_e_star['e_hat']
        df_e_star['T'] = df_e_star.index[::-1] + maxL
        return df_e_star[['e_star','T']]
    


    #  a dataframe that initializaztion and addition "eta_star" for AR process.
    #  "eta_star" = "rho"*"eta_star_t-1" + "e_star"
    def gen_eta_star_by_ARmodel(self, df_eta_i, df_e_star, model_AR_i):
        # initialize - to generate the first "eta_star" from the AR model.
        # generate eta_star from the AR model.
        #   add e_star
        # eta_star = rho*eta_star_t-1 + e_star
        maxL, maxT = self.maxL, self.maxT
        df_estimate_rho_hat = self._transform_to_wide(
                                    df_eta_i.reset_index(drop=True),
                                    var_to_tr='eta_hat'
                                )
        # initialization - drop L0 ('eta_hat')
        df_eta_star_b = df_estimate_rho_hat.copy().drop(columns=['eta_hat'])
        # Fill NaN except the initial point
        df_eta_star_b.loc[df_eta_star_b['T'] != maxL, df_eta_star_b.filter(like="L").columns] = np.nan
        # merge e_star
        df_eta_star_b = df_eta_star_b.merge(df_e_star, on='T', how='left')
        # 
        # fill lagged eta. Once "eta_star" is generated, it is used for the next time point.
        for _, T in enumerate(range(maxL, maxT + 1)):
            # print(T)
            # Copy lagged data from previous T
            locationL0 = df_eta_star_b['T'] == T
            locationL1 = df_eta_star_b['T'] == T - 1
            # 
            if T > maxL:
                # L1 - assign L0 at T-1 as L1 at T.
                df_eta_star_b.loc[locationL0,  # locate T (T is the row).
                                df_eta_star_b.filter(like='L').columns[0]   # locate L1 (L is the column).
                                    ] = df_eta_star_b.loc[locationL1,
                                                        'eta_star'].values # assign L1 value as T-1 L0 value (eta_star).
                # L2 to L24
                df_eta_star_b.loc[locationL0, 
                                df_eta_star_b.filter(like='L').columns[1:]
                                    ] = df_eta_star_b.loc[locationL1, 
                                                            df_eta_star_b.filter(like='L').columns[:-1]].values
            # Generate eta_star = eta_hat + e_star
            eta_hat = (df_eta_star_b.loc[locationL0].filter(like='L').dot(model_AR_i.params))
            e_star = df_eta_star_b.loc[locationL0, 'e_star']
            df_eta_star_b.loc[locationL0, ['eta_hat']] = eta_hat
            df_eta_star_b.loc[locationL0, ['eta_star']] = eta_hat + e_star
        return df_eta_star_b




    # generate y_star_b
    def generate_y_star_b(self, df_eta_star_ib, fips):
        _df = self.df_reg[['fips','T','y_hat']].loc[self.df_reg[self.CTRF.var_id] == fips].copy()
        _df['eta_star'] = _df.merge(df_eta_star_ib, on='T', how='inner')['eta_star']
        _df = _df.dropna()
        _df['y_star'] = _df['y_hat'] + _df['eta_star']
        # _df = _df.sort_values(by='date', ascending = False).reset_index(drop=True)
        return _df





    
    # 4. caluclate quantiles.

    def find_bootstrap_tails(self, idx, df_y_star_B):            
        # find the 2.5 and 97.5 percentiles of 999 y_star and 1 y_actual.
        bootstrapped_y_star_at_T = np.array(df_y_star_B.loc[idx]).reshape(-1,1)
        quantile_transform = QuantileTransformer()
        quantile_transform.fit(bootstrapped_y_star_at_T)
        # Compute the 2.5th and 97.5th percentiles
        tau025 = quantile_transform.inverse_transform(np.array([[0.025]]))[0][0]
        tau975 = quantile_transform.inverse_transform(np.array([[0.975]]))[0][0]
        # 
        return tau025, tau975



    def find_bootstrap_tails_torch_gpu(self, df_y_star_B):
        """
        Compute the 2.5th and 97.5th percentiles for each row of df_y_star_B.

        Parameters:
        df_y_star_B : pandas.DataFrame
            DataFrame where rows correspond to a given index (e.g., time point)
            and columns to different bootstrap replicates.
        use_gpu : bool
            Whether to use the GPU via PyTorch.
        device : str
            Device to use if use_gpu is True (e.g., 'cuda').

        Returns:
        tau025, tau975 : np.ndarray
            Arrays of quantile values for each row.
        """
        # Convert the entire DataFrame to a tensor
        tensor_y = torch.tensor(df_y_star_B.values, dtype=torch.float32)
        tensor_y = tensor_y.to(self.device)

        # Compute quantiles along the column dimension (dim=1)
        tau025 = torch.quantile(tensor_y, 0.025, dim=1)
        tau975 = torch.quantile(tensor_y, 0.975, dim=1)

        # Move results back to CPU and convert to NumPy arrays
        tau025 = tau025.cpu().numpy()
        tau975 = tau975.cpu().numpy()

        return tau025, tau975

















    # assign T to the dataframe
    def _assign_time_index(self):
        df_T = self.df_reg[['date']].drop_duplicates().copy()
        df_T['T'] = (df_T['date'].rank() -1).astype(int)
        self.df_reg['T'] = self.df_reg.drop(columns='T', errors='ignore').merge(df_T, on='date', how='left')['T'] 


    

    # transform TS style df to wide style df to estimate AR model.
    # demeaning the feeded df (it may redundant but it is not hearting.)
    def _transform_to_wide(self, df_ts_style, var_to_tr:str = 'eta_hat'):
        # 
        maxL, maxT = self.maxL, self.maxT
        colnames = [f'L{L}' for L in range(1, maxL + 1)]
        # 
        _df = df_ts_style.reset_index(drop=True)
        _var = var_to_tr
        # 
        # demeaning
        _df[_var] = _df[_var] - _df[_var].mean()
        # 
        df_wide_style = None
        for T in range(0 + maxL, maxT + 1)[::-1]:
            # Subset _df_eta with a specific id and time
            _df_t = _df.loc[(_df['T'] == T)]
            # Generate each row of lags
            _df_eta_ctrf_L = (
                            _df.loc[_df['T'].isin(range(T - maxL, T))][[_var]]
                                .set_index(np.array(colnames)).T
                            )
            _df_eta_ctrf_L.set_index(_df_t.index, inplace=True)
            _df_t = pd.concat([_df_t, _df_eta_ctrf_L], axis=1)
            # 
            if T == maxT:
                df_wide_style = _df_t
            else:
                df_wide_style = pd.concat([df_wide_style, _df_t], axis=0)
        return df_wide_style





