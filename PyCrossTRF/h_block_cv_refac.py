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
simplefilter('ignore', SettingWithCopyWarning)


class CV_h_block:
    """
    Cross-validation using h-block method to evaluate Prediction Error (PE) over different p and q orders.
    """

    def __init__(self, df: pd.DataFrame, dep: str, indep: str, pq_order: dict, cov_scale: dict = None):
        self.y = df[dep].copy()
        self.r = df[indep].copy()
        self.pq_order = pq_order
        self.scale_method = cov_scale.get(indep, {}).get('scale', 'minmax') if cov_scale else 'minmax'
        self.df_pe_res = pd.DataFrame()
        self.best_pq_order = None
        self.lowest_mean_pe = np.inf

    def gen_pq_combination(self) -> list:
        return [{'p': p, 'q': q} for q in range(1, self.pq_order['q'] + 1)
                for p in range(1, self.pq_order['p'] + 1)]

    def gen_regressor_lst(self, indep: str, pq_order: dict) -> list:
        regressors = ['Intercept']
        regressors += [f'{indep}_{p}0' for p in range(1, pq_order['p'] + 1)]
        regressors += [f'{indep}_0{q}{suffix}' for q in range(1, pq_order['q'] + 1) for suffix in ['c', 's']]
        return regressors

    @staticmethod
    def gen_ij_selector(i: int, n: int, h: int) -> list:
        if i <= h:
            return list(range(i + h + 1, n))
        elif h < i < n - h:
            return list(range(0, i)) + list(range(i + h + 1, n))
        elif i >= n - h:
            return list(range(0, i - h))
        raise ValueError("Invalid state in gen_ij_selector")

    def h_block_e_pe(self, ys: pd.Series, xs: pd.DataFrame, crit_segment: int = 6) -> float:
        n = len(ys)
        crit_h = int(np.ceil(n / crit_segment))

        def compute_pe(i):
            selector = self.gen_ij_selector(i, n, crit_h)
            model = OLS(ys.iloc[selector], xs.iloc[selector]).fit()
            return model.predict(xs.iloc[i])[0]

        original_affinity = psutil.Process().cpu_affinity()
        # p_cores = [core for core in original_affinity if core < 12]
        # psutil.Process().cpu_affinity(p_cores)

        try:
            pe_lst = Parallel(n_jobs=-1)(delayed(compute_pe)(i) for i in range(n))
        finally:
            psutil.Process().cpu_affinity(original_affinity)

        return np.mean(pe_lst)

    def compute_cv(self) -> tuple:
        pq_combinations = self.gen_pq_combination()
        normalized_temp = ctrf_utils().normalizer(x=self.r, method=self.scale_method)
        df = ctrf_utils().gen_df_xs(temp_s=normalized_temp, pq_order=self.pq_order)

        pe_results = []
        for pq in pq_combinations:
            regressors = self.gen_regressor_lst(indep=self.r.name, pq_order=pq)
            pe_mean = self.h_block_e_pe(ys=self.y, xs=df[regressors])
            pe_results.append([pq, pe_mean])

        self.df_pe_res = pd.DataFrame(pe_results, columns=['pq_order', 'CV'])
        self.best_pq_order = self.df_pe_res.loc[self.df_pe_res['CV'].idxmin(), 'pq_order']
        self.lowest_mean_pe = self.df_pe_res['CV'].min()

        return self.best_pq_order, self.lowest_mean_pe
