import pandas as pd
import numpy as np
import torch
from statsmodels.api import OLS
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ValueWarning
from pandas.errors import SettingWithCopyWarning
from .utils import ctrf_utils
import psutil

# Suppress specific warnings
simplefilter('ignore', ValueWarning)
simplefilter('ignore', SettingWithCopyWarning)


class CV_h_block_torch:
    """
    Cross-validation using h-block method implemented in PyTorch to evaluate Prediction Error (PE) over different p and q orders.
    """

    def __init__(self, df: pd.DataFrame, dep: str, indep: str, pq_order: dict, cov_scale: dict = None, device='cuda'):
        self.y = torch.tensor(df[dep].values, dtype=torch.float32, device=device)
        self.r = df[indep].copy()
        self.pq_order = pq_order
        self.scale_method = cov_scale.get(indep, {}).get('scale', 'minmax') if cov_scale else 'minmax'
        self.df_pe_res = pd.DataFrame()
        self.best_pq_order = None
        self.lowest_mean_pe = np.inf
        self.device = device

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

    def h_block_e_pe(self, ys: torch.Tensor, xs: torch.Tensor, crit_segment: int = 6) -> float:
        n = ys.size(0)
        crit_h = int(np.ceil(n / crit_segment))
        pe_lst = []

        for i in range(n):
            selector = self.gen_ij_selector(i, n, crit_h)
            X_train = xs[selector, :]
            y_train = ys[selector].unsqueeze(1)

            # PyTorch least squares using torch.linalg.lstsq
            lstsq_result = torch.linalg.lstsq(X_train, y_train)
            beta_hat = lstsq_result.solution

            # Prediction
            x_test = xs[i, :].unsqueeze(0)
            y_pred = x_test @ beta_hat
            pe_lst.append(y_pred.item())

        return np.mean(pe_lst)


    def compute_cv(self) -> tuple:
        pq_combinations = self.gen_pq_combination()
        normalized_temp = ctrf_utils().normalizer(x=self.r, method=self.scale_method)
        df = ctrf_utils().gen_df_xs(temp_s=normalized_temp, pq_order=self.pq_order)

        xs_tensor = torch.tensor(df.values, dtype=torch.float32, device=self.device)

        pe_results = []
        for pq in pq_combinations:
            regressors = self.gen_regressor_lst(indep=self.r.name, pq_order=pq)
            indices = [df.columns.get_loc(col) for col in regressors]
            xs_subset = xs_tensor[:, indices]

            pe_mean = self.h_block_e_pe(ys=self.y, xs=xs_subset)
            pe_results.append([pq, pe_mean])

        self.df_pe_res = pd.DataFrame(pe_results, columns=['pq_order', 'CV'])
        self.best_pq_order = self.df_pe_res.loc[self.df_pe_res['CV'].idxmin(), 'pq_order']
        self.lowest_mean_pe = self.df_pe_res['CV'].min()

        return self.best_pq_order, self.lowest_mean_pe
