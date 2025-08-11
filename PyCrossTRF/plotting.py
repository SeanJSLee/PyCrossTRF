import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.axes import Axes

from statsmodels.api import OLS
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.regression.linear_model import RegressionResultsWrapper


from typing import Dict, Optional, Union, Literal, Tuple, Any

from itertools import cycle


# from .cross_trf import *
from .cross_trf import CTRF, recover_ctrf, calc_ctrf
from .sieve_bs import SieveBootstrap


class Plot():
    def __init__(self, model:CTRF):
        self.model      = model
        self.df         = model.df
        self.cross_id   = model.cross_id
        self.time_id    = model.time_id
        self.mmt_s      = model.mmt_s
        # self.model      = model.model
        self.reg_res:RegressionResultsWrapper = model.reg_res_trf if model.model == 'trf' else model.reg_res_ctrf # type: ignore
        self.params     = self.reg_res.params
        self.s_pred     = model.s_pred
        self.pq_order   = model.pq_order



    def residual(self, 
                 resid_mean_only = False, 
                 additonal_scatter_fips:list = [('12057','Hillsborough County, FL', 'r')]
                 ):
        # Residual dataframe gen from 'ctrf_result'
        df_resid = self.df[[self.cross_id, self.time_id]].copy()
        df_resid['resid'] = np.array(self.reg_res.resid)
        df_resid_mean = df_resid[['date','resid']].groupby('date').agg('mean').reset_index()
        if resid_mean_only:
            return df_resid_mean
        # 
        fig, ax = plt.subplots(figsize=(12,8))
        ax.scatter(df_resid['date'], df_resid['resid'], 
                   s=0.2, alpha= 0.5, c='b',
                   label=f'{self.reg_res.y.name}')
        ax.plot(df_resid_mean['date'],df_resid_mean['resid'], 
                lw=3, alpha=1, c='k',
                label=f'{self.reg_res.y.name} mean')
        if len(additonal_scatter_fips) > 0 :
            for fips in additonal_scatter_fips :
                df_i = df_resid.loc[df_resid['fips']==fips[0]]
                ax.scatter(df_i['date'], 
                           df_i['resid'],
                    s=2, alpha= 1,c=fips[2], label=f'FIPS: {fips[0]}, {fips[1]}')
        ax.legend()
        plt.ylabel(f'Residual plot - {self.reg_res.y.name}')
        plt.xlabel('Date')
        plt.show()


    def plot_recover_trf(self, ctrf:Optional[list] = None):
        base_trf = (recover_ctrf( s_pred    = self.s_pred, 
                                  coef      = self.params, 
                                  pq_order  = self.pq_order, 
                                  ctrf_pred_lst = ctrf,
                                  verbose   = False)
                    )
        return base_trf 
            #     {'base':  0      0.886116
            #               1      0.887168
            #       Length: 1000, dtype: float64}


    def ctrf_plot(  self, 
                    x_scale = pd.DataFrame(np.arange(0,1,0.001), columns=['quantile'])['quantile'], 
                    label   = 'base TRF',
                    y_label = 'Deaths per 100k in 30 days (ln)', 
                    # x_label = f'Temperature {quantile}',
                    color = 'k',
                    ymin = None,
                    ymax = None,
                    # x_axis_scale = ('q', pd.DataFrame(np.arange(0,1,0.001), columns=['quantile']) ),
                    # map_quantile_temp = pd.DataFrame(),
                    save_fig = '',
                    figsize = (8,8),
                    ctrf =[],
                    ctrf_base_comb = True
                ):
        # 
        
        df = pd.DataFrame(self.plot_recover_trf(ctrf))
        # 
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
        # 
        # MMT
        MMT_s = self.mmt_s
        # 
        # scale
        x_scale_10th = x_scale.loc[int(len(x_scale)*0.1)]
        x_scale_90th = x_scale.loc[int(len(x_scale)*0.9)]
        mmt_r = x_scale.loc[int(MMT_s * len(x_scale))]
        # print(x_scale_10th, x_scale_90th, mmt_r)

        # figure
        fig, ax = plt.subplots(figsize=figsize)
        # main plot
        if ctrf_base_comb :
            ax.plot(x_scale, df['base'], color=color, label=label)
            ax.fill_between(x_scale, y1=ax.get_ylim()[0], y2=ax.get_ylim()[1], where=(x_scale <= x_scale_10th), color='gray', alpha=0.2)
            ax.fill_between(x_scale, y1=ax.get_ylim()[0], y2=ax.get_ylim()[1], where=(x_scale >= x_scale_90th), color='gray', alpha=0.2)
        if len(ctrf) > 0 :
            for idx, i_trf in enumerate(df.columns[1:]) :
                if ctrf_base_comb :
                    ax.plot(x_scale, df['base'] + df[i_trf], color = next(colors), label = i_trf)
                    # ax.fill_between(x_scale, df['base'] + df[i_trf], where=(x_scale < 0.1), color='gray', alpha=0.5)
                else :
                    ax.plot(x_scale,              df[i_trf], color = next(colors), label = i_trf)
                    ax.fill_between(x_scale, y1=ax.get_ylim()[0], y2=ax.get_ylim()[1], where=(x_scale <= x_scale_10th), color='gray', alpha=0.2)
                    ax.fill_between(x_scale, y1=ax.get_ylim()[0], y2=ax.get_ylim()[1], where=(x_scale >= x_scale_90th), color='gray', alpha=0.2)
        # 
        # dotting mmt
        if ctrf_base_comb :
            coord_mmt_x = mmt_r
            coord_mmt_y = df['base'].loc[int(MMT_s * len(x_scale))]
            ax.plot(coord_mmt_x, coord_mmt_y, 'ro')
            ax.axvline(x=coord_mmt_x, color='gray', lw=0.5,linestyle='--')  # Vertical line to x-axis
            ax.axhline(y=coord_mmt_y, color='gray', lw=0.5,linestyle='--')  # Horizontal line to y-axis
            ax.annotate(f'\n\nMMT = {coord_mmt_x:.2f}', xy=(coord_mmt_x, ax.get_ylim()[0]), xytext=(coord_mmt_x, -0.1),
                textcoords='offset points', ha='center', va='top', fontsize=10, color='black')
            ax.annotate(f'\n\nE[y|MMT] = {coord_mmt_y:.2f}', xy=(0, coord_mmt_y), xytext=(-15, coord_mmt_y),
                textcoords='offset points', ha='right', va='center', fontsize=10, color='black')
        # 
        if ax.get_ylim()[0] < 0 < ax.get_ylim()[1] :
            ax.axhline(y=0, color='k', lw=0.5)  # Vertical y=0
        # 
        ax.set_ylim(ymin = ymin, ymax=ymax)
        ax.set_xlim(0.04, 0.96)
        ax.legend()
        plt.ylabel(y_label)
        plt.xlabel(x_scale.name)
        # 
        if len(save_fig) > 1 :
            plt.savefig(save_fig)
        # 
        plt.show()



    def trf_plot(self,
                 ax:Optional[Axes] = None,
                 df_pred:bool = False,
                 label_yaxis:Optional[str] = None,  # type: ignore
                 label_xaxis:Optional[str] = 'Normalized Temp',
                 **keywarg
                 ):
        '''
        Generate 'TRF model' or 'base TRF of CTRF model' plot.
        '''
        # Recover TRF (base TRF) from estimated result.
        if self.model.model == 'trf':
            label = 'DRF'
        else : 
            label = 'base DRF'
        if label_yaxis is None: label_yaxis:str = self.model.y.name  # type: ignore
        # 
        df_plot = pd.DataFrame(self.plot_recover_trf())
        trf_var:str = df_plot.columns[0]
        df_plot['s'] = df_plot.index / len(df_plot.index)

        sns.set(style="white", font="Times New Roman", rc={"font.size": 10})
        if ax :
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(figsize = (10,6))
        # fig.subplots_adjust(left=0.1, right=0.9)  # type: ignore
        ax.plot(df_plot['s'], df_plot[trf_var], label=label, **keywarg)
        if ax is None :
            # Set the axis labels
            ax.set_xlabel(label_xaxis) # type: ignore
            ax.set_ylabel(label_yaxis)

            # Display the legend
            ax.legend()
            plt.tight_layout()
            plt.show()
        elif ax :
            return ax
        elif df_pred :
            return df_plot






# ########################################################
# # 3d plotting
# def plot_3d_ctrf(model, 
#                    pq_order, 
#                    var:str = 'date_lin', 
#                    z_vec = np.arange(0,1.01,0.01),
#                    val_return_ctrf_only = False):
#     # 
#     if len(z_vec) == 0 :
#         return pd.DataFrame(arry['base'],columns=[0.0])
#     else :
#         df = pd.DataFrame()
#         # 
#         for z_val in z_vec :
#             arry = (ctrf..recover_ctrf([ 
#                                 {var:[z_val]}], 
#                                 model.s_pred, 
#                                 model.reg_res_ctrf.params, 
#                                 pq_order, 
#                                 verbose=False))
#             pick = list(arry.keys())[1]
#             if val_return_ctrf_only :
#                 df_i = pd.DataFrame(arry[pick], columns=[z_val])
#             else :
#                 df_i = pd.DataFrame(arry[pick] + arry['base'], columns=[z_val])

#             df = pd.concat([df,df_i],
#                         axis=1)
#         return df
# ################################