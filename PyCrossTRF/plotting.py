import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource

from itertools import cycle


from .cross_trf import *


class Plot():
    def __init__(self, df, ctrf_res, var_id:str='fips', var_date:str='date'):
        self.df = df
        self.ctrf_res = ctrf_res
        self.var_id = var_id
        self.var_date = var_date
        self.mmt_s = ctrf_res.MMT_s


    def residual(self, resid_mean_only = False, additonal_scatter_fips:list = [('12057','Hillsborough County, FL', 'r')]):
        # Residual dataframe gen from 'ctrf_result'
        df_resid = self.df[[self.var_id, self.var_date]].copy()
        df_resid['resid'] = np.array(self.ctrf_res.reg_res_ctrf.resid)
        df_resid_mean = df_resid[['date','resid']].groupby('date').agg('mean').reset_index()
        if resid_mean_only:
            return df_resid_mean
        # 
        fig, ax = plt.subplots(figsize=(12,8))
        ax.scatter(df_resid['date'], df_resid['resid'], 
                   s=0.2, alpha= 0.5, c='b',
                   label=f'{self.ctrf_res.y.name}')
        ax.plot(df_resid_mean['date'],df_resid_mean['resid'], 
                lw=3, alpha=1, c='k',
                label=f'{self.ctrf_res.y.name} mean')
        if len(additonal_scatter_fips) > 0 :
            for fips in additonal_scatter_fips :
                df_i = df_resid.loc[df_resid['fips']==fips[0]]
                ax.scatter(df_i['date'], 
                           df_i['resid'],
                    s=2, alpha= 1,c=fips[2], label=f'FIPS: {fips[0]}, {fips[1]}')
        ax.legend()
        plt.ylabel(f'Residual plot - {self.ctrf_res.y.name}')
        plt.xlabel('Date')
        plt.show()


    def plot_recover_trf(self, ctrf = []):
        base_trf = (CTRF_recover()
                        .recover_ctrf( ctrf, 
                            self.ctrf_res.s_pred, 
                            self.ctrf_res.reg_res_ctrf.params, 
                            self.ctrf_res.pq_order, 
                            verbose=False)
                    )
        return base_trf 
            #     {'base':  0      0.886116
            #               1      0.887168
            #       Length: 1000, dtype: float64}


    def ctrf_plot(  self, 
                    x_scale = pd.DataFrame(np.arange(0,1,0.001), columns=['quantile'])['quantile'], 
                    label   = 'base TRF',
                    y_label = 'Averaged daily deaths per 100k', 
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
        mmt_s = self.mmt_s
        # 
        # scale
        x_scale_10th = x_scale.loc[int(len(x_scale)*0.1)]
        x_scale_90th = x_scale.loc[int(len(x_scale)*0.9)]
        mmt_r = x_scale.loc[int(mmt_s * len(x_scale))]
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
            coord_mmt_y = df['base'].loc[int(mmt_s * len(x_scale))]
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
#             arry = (ctrf.CTRF_recover().recover_ctrf([ 
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