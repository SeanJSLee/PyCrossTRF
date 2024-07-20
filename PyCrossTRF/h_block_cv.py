import pandas as pd
import numpy as np
# from itertools import chain
from patsy import dmatrices
from statsmodels.api import OLS
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression
from warnings import simplefilter
from statsmodels.tools.sm_exceptions import ValueWarning
from pandas.errors import SettingWithCopyWarning


from multiprocessing import Pool, cpu_count


# import cross_trf
# import PyCrossTRF.utils as utils

from .utils import ctrf_utils


simplefilter('ignore', ValueWarning)
simplefilter(action="ignore", category=SettingWithCopyWarning)






class CV_h_block():
    def __init__(self, df: pd.DataFrame, dep: str, indep: str, pq_order: dict):
        # initialize
        self.df = df[[dep,indep]].copy().reset_index(drop=True)
        self.dep = dep
        self.indep = indep
        self.pq_order = pq_order
        # 
        self.df_pq_powered = pd.DataFrame()
        self.pq_combination = []        # p,q combinations to calculate criteria
        self.mse = {}
        self.order_updated = dict()

    # def
        

# Cross validation
    #  1.h-block c.v. method
    #  2.report E(Prediction Error)
    #  3.repeat over p and q order
    def gen_po_powered_df(self, verbose=False):
        df_pq_powered = pd.concat( [ self.df[[self.dep]], # y
                                     ctrf_utils().pq_powering(temp_s=self.df[[self.indep]],
                                        pq_order=self.pq_order) # X
                                    ],
                                    axis=1
                                    )
        
        self.df_pq_powered = df_pq_powered
        # 
        if verbose : print(df_pq_powered)
        #
        return df_pq_powered


    def gen_pq_combination(self , verbose=False):
        pq_combination = []
        for q in range (1, self.pq_order['q']+1) :
            for p in range(1, self.pq_order['p']+1) : 
                pq_combination += [{'p':p, 'q':q}]
        # 
        self.pq_combination = pq_combination
        if verbose : print(self.pq_combination)
        #
        return pq_combination
    

    def gen_regressor_lst(self, pq_order:dict, verbose=False):
        x_lst = []
        x_lst += ['Intercept']

        for p in range(1, pq_order['p']+1) : 
            x_lst += [f'{self.indep}_{p}0']

        for q in range (1, pq_order['q']+1) :
            x_lst += [f'{self.indep}_0{q}c']
            x_lst += [f'{self.indep}_0{q}s']
        
        if verbose : print(x_lst)

        return x_lst
    

    def gen_ij_selector(self, i:int, n:int, h:int, verbose=False) :
        # i indicates the test obs.
        # for i in range(n) :
        lst_selector = []
        # j indicates training obs.
        for j in range(n) :
            if (i <= h) :
                # left training set: remove from i until i+h+1.
                # always leave h number of obs the right side of i.
                lst_selector = list(range(i+h+1, n))
                
            elif (i > h) & (i < n-h) :
                # middle: leave left and right side each, h obs
                lst_selector = list(range(0, i)) + list(range(i+h+1, n))
                
            elif (i >= n-h) :
                # right: remove h obs left and rest of the obs on the right.
                lst_selector = list(range(0, i - h))
                
            else :
                raise

        if verbose : print(i,lst_selector)

        return lst_selector
        


    

    def h_block_e_pe(self, ys, xs, crit_segment:int = 6, multi_processing=False, verbose=False):
        # determin h = n / 6 ; followed by Berman 1994.
        # shifting index by 1 in python.
        n = ys.index.max() + 1
        crit_h = np.ceil( (n) / crit_segment ).astype(int)
        pe_sum = []
        # 
        if not multi_processing :
            # 
            for i in range(n) :
                selector = self.gen_ij_selector(i=i, n=n, h=crit_h)
                # print(ys)
                # print(selector)
                # print(ys.iloc[selector])
                # print(xs.iloc[selector])
                model = OLS(ys.iloc[selector], xs.iloc[selector]).fit()
                pe = model.predict(xs.iloc[i]).iloc[0]
                pe_sum += [pe]
                # if verbose : print(i, pe)
            pe_mean = np.array(pe_sum).mean()
            if verbose : print(pe_mean, pe_sum)
        # 
        elif multi_processing :
            

            with Pool(processes=cpu_count()) as pool :
                args = [(i, ys, xs, crit_h) for i in range(n)]
                pe_sum = pool.map(multi_pe, args)

            # with Pool(processes=cpu_count()) as pool:
            #     args = [(i, ys, xs, crit_h, self.indep, self.dep, self.pq_order) for i in range(n)]
            #     pe_sum = pool.map(multi_pe, args)

            pe_mean = np.mean(pe_sum)

            if verbose: print(pe_mean, pe_sum)
        # 
        return pe_mean





    def compute_cv(self, multi_processing=False, verbose = False):
        # 1.grap pq combination
        # 2.gen pq ordered data
        # 3.run h-block and find mse
        # print('compute')
        pq_combination = self.gen_pq_combination(verbose=verbose)
        
        # 
        df = self.gen_po_powered_df(verbose=verbose)
        name_y = self.dep
        # 
        pe_res = []
        # 
        for idx, pq_order in enumerate(pq_combination) :
            ys = df[[name_y]]
            xs = df[self.gen_regressor_lst(pq_order=pq_order, verbose=verbose)]
            # average prediction error of the give pq_order
            pe_mean = self.h_block_e_pe(ys=ys, xs=xs, crit_segment=6, multi_processing=multi_processing, verbose=verbose)
            print(pe_mean)
            pe_res += [[pq_order, pe_mean]]
        if verbose : print(pd.DataFrame(pe_res, columns=['pq_order','CV']))
        return pe_res
                
      
        

# for multiprocessing
# 

# def multi_pe(self,args) :
#     i, ys, xs, crit_h = args
#     selector = self.gen_ij_selector(i=i, n=len(ys), h=crit_h)
#     model = OLS(ys.iloc[selector], xs.iloc[selector]).fit()
#     pe = model.predict(xs.iloc[i]).iloc[0]
#     return pe

def multi_pe(args):
    i, ys, xs, crit_h = args
    selector = gen_ij_selector(i, len(ys), crit_h)
    model = OLS(ys.iloc[selector], xs.iloc[selector]).fit()
    pe = model.predict(xs.iloc[i]).iloc[0]
    return pe


def gen_ij_selector(i:int, n:int, h:int, verbose=False) :
    # i indicates the test obs.
    # for i in range(n) :
    lst_selector = []
    # j indicates training obs.
    for j in range(n) :
        if (i <= h) :
            # left training set: remove from i until i+h+1.
            # always leave h number of obs the right side of i.
            lst_selector = list(range(i+h+1, n))
            
        elif (i > h) & (i < n-h) :
            # middle: leave left and right side each, h obs
            lst_selector = list(range(0, i)) + list(range(i+h+1, n))
            
        elif (i >= n-h) :
            # right: remove h obs left and rest of the obs on the right.
            lst_selector = list(range(0, i - h))
            
        else :
            raise ValueError("Invalid selector state.")

    if verbose : print(i,lst_selector)

    return lst_selector
 



    # 
    # h-block C.V.
    def h_block_cv(self, df:pd.DataFrame = pd.DataFrame()):
        # # 1. construct data for ctrf_base
        # df_reg = self.df_reg.copy().reset_index(drop=True)
        # print(self.spec)
        # # 2. slice data
        # y, x = dmatrices(self.spec, df_reg, return_type='dataframe')
        y = df[df.columns[:1]]
        Xs = df[df.columns[1:]]
        # determin h = n / 6 ; followed by Berman 1994.
        # shifting index by 1 in python.
        h = np.ceil((y.index.max()+1) / 6).astype(int)
        # h=3
        # print(h)
        n = y.index.max()
        #         # 
        weight_sum = 0
        mse = 0
        stat_pe = []
        # 
        weights = []
        mses = []
        # 
        for j in range(0, n+1)[:] :
            # for i in range(1, n) :
            # 
            # if ((j+1) >= 1) & ((j+1) <= h) :
            if ((j+1) >= 1) & ((j+1) <= h) :
                # training set remove j + h index
                # selec_row = list(range(j-1+h, n))
                selec_row = list(range((j+1)+1+h, n+1))
                weight_ij   = 1 / (n-j-h)
                # print(j)
                # 
            # elif (j > h) & (j <= n-h) :
            elif ((j+1) > h) & ((j+1) <= n-h) :
                # 
                # selec_row = list(range(0, j-1-h)) + list(range(j+h,n))
                selec_row = list(range(0, j-1-h)) + list(range(j+h, n+1))
                weight_ij   = 1 / (n-2*h-1)
                # 
            # elif (j > n-h) & (j <= n):
            elif ((j+1) > n-h) & ((j+1) <= n) :
                # selec_row = list(range(0, j-1-h))
                selec_row = list(range(0, j-1-h))
                weight_ij   = 1 / (j-h-1)
                # print(j)
                # 
            weights += [weight_ij]
            # weight_sum += weight_ij
            # 
            # res_cv = OLS(y.iloc[selec_row], x.iloc[selec_row]).fit()
            # print(res_cv.summary())
            # print(f'j:{j}, sel_row: {selec_row[0]}, h: {h}, w_ij: {weight_ij}')
            # print('sel_row', selec_row[:10], selec_row[241:])
            # print(f'weight_sum: {weight_sum}')
                    
            # 3. train
            train = OLS(y.iloc[selec_row], Xs.iloc[selec_row]).fit()
            y_hat = train.predict(Xs.iloc[j])
            # print('y_hat:', y_hat[0], 'y:',y.iloc[j-1][0] )
            # 4. test and save the PE
            # print(f'y: {j}, {y.iloc[j][0]}, y_hat: {y_hat[0]}, weight: {weight_ij}')
            mse_i = (y.iloc[j][0] - y_hat[0])**2 * (weight_ij)
            # print(mse_i)
            mse += mse_i
            mses += [mse_i]
            stat_pe += [mse_i]
            
        rmse = mse ** 0.5
        print(f'mse: {mse}, rmse: {rmse}')
        print(f'h: {h}')
        print(f'E(PE): {np.average(stat_pe)}')
        print(f'MSE:{sum(mses)}, Weight Sum: {sum(weights)}')
        # print(df.iloc[selec_row])
        print(len(weights), weights)
        print(len(mses), mses)

        # return df.iloc[selec_row]
        


    def pe_weights(self, method={'berman', 'uniform'}) :
        print(method)
        if method == 'berman' :
            n = self.df.index.max() + 1
            h = np.ceil((n) / 6).astype(int)
            weights = []
            for j in range (1, n + 1) :
                if (1 <= j) & (j <=h) :
                    weight = 1 / (n - j - h)
                    row = []
                    for i in range (1, n + 1) :
                        if (1 <= i) & (i <= j + h) :
                            row += [0]
                        else :
                            row += [weight]

                elif (h < j) & (j <= n-h) :
                    weight = 1 / (n - 2*h - 1)
                    row = []
                    for i in range (1, n + 1) :
                        if (j-h <= i) & (i <= j + h) :
                            row += [0]
                        else :
                            row += [weight]

                elif (n-h < j) & (j <= n) :
                    weight = 1 / (j - h -1)
                    row = []
                    for i in range (1, n + 1) :
                        if (j-h <= i) & (i <= n) :
                            row += [0]
                        else :
                            row += [weight]
                weights += [row]
            return weights
        # 
        elif method == 'uniform':
            n = self.df.index.max() + 1
            h = np.ceil((n) / 6).astype(int)
            weights = []
            for j in range (1, n + 1) :
                if (1 <= j) & (j <=h) :
                    # weight = 1 / (n - j - h)
                    row = []
                    for i in range (1, n + 1) :
                        if (1 <= i) & (i <= j + h) :
                            row += [0]
                        else :
                            row += [1]
                    
                    row = np.divide(row , ([sum(row)] * len(row)))
                    row = list(row)

                elif (h < j) & (j <= n-h) :
                    # weight = 1 / (n - 2*h - 1)
                    row = []
                    for i in range (1, n + 1) :
                        if (j-h <= i) & (i <= j + h) :
                            row += [0]
                        else :
                            row += [1]
                    
                    row = np.divide(row , ([sum(row)] * len(row)))
                    row = list(row)

                elif (n-h < j) & (j <= n) :
                    # weight = 1 / (j - h -1)
                    row = []
                    for i in range (1, n + 1) :
                        if (j-h <= i) & (i <= n) :
                            row += [0]
                        else :
                            row += [1]
                    
                    row = np.divide(row , ([sum(row)] * len(row)))
                    row = list(row)

                weights += [row]
            return weights        

