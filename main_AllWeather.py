import random

import pandas as pd
import porfolio_vis as pv


DD = pd.DataFrame()
# cls = pv.strategy(data_list=['SPDW','VWO', 'SCHX', 'VB','SCHR', 'IGOV', 'BIL', 'SCHH', 'GSP'], country='us', window_hold='M', rebalancing_date=-1)
cls = pv.strategy(data_list=['SPY', 'IEF', 'TLT', 'IAU', 'DBC'], country='us', window_hold='M', rebalancing_date=-1)
gr = cls.get_group(window_fit='M')
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=[0.3, 0.4, 0.15, 0.07, 0.08])
df.index.name = 'date'
ans, daily_ratio = cls.get_return_and_ratio(df, cost=0.01)
daily_ratio = daily_ratio.div(daily_ratio.sum(1), axis=0).cumsum(1)
DD['All Seasons Portfolio(M)'] = ans[ans.columns[0]]
pv.report.OnePortfolio(DD, daily_ratio).onereport_plotly(save_name='./output/AllSeasons_M', show_auto=False)

cls = pv.strategy(data_list=['SPY', 'IEF', 'TLT', 'IAU', 'DBC'], country='us', window_hold='Q', rebalancing_date=-1)
gr = cls.get_group(window_fit='Q')
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=[0.3, 0.4, 0.15, 0.07, 0.08])
df.index.name = 'date'
ans, daily_ratio = cls.get_return_and_ratio(df, cost=0.01)
DD['All Seasons Portfolio(Q)'] = ans[ans.columns[0]]
pv.report.OnePortfolio(DD, daily_ratio).onereport_plotly(save_name='./output/AllSeasons_Q', show_auto=False)


DD['S&P500'] =  pv.get_data.get_data_yahoo_close('^GSPC')


report_cls = pv.report.Portfolio(DD).report_plotly(save_name='./output/AllSeasons')



DD[['SPY']] = pv.get_data.get_data_yahoo_close('SPY')
DD[['IEF']] = pv.get_data.get_data_yahoo_close('IEF')
DD[['TLT']] = pv.get_data.get_data_yahoo_close('TLT')
DD[['IAU']] = pv.get_data.get_data_yahoo_close('IAU')
DD[['DBC']] = pv.get_data.get_data_yahoo_close('DBC')
DD = (1+DD.dropna().pct_change()).cumprod()
DD.iloc[0]=1


mu = DD.pct_change()
# 월별 수익률
mu_m = mu.assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:(1+x).cumprod().tail(1)).droplevel(1)

# 월별 수익률 표준편차
std_m = mu.assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:x.std())

from random import uniform
import numpy as np
from tqdm import tqdm

rnds = np.array([uniform(0, 1) for x in range(6)])
normed_rnds = rnds/rnds.sum()
for iter in tqdm(range(100)):
    rnd = uniform(0, 1)
    mu_m[f'Port{iter}_mu'] = mu.assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:(1+x).cumprod().tail(1)).assign(Port=lambda x:(x['SPY']*rnd)+x['IEI']*(1-rnd)).droplevel(1)['Port']

