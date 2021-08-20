import random

import pandas as pd
import porfolio_vis as pv


DD = pd.DataFrame()
# cls = pv.strategy(data_list=['SPDW','VWO', 'SCHX', 'VB','SCHR', 'IGOV', 'BIL', 'SCHH', 'GSP'], country='us', window_hold='M', rebalancing_date=-1)
cls = pv.strategy(data_list=['SPY','IEF'], country='us', window_hold='M', rebalancing_date=-1)
gr = cls.get_group(window_fit='M')
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=[0.6, 0.4])
df.index.name = 'date'
ans = cls.get_return(df, cost=0.01)
DD['60:40 Portfolio(M)'] = ans[ans.columns[0]]

cls = pv.strategy(data_list=['SPY','IEF'], country='us', window_hold='Q', rebalancing_date=-1)
gr = cls.get_group(window_fit='Q')
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=[0.6, 0.4])
df.index.name = 'date'
ans = cls.get_return(df, cost=0.01)
DD['60:40 Portfolio(Q)'] = ans[ans.columns[0]]

cls = pv.strategy(data_list=['SPY', 'IEF', 'TLT', 'IAU', 'DBC'], country='us', window_hold='M', rebalancing_date=-1)
gr = cls.get_group(window_fit='M')
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=[0.3, 0.4, 0.15, 0.07, 0.08])
df.index.name = 'date'
ans = cls.get_return(df, cost=0.01)
DD['All Seasons Portfolio(M)'] = ans[ans.columns[0]]

cls = pv.strategy(data_list=['SPY', 'IEF', 'TLT', 'IAU', 'DBC'], country='us', window_hold='Q', rebalancing_date=-1)
gr = cls.get_group(window_fit='Q')
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=[0.3, 0.4, 0.15, 0.07, 0.08])
df.index.name = 'date'
ans = cls.get_return(df, cost=0.01)
DD['All Seasons Portfolio(Q)'] = ans[ans.columns[0]]


# DD[['SPY']] = pv.get_data.get_data_yahoo_close('SPY')
# DD[['IEI']] = pv.get_data.get_data_yahoo_close('IEI')



DD['S&P500'] =  pv.get_data.get_data_yahoo_close('^GSPC')





report_cls = pv.report.Portfolio(DD).report_plotly()

asasddasds




DD = (1+DD.dropna().pct_change()).cumprod()
DD.iloc[0]=1


mu = DD.pct_change()

mu_m = mu.assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:(1+x).cumprod().tail(1)).droplevel(1).drop(['SPY', 'IEI'], axis=1)
std_m = mu.assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:(1+x).cumprod())\
          .assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:x.pct_change())\
          .assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:x.std())


from random import uniform
from tqdm import tqdm
for iter in tqdm(range(100)):
    rnd = uniform(0, 1)
    mu_m[f'Port{iter}'] = mu.assign(ym=lambda x:x.index.strftime('%Y%m'))\
                            .groupby('ym')\
                            .apply(lambda x:(1+x).cumprod().tail(1))\
                            .assign(Port=lambda x:(x['SPY']*rnd)+x['IEI']*(1-rnd))\
                            .droplevel(1)['Port']

