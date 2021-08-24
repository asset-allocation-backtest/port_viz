import random

import pandas as pd
import porfolio_vis as pv


DD = pd.DataFrame()
cls = pv.strategy(data_list=['SPY', 'IEF', 'TLT', 'IAU', 'DBC'], country='us', window_hold='M', rebalancing_date=-1)
gr = cls.get_group(window_fit='M')
df = gr.apply(cls.func, ratio=[0.3, 0.4, 0.15, 0.07, 0.08])
df.index.name = 'date'
ans, daily_ratio = cls.get_return_and_ratio(df, cost=0.01)
DD['All Seasons Portfolio(M)'] = ans[ans.columns[0]]

cls = pv.strategy(data_list=['SPY', 'IEF', 'TLT', 'IAU', 'DBC'], country='us', window_hold='Q', rebalancing_date=-1)
gr = cls.get_group(window_fit='Q')
df = gr.apply(cls.func, ratio=[0.3, 0.4, 0.15, 0.07, 0.08])
df.index.name = 'date'
ans, daily_ratio = cls.get_return_and_ratio(df, cost=0.01)
DD['All Seasons Portfolio(Q)'] = ans[ans.columns[0]]


DD['S&P500'] =  pv.get_data.get_data_yahoo_close('^GSPC')


report_cls = pv.report.Portfolio(DD).report_plotly()