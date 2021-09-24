import pandas as pd
from random import sample, uniform
import numpy as np
import porfolio_vis as pv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm
def gen_random_numbers_normed(num):
    ns = np.array([uniform(0, 1) for iters in range(num)])
    return ns/ns.sum()
def gen_random_Portfolio(Port_price_df, num_=50, window_fit='M', cost=0.01, num_univ=len(['SPY', 'IEF', 'TLT', 'IAU', 'DBC'])):
    for itr in tqdm(range(num_)):
        Port_price_df[f'rnd_{itr}'] = cls.get_return_and_ratio(gr.apply(cls.func, ratio=gen_random_numbers_normed(num_univ)), cost=cost)[0]
        ggg = cls.get_data_group(Port_price_df, window_fit=window_fit)
        period_ret = ggg.apply(lambda x:x.pct_change().add(1).cumprod().tail(1).sub(1)).droplevel([1,2])
        period_std = ggg.apply(lambda x:x.pct_change().std())
    return Port_price_df, period_ret, period_std

Port_df = pd.DataFrame()
UNIVERSE = ['SPY', 'IEF']
ratio = [0.6, 0.4]


holding_period = 'M'
Port_df_temp = pd.DataFrame()
# cls = pv.strategy(data_list=['SPDW','VWO', 'SCHX', 'VB','SCHR', 'IGOV', 'BIL', 'SCHH', 'GSP'], country='us', window_hold='M', rebalancing_date=-1)
cls = pv.strategy(data_list=UNIVERSE, country='us', window_hold=holding_period, rebalancing_date=-1)
gr = cls.get_group(window_fit=holding_period)
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=ratio)
df.index.name = 'date'
ans, daily_ratio = cls.get_return_and_ratio(df, cost=0.01)

Port_df_temp['Classical Portfolio(M)'] = ans[ans.columns[0]]
Port_df['Classical Portfolio(M)'] = ans[ans.columns[0]]

Port_price_df, period_ret, period_std = gen_random_Portfolio(Port_df_temp, window_fit=holding_period, num_univ=len(ratio))
pv.report.OnePortfolio_with_Scatter(Port_price_df, daily_ratio, period_ret, period_std).onereport_plotly_scatter(save_name='./output/Classical_M_ws', show_auto=True)
# self = pv.report.OnePortfolio_with_Scatter(Port_price_df, daily_ratio, period_ret, period_std)
# self.onereport_plotly_scatter(save_name='./output/Classical_M_ws', show_auto=True)



holding_period = 'Q'
Port_df_temp = pd.DataFrame()
# cls = pv.strategy(data_list=['SPDW','VWO', 'SCHX', 'VB','SCHR', 'IGOV', 'BIL', 'SCHH', 'GSP'], country='us', window_hold='M', rebalancing_date=-1)
cls = pv.strategy(data_list=UNIVERSE, country='us', window_hold=holding_period, rebalancing_date=-1)
gr = cls.get_group(window_fit=holding_period)
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=ratio)
df.index.name = 'date'
ans, daily_ratio = cls.get_return_and_ratio(df, cost=0.01)

Port_df_temp['Classical Portfolio(Q)'] = ans[ans.columns[0]]
Port_df['Classical Portfolio(Q)'] = ans[ans.columns[0]]

Port_price_df, period_ret, period_std = gen_random_Portfolio(Port_df_temp, window_fit=holding_period, num_univ=len(ratio))
pv.report.OnePortfolio_with_Scatter(Port_price_df, daily_ratio, period_ret, period_std).onereport_plotly_scatter(save_name='./output/Classical_Q_ws', show_auto=True)


Port_df['S&P500'] = pv.get_data.get_data_yahoo_close('^GSPC')
report_cls = pv.report.Portfolio(Port_df).report_plotly(save_name='./output/Classical')