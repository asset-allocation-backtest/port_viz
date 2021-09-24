import pandas as pd
from random import sample
import numpy as np
import porfolio_vis as pv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm


DD = pd.DataFrame()
# cls = pv.strategy(data_list=['SPDW','VWO', 'SCHX', 'VB','SCHR', 'IGOV', 'BIL', 'SCHH', 'GSP'], country='us', window_hold='M', rebalancing_date=-1)
cls = pv.strategy(data_list=['SPY','IEF'], country='us', window_hold='M', rebalancing_date=-1)
gr = cls.get_group(window_fit='M')
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=[0.6, 0.4])
df.index.name = 'date'
ans, daily_ratio = cls.get_return_and_ratio(df, cost=0.01)
daily_ratio = daily_ratio.div(daily_ratio.sum(1), axis=0).cumsum(1)
DD['60:40 Portfolio(M)'] = ans[ans.columns[0]]
pv.report.OnePortfolio(DD, daily_ratio).onereport_plotly(save_name='./output/Classical_M', show_auto=False)

cls = pv.strategy(data_list=['SPY','IEF'], country='us', window_hold='Q', rebalancing_date=-1)
gr = cls.get_group(window_fit='Q')
# df = gr.apply(cls.func, ratio=[0.08, 0.08, 0.13, 0.12, 0.17, 0.08, 0.09, 0.08, 0.17])
df = gr.apply(cls.func, ratio=[0.6, 0.4])
df.index.name = 'date'
ans, daily_ratio = cls.get_return_and_ratio(df, cost=0.01)
DD['60:40 Portfolio(Q)'] = ans[ans.columns[0]]
pv.report.OnePortfolio(DD, daily_ratio).onereport_plotly(save_name='./output/Classical_Q', show_auto=False)


DD['S&P500'] = pv.get_data.get_data_yahoo_close('^GSPC')
report_cls = pv.report.Portfolio(DD).report_plotly(save_name='./output/Classical')



DD[['SPY']] = pv.get_data.get_data_yahoo_close('SPY')
DD[['IEI']] = pv.get_data.get_data_yahoo_close('IEI')
DD = (1+DD.dropna().pct_change()).cumprod()
DD.iloc[0]=1


mu = DD.pct_change()
# 월별 수익률
mu_m = mu.resample('BM').apply(lambda x:(1+x[['60:40 Portfolio(M)','60:40 Portfolio(Q)']]).cumprod().tail(1).sub(1)).droplevel(1)
# 월별 수익률 표준편차
std_m = mu.resample('BM').apply(lambda x:x[['60:40 Portfolio(M)','60:40 Portfolio(Q)']].std())

mu_m = mu.groupby(pd.Grouper(freq='BM')).apply(lambda x:(1+x[['SPY', 'IEI']]).cumprod().tail(1).sub(1).dot([0.6, 0.4])).droplevel(1)
std_m = mu.groupby(pd.Grouper(freq='BM')).apply(lambda x:(1+x[['SPY', 'IEI']]).cumprod().sub(1).dot([0.6, 0.4]).std())

rnds = np.array(sample(sorted(np.linspace(0, 1, 5000)), 2500))
rnd_list = np.array([[i,j] for i, j in zip(rnds, 1-rnds)]).reshape(2, -1)
mu_other = mu.groupby(pd.Grouper(freq='BM')).apply(lambda x:(1+x[['SPY', 'IEI']]).cumprod().tail(1).sub(1).dot(rnd_list)).droplevel(1)
std_other = mu.groupby(pd.Grouper(freq='BM')).apply(lambda x:(1+x[['SPY', 'IEI']]).cumprod().sub(1).dot(rnd_list).std())







color_list = px.colors.qualitative.Safe
fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter"}]])
for i in tqdm(std_other.index):
    fig.add_trace(
        go.Scatter(
            x=std_other.loc[i].values, y=mu_other.loc[i].values,
            mode="markers",
            legendgroup=f"random",
            name=f'random',
            line_color=color_list[0],
        ),
        row=1, col=1
    )
for i in tqdm(std_other.index):
    fig.add_trace(
        go.Scatter(
            x=[std_m.loc[i]], y=[mu_m.loc[i]],
            mode="markers",
            legendgroup=f"Portfolio",
            name=f'Portfolio',
            line_color=color_list[1],
        ),
        row=1, col=1
    )
fig.update_layout(showlegend=False)
fig.show()



std_m = mu.assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:x.std())

from random import uniform
from tqdm import tqdm
for iter in tqdm(range(100)):
    mu_m[f'Port{iter}_mu'] = mu.assign(ym=lambda x:x.index.strftime('%Y%m')).groupby('ym').apply(lambda x:(1+x).cumprod().tail(1)).assign(Port=lambda x:(x['SPY']*rnd)+x['IEI']*(1-rnd)).droplevel(1)['Port']
