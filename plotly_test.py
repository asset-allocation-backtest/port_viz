import pandas as pd
import plotly.offline
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from porfolio_vis import *
data = pd.read_csv('./SPYIEF_bench.csv', parse_dates=['date'], index_col='date')
data.index = pd.to_datetime(data.index.strftime('%Y-%m-%d'))

# nx.report(data, name='./nd')





cls = strategy(data_list=['SPY','ACWI','IEI','IEF'],
               country='us',
               window_hold='M',
               rebalancing_date=-1
               )

gr = cls.get_group(window_fit='M')
df = gr.apply(cls.func, ratio=[0.1,0.4,0.2,0.3])
df.index.name = 'date'
ans = cls.get_return(df, cost=0.01)


port_result = report.Portfolio(ans)




# 결과값
st_name = port_result.st_name
Cum_Com = port_result.compound.apply(lambda x:str(round(x*100, 2))+'%')
CAGR = port_result.cagr.apply(lambda x:str(round(x*100, 2))+'%')
Sharpe = port_result.sharpe.apply(lambda x:str(round(x*100, 2))+'%')
Std = port_result.std.apply(lambda x:str(round(x*100, 2))+'%')
MDD = port_result.mdd.apply(lambda x:str(round(x*100, 2))+'%')
Avg_DD = port_result.dd.apply(lambda x:str(round(x.mean()*100, 2))+'%')
Cum_Sim = port_result.cumsum.apply(lambda x:str(round(x*100, 2))+'%')

Table_df = pd.DataFrame({
                          "Portfolio": st_name,
                          "Compound Return": Cum_Com.tolist(),
                          "CAGR": CAGR.tolist(),
                          "Sharpe Ratio":Sharpe.tolist(),
                          "Standard Deviation": Std.tolist(),
                          "MDD": MDD.tolist(),
                          "Average Drawdown": Avg_DD.tolist()
                          })

# 시계열
Timeseries_Cum_Com = port_result.compound_sr
Timeseries_DD = port_result.dd

# 최종 리밸런싱 정보
# last_rebal = df.iloc[-1]



# report_cls.compound_sr['2'] = data['strategy']
# report_cls.compound_sr = report_cls.compound_sr.dropna()
pio.renderers.default = "browser"
color_list = px.colors.qualitative.Dark24
fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    row_width=[0.425, 0.15, 0.425],
                    vertical_spacing=0.03,
                    specs=[
                            [{"type": "scatter"}],
                            [{"type": "scatter"}],
                            [{"type": "table"}],
                           ]
                    )

for i in range(len(Timeseries_Cum_Com.columns)):
    fig.add_trace(
        go.Scatter(
            x=Timeseries_Cum_Com.index.astype(str),
            y=Timeseries_Cum_Com[Timeseries_Cum_Com.columns[i]],
            mode="lines",
            name=f"{st_name[i]}",
            line_color=color_list[i],
            legendgroup=f"{st_name[i]}",

        ),
        row=1, col=1
    )
    if log_scale:
        fig.update_yaxes(type="log", row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=Timeseries_DD.index.astype(str),
            y=Timeseries_DD[Timeseries_DD.columns[i]],
            mode="lines",
            name=f"{st_name[i]}",
            line_color=color_list[i],
            legendgroup=f"{st_name[i]}",
            showlegend=False
        ),
        row=2, col=1
    )


fig.add_trace(
    go.Table(
        header=dict(
                    # values=["Date", "Number<br>Transactions", "Output<br>Volume (BTC)", "Market<br>Price", "Hash<br>Rate", "Cost per<br>trans-USD", "Mining<br>Revenue-USD", "Trasaction<br>fees-BTC"],
                    # 띄어쓰기를 <br>로 마크다운처럼 쓰는건가보넹
                    values=["Portfolio", "Compound Return", "CAGR", "Sharpe Ratio", "Standard Deviation", "MDD", "Average Drawdown"],
                    font=dict(size=10),
                    align="left"
                   ),
        cells=dict(
                   values=[Table_df[k].tolist() for k in Table_df.columns],
                   align = "left"
                  )
    ),
    row=3, col=1
)
# fig.update_layout(
#     height=1000,
#     showlegend=True,
# )

# fig.show()
plotly.offline.plot(fig, filename = './test.html', auto_open=True)

