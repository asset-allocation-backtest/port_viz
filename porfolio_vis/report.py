# Using graph_objects
import pandas as pd
import numpy as np
from tqdm import tqdm
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row, widgetbox
from bokeh.models.widgets import *
from bokeh.models import Legend, Column
from bokeh import palettes
class Portfolio:
    def __init__(self, price, st_name=None):
        price.index = pd.to_datetime(price.index)
        price = price.dropna()

        self.st_name = st_name if st_name else price.columns
        self.price = price
        self.pct = price.pct_change().fillna(0)
        year_dates = self.price.groupby(self.price.index.year).count()
        if len(year_dates) <=2:
            self.year_dates = 250
        else:
            self.year_dates = year_dates.iloc[1:-1].mean().iloc[0]
        self.std = np.sqrt(np.log(self.pct+1).var()*252)
        self.cumsum_sr = self.pct.cumsum() +1
        self.compound_sr = (self.pct+1).cumprod()
        self.cumsum = self.cumsum_sr.iloc[-1]
        self.compound = self.compound_sr.iloc[-1]
        self.dd = self.get_dd(self.price)
        self.mdd = self.dd.min()
        self.cagr = self.get_cagr(self.compound_sr, self.year_dates)
        self.sharpe = self.get_sharpe(self.pct, self.year_dates)
        self.color_list = ['#ec008e', '#361b6f', '#0086d4', '#8c98a0'] + list(palettes.Category20_20)
    @staticmethod
    def get_dd(price):
        return price/price.expanding().max() -1
    @staticmethod
    def get_cagr(compound_sr, year_dates):
        return compound_sr.iloc[-1]**(year_dates/compound_sr.shape[0])-1
    @staticmethod
    def get_sharpe(pct, year_dates):
        return ((1 + pct.mean()) ** year_dates - 1)/(pct.std() * np.sqrt(year_dates))
    def report(self, simple=False, output_name='제목없음'):
        output_file(output_name + '.html')
        def to_source(df):
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
            return ColumnDataSource(df)
        static_data = pd.concat(
            [self.compound, self.cagr, self.sharpe, self.mdd],
            axis=1)
        static_data.columns = ['Compound_Return', 'CAGR', 'Sharpe Ratio', 'MDD']

        for col in static_data.columns:
            if col in ['Simple_Return', 'CAGR', 'MDD', 'Average Drawdown', 'Standard Deviation']:
                # if col in ['Cumulative_Return', 'CAGR', 'MDD', 'Average Drawdown']:
                static_data.loc[:, col] = static_data.loc[:, col].apply(
                    lambda x: str(np.around((x * 100), decimals=2)) + "%")
            else:
                static_data.loc[:, col] = static_data.loc[:, col].apply(lambda x: np.around(x, decimals=4))

        static_data.reset_index(inplace=True)
        static_data.rename(columns={'index': 'Portfolio'}, inplace=True)
        source = ColumnDataSource(static_data)
        columns = [TableColumn(field=col, title=col) for col in static_data.columns]
        data_table = DataTable(source=source, columns=columns, width=1500, height=200, index_position=None)
        if simple:
            # Plot 단리
            source = to_source(self.cumsum_sr)
            source_for_chart = to_source(self.cumsum_sr - 1)
            p1 = figure(x_axis_type='datetime',
                        title='Simple Return' + f'({self.cumsum_sr.index[0].strftime("%Y-%m-%d")} ~ {self.cumsum_sr.index[-1].strftime("%Y-%m-%d")})',
                        plot_width=1500, plot_height=400, toolbar_location="above")
        elif simple=='log':
            # Plot 로그
            source = to_source(self.compound_sr)
            source_for_chart = to_source(self.compound_sr)

            p1 = figure(x_axis_type='datetime', y_axis_type='log',
                        title='Log Compound Return' + f'({self.compound_sr.index[0].strftime("%Y-%m-%d")} ~ {self.compound_sr.index[-1].strftime("%Y-%m-%d")})',
                        plot_width=1500, plot_height=450, toolbar_location="above")

        else:
            # Plot 복리
            source = to_source(self.compound_sr)
            source_for_chart = to_source(self.compound_sr - 1)

            p1 = figure(x_axis_type='datetime',
                        title='Compound Return' + f'({self.compound_sr.index[0].strftime("%Y-%m-%d")} ~ {self.compound_sr.index[-1].strftime("%Y-%m-%d")})',
                        plot_width=1500, plot_height=450, toolbar_location="above")



        legend_list = []
        for i, col in enumerate(self.compound_sr.columns):
            p_line = p1.line(source=source_for_chart, x='date', y=col, color=self.color_list[i], line_width=2)
            legend_list.append((col, [p_line]))
        legend = Legend(items=legend_list, location='center')

        # Plot drawdown
        source_p3 = to_source(self.dd)
        p3 = figure(x_axis_type='datetime',
                    title='Drawdown',
                    plot_width=1500, plot_height=170, toolbar_location="above")
        legend_list = []
        for i, col in enumerate(self.dd.columns):
            # p3.line(source=source, x='date', y=col, color=color_list[i], legend=col + " Drawdown")
            baseline = np.zeros_like(self.dd[col].values)
            y = np.append(baseline, self.dd[col].values[::-1])
            x = self.dd.index.values
            x = np.append(x, x[::-1])

            p_line = p3.line(source=source_p3, x='date', y=col, color=self.color_list[i], line_width=2)
            legend_list.append((col, [p_line]))
        legend_3 = Legend(items=legend_list, location='center')

        p1.add_layout(legend, 'right')
        p1.legend.click_policy = "hide"

        p3.add_layout(legend_3, 'right')
        p3.legend.click_policy = "hide"

        from bokeh.models import NumeralTickFormatter
        p1.yaxis.formatter = NumeralTickFormatter(format='0 %')
        p3.yaxis.formatter = NumeralTickFormatter(format='0 %')

        show(column(p1, p3, Column(data_table)))
    def report_plotly(self, save_name='./제목없음', log_scale=True, show_auto=True):
        """
        인풋 데이터는 기준가 데이터
        """
        import plotly.offline
        import plotly.express as px
        import plotly.io as pio
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # price_ans = ans.copy()
        price_ans = self.price
        pio.renderers.default = "browser"
        port_result = Portfolio(price_ans)

        # 결과값
        st_name = port_result.st_name
        Cum_Com = port_result.compound.apply(lambda x:str(round(x*100, 2))+'%')
        CAGR = port_result.cagr.apply(lambda x:str(round(x*100, 2))+'%')
        Sharpe = port_result.sharpe.apply(lambda x:str(round(x*100, 2))+'%')
        Std = port_result.std.apply(lambda x:str(round(x*100, 2))+'%')
        MDD = port_result.mdd.apply(lambda x:str(round(x*100, 2))+'%')
        Avg_DD = port_result.dd.apply(lambda x:str(round(x.mean()*100, 2))+'%')
        Cum_Sim = port_result.cumsum.apply(lambda x:str(round(x*100, 2))+'%')

        # Plotly Table에 표시할 결과값 정리.
        Table_df = pd.DataFrame(
            {"Portfolio": st_name,
             "Compound Return": Cum_Com.tolist(),
             "CAGR": CAGR.tolist(),
             "Sharpe Ratio":Sharpe.tolist(),
             "Standard Deviation": Std.tolist(),
             "MDD": MDD.tolist(),
             "Average Drawdown": Avg_DD.tolist()
            })

        # Plotly Plot에 표시할 시계열
        Timeseries_Cum_Com = port_result.compound_sr
        Timeseries_DD = port_result.dd
        # 시계열 시작점/끝점
        TS_beg=Timeseries_Cum_Com.index.min().strftime('%Y-%m-%d')
        TS_end=Timeseries_Cum_Com.index.max().strftime('%Y-%m-%d')

        # 월별 수익률
        Ret_Montly = Timeseries_Cum_Com.assign(ym=Timeseries_Cum_Com.index.strftime('%Y-%m')).groupby('ym').apply(lambda x:((1+x.pct_change()).cumprod()-1).tail(1)).droplevel(1)

        # plot할 때의 색깔
        color_list = px.colors.qualitative.Safe



        # Subplot 객체 생성
        fig = make_subplots(rows=4, cols=1,
                            # shared_xaxes=True,
                            row_width=[0.425, 0.125, 0.2, 0.25][::-1], # row_width=[ #] in reverse order. don't ask why…!
                            subplot_titles=(f"Compound Return({TS_beg} - {TS_end})", "Drawdown", "Performance Analysis", "Monthly Return"),
                            vertical_spacing=0.06,
                            specs=[[{"type": "scatter"}],
                                   [{"type": "scatter"}],
                                   [{"type": "table"}],
                                   [{"type": "bar"}],
                                   ]
                            )

        # 전체 수익률과 DD 시계열, MOnt 표시
        for i in range(len(Timeseries_Cum_Com.columns)):
            # 수익률 그래프
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
            if log_scale: # 수익률그래프 log scale 선택
                fig.update_yaxes(type="log", row=1, col=1)

            # DD 그래프
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

            # montly return Bar
            fig.add_trace(
                go.Bar(
                    x=Ret_Montly.index.astype(str),
                    y=Ret_Montly[Ret_Montly.columns[i]],
                    legendgroup=f"{st_name[i]}",
                    name=f"{st_name[i]}",
                    marker_color=color_list[i],
                    showlegend=False
                ),
                row=4, col=1
            )
        fig.layout.annotations[0].update(x=0.105)

        # 테이블 표시
        fig.add_trace(
            go.Table(
                header=dict(
                    # values=["Date", "Number<br>Transactions", "Output<br>Volume (BTC)", "Market<br>Price", "Hash<br>Rate", "Cost per<br>trans-USD", "Mining<br>Revenue-USD", "Trasaction<br>fees-BTC"],
                    # 띄어쓰기를 <br>로 마크다운처럼 쓰는건가보넹
                    values=["Portfolio", "Compound Return", "CAGR", "Sharpe Ratio", "Standard Deviation", "MDD", "Average Drawdown"], # 표 column
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[Table_df[k].tolist() for k in Table_df.columns],        # 표 셀의 값
                    align = "left"
                )
            ),
            row=3, col=1
        )
        # 왼쪽 정렬 기능 찾아내는 거 실패!!
        # fig.layout.annotations[0].update(align='left')
        # fig.layout.annotations[1].update(align='left')
        # fig.layout.annotations[2].update(align='left')
        # 어쩔 수 없이 그냥 숫자로 맞춰놔야지 일단..
        fig.layout.annotations[0].update(x=0.105)
        fig.layout.annotations[1].update(x=0.025)
        fig.layout.annotations[2].update(x=0.05)
        fig.layout.annotations[3].update(x=0.035)
        fig.update_layout(
                        height=1000,
                        showlegend=True,
                        title_text="Portfolio Analysis",
                        )
        fig.update_layout(title_font_family="Arial",
                          title_font_color="Black",
                          title_font=dict(size=30)
                          )

        # fig.update_xaxes(
        #     rangeslider_visible=False,
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(count=1, label="YTD", step="year", stepmode="todate"),
        #             dict(count=1, label="1y", step="year", stepmode="backward"),
        #             dict(step="all")
        #         ])
        #     )
        # )

        # fig.layout.font.update(size=20)

        # fig.show()
        # 저장
        plotly.offline.plot(fig, filename=f'{save_name}.html', auto_open=show_auto)
    def onereport_plotly(self, save_name='./제목없음', log_scale=True, show_auto=True):
        """
        인풋 데이터는 기준가 데이터
        """
        import plotly.offline
        import plotly.express as px
        import plotly.io as pio
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # price_ans = ans.copy()
        price_ans = self.price
        pio.renderers.default = "browser"
        port_result = Portfolio(price_ans)

        # 결과값
        st_name = port_result.st_name
        Cum_Com = port_result.compound.apply(lambda x:str(round(x*100, 2))+'%')
        CAGR = port_result.cagr.apply(lambda x:str(round(x*100, 2))+'%')
        Sharpe = port_result.sharpe.apply(lambda x:str(round(x*100, 2))+'%')
        Std = port_result.std.apply(lambda x:str(round(x*100, 2))+'%')
        MDD = port_result.mdd.apply(lambda x:str(round(x*100, 2))+'%')
        Avg_DD = port_result.dd.apply(lambda x:str(round(x.mean()*100, 2))+'%')
        Cum_Sim = port_result.cumsum.apply(lambda x:str(round(x*100, 2))+'%')

        # Plotly Table에 표시할 결과값 정리.
        Table_df = pd.DataFrame(
            {"Portfolio": st_name,
             "Compound Return": Cum_Com.tolist(),
             "CAGR": CAGR.tolist(),
             "Sharpe Ratio":Sharpe.tolist(),
             "Standard Deviation": Std.tolist(),
             "MDD": MDD.tolist(),
             "Average Drawdown": Avg_DD.tolist()
            })

        # Plotly Plot에 표시할 시계열
        Timeseries_Cum_Com = port_result.compound_sr
        Timeseries_DD = port_result.dd
        # 시계열 시작점/끝점
        TS_beg=Timeseries_Cum_Com.index.min().strftime('%Y-%m-%d')
        TS_end=Timeseries_Cum_Com.index.max().strftime('%Y-%m-%d')

        # 월별 수익률
        Ret_Montly = Timeseries_Cum_Com.assign(ym=Timeseries_Cum_Com.index.strftime('%Y-%m')).groupby('ym').apply(lambda x:((1+x.pct_change()).cumprod()-1).tail(1)).droplevel(1)

        # plot할 때의 색깔
        color_list = px.colors.qualitative.Safe



        # Subplot 객체 생성
        fig = make_subplots(rows=4, cols=1,
                            # shared_xaxes=True,
                            row_width=[0.425, 0.125, 0.2, 0.25][::-1], # row_width=[ #] in reverse order. don't ask why…!

                            subplot_titles=(f"Compound Return({TS_beg} - {TS_end})", "Drawdown", "Performance Analysis", "Monthly Return"),
                            vertical_spacing=0.06,
                            specs=[[{"type": "scatter"}],
                                   [{"type": "scatter"}],
                                   [{"type": "table"}],
                                   [{"type": "bar"}],
                                   ]
                            )

        # 전체 수익률과 DD 시계열, MOnt 표시
        for i in range(len(Timeseries_Cum_Com.columns)):
            # 수익률 그래프
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
            if log_scale: # 수익률그래프 log scale 선택
                fig.update_yaxes(type="log", row=1, col=1)

            # DD 그래프
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

            # montly return Bar
            fig.add_trace(
                go.Bar(
                    x=Ret_Montly.index.astype(str),
                    y=Ret_Montly[Ret_Montly.columns[i]],
                    legendgroup=f"{st_name[i]}",
                    name=f"{st_name[i]}",
                    marker_color=color_list[i],
                    showlegend=False
                ),
                row=4, col=1
            )
        fig.layout.annotations[0].update(x=0.105)

        # 테이블 표시
        fig.add_trace(
            go.Table(
                header=dict(
                    # values=["Date", "Number<br>Transactions", "Output<br>Volume (BTC)", "Market<br>Price", "Hash<br>Rate", "Cost per<br>trans-USD", "Mining<br>Revenue-USD", "Trasaction<br>fees-BTC"],
                    # 띄어쓰기를 <br>로 마크다운처럼 쓰는건가보넹
                    values=["Portfolio", "Compound Return", "CAGR", "Sharpe Ratio", "Standard Deviation", "MDD", "Average Drawdown"], # 표 column
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[Table_df[k].tolist() for k in Table_df.columns],        # 표 셀의 값
                    align = "left"
                )
            ),
            row=3, col=1
        )
        # 왼쪽 정렬 기능 찾아내는 거 실패!!
        # fig.layout.annotations[0].update(align='left')
        # fig.layout.annotations[1].update(align='left')
        # fig.layout.annotations[2].update(align='left')
        # 어쩔 수 없이 그냥 숫자로 맞춰놔야지 일단..
        fig.layout.annotations[0].update(x=0.105)
        fig.layout.annotations[1].update(x=0.025)
        fig.layout.annotations[2].update(x=0.05)
        fig.layout.annotations[3].update(x=0.035)
        fig.update_layout(
                        height=1000,
                        showlegend=True,
                        title_text="Portfolio Analysis",
                        )
        fig.update_layout(title_font_family="Arial",
                          title_font_color="Black",
                          title_font=dict(size=30)
                          )

        # fig.update_xaxes(
        #     rangeslider_visible=False,
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(count=1, label="YTD", step="year", stepmode="todate"),
        #             dict(count=1, label="1y", step="year", stepmode="backward"),
        #             dict(step="all")
        #         ])
        #     )
        # )

        # fig.layout.font.update(size=20)

        # fig.show()
        # 저장
        plotly.offline.plot(fig, filename=f'{save_name}.html', auto_open=show_auto)
class OnePortfolio(Portfolio):
    def __init__(self, price, stacked_ratio, st_name=None):
        price.index = pd.to_datetime(price.index)
        price = price.dropna()

        self.st_name = st_name if st_name else price.columns
        self.price = price
        self.stacked_ratio = stacked_ratio.copy()
    def onereport_plotly(self, save_name='./제목없음', log_scale=True, show_auto=True):
        """
        인풋 데이터는 기준가 데이터
        """
        import plotly.offline
        import plotly.express as px
        import plotly.io as pio
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # price_ans = ans.copy()
        price_ans = self.price
        pio.renderers.default = "browser"
        port_result = Portfolio(price_ans)

        # 결과값
        st_name = port_result.st_name
        Cum_Com = port_result.compound.apply(lambda x:str(round(x*100, 2))+'%')
        CAGR = port_result.cagr.apply(lambda x:str(round(x*100, 2))+'%')
        Sharpe = port_result.sharpe.apply(lambda x:str(round(x*100, 2))+'%')
        Std = port_result.std.apply(lambda x:str(round(x*100, 2))+'%')
        MDD = port_result.mdd.apply(lambda x:str(round(x*100, 2))+'%')
        Avg_DD = port_result.dd.apply(lambda x:str(round(x.mean()*100, 2))+'%')
        Cum_Sim = port_result.cumsum.apply(lambda x:str(round(x*100, 2))+'%')

        # Plotly Table에 표시할 결과값 정리.
        Table_df = pd.DataFrame(
            {"Portfolio": st_name,
             "Compound Return": Cum_Com.tolist(),
             "CAGR": CAGR.tolist(),
             "Sharpe Ratio":Sharpe.tolist(),
             "Standard Deviation": Std.tolist(),
             "MDD": MDD.tolist(),
             "Average Drawdown": Avg_DD.tolist()
            })

        # Plotly Plot에 표시할 시계열
        Timeseries_Cum_Com = port_result.compound_sr
        Timeseries_DD = port_result.dd
        # 시계열 시작점/끝점
        TS_beg=Timeseries_Cum_Com.index.min().strftime('%Y-%m-%d')
        TS_end=Timeseries_Cum_Com.index.max().strftime('%Y-%m-%d')

        # 월별 수익률
        Ret_Montly = Timeseries_Cum_Com.assign(ym=Timeseries_Cum_Com.index.strftime('%Y-%m')).groupby('ym').apply(lambda x:((1+x.pct_change()).cumprod()-1).tail(1)).droplevel(1)

        # plot할 때의 색깔
        color_list = px.colors.qualitative.Dark24
        color_list2 = px.colors.qualitative.Pastel



        # Subplot 객체 생성
        fig = make_subplots(rows=5, cols=1,
                            # shared_xaxes=True,
                            row_width=[0.375, 0.075, 0.15, 0.2, 0.2][::-1], # row_width=[ #] in reverse order. don't ask why…!
                            subplot_titles=(f"Compound Return({TS_beg} - {TS_end})", "Drawdown", "Performance Analysis", "Monthly Return"),
                            vertical_spacing=0.06,
                            specs=[[{"type": "scatter"}],
                                   [{"type": "scatter"}],
                                   [{"type": "table"}],
                                   [{"type": "bar"}],
                                   [{"type": "scatter"}],
                                   ]
                            )

        # 전체 수익률과 DD 시계열, MOnt 표시
        for i in range(len(Timeseries_Cum_Com.columns)):
            # 수익률 그래프
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
            if log_scale: # 수익률그래프 log scale 선택
                fig.update_yaxes(type="log", row=1, col=1)

            # DD 그래프
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

            # montly return Bar
            fig.add_trace(
                go.Bar(
                    x=Ret_Montly.index.astype(str),
                    y=Ret_Montly[Ret_Montly.columns[i]],
                    legendgroup=f"{st_name[i]}",
                    name=f"{st_name[i]}",
                    marker_color=color_list[i],
                    showlegend=False
                ),
                row=4, col=1
            )
        fig.layout.annotations[0].update(x=0.105)

        # 테이블 표시
        fig.add_trace(
            go.Table(
                header=dict(
                    # values=["Date", "Number<br>Transactions", "Output<br>Volume (BTC)", "Market<br>Price", "Hash<br>Rate", "Cost per<br>trans-USD", "Mining<br>Revenue-USD", "Trasaction<br>fees-BTC"],
                    # 띄어쓰기를 <br>로 마크다운처럼 쓰는건가보넹
                    values=["Portfolio", "Compound Return", "CAGR", "Sharpe Ratio", "Standard Deviation", "MDD", "Average Drawdown"], # 표 column
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[Table_df[k].tolist() for k in Table_df.columns],        # 표 셀의 값
                    align = "left"
                )
            ),
            row=3, col=1
        )
        for col_i in range(len(self.stacked_ratio.columns)):
            fig.add_trace(
                            go.Scatter(
                                x=self.stacked_ratio.index, y=self.stacked_ratio.iloc[:,col_i],
                                mode='lines',
                                name=f'{self.stacked_ratio.columns[col_i]}',
                                line=dict(width=0.5, color=color_list2[col_i]),
                                stackgroup='one',
                                legendgroup='stacked line',
                                groupnorm='percent', # sets the normalization for the sum of the stackgroup
                                showlegend=True,

                                        ),
                            row=5, col=1
                         )
        # 왼쪽 정렬 기능 찾아내는 거 실패!!
        # fig.layout.annotations[0].update(align='left')
        # fig.layout.annotations[1].update(align='left')
        # fig.layout.annotations[2].update(align='left')
        # 어쩔 수 없이 그냥 숫자로 맞춰놔야지 일단..
        fig.layout.annotations[0].update(x=0.105)
        fig.layout.annotations[1].update(x=0.025)
        fig.layout.annotations[2].update(x=0.05)
        fig.layout.annotations[3].update(x=0.035)
        fig.update_layout(
                        height=1000,
                        showlegend=True,
                        title_text="Portfolio Analysis",
                        # legend=list(tracegroupgap=150),
                        )
        fig.update_layout(title_font_family="Arial",
                          title_font_color="Black",
                          title_font=dict(size=30)
                          )

        # fig.update_xaxes(
        #     rangeslider_visible=False,
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(count=1, label="YTD", step="year", stepmode="todate"),
        #             dict(count=1, label="1y", step="year", stepmode="backward"),
        #             dict(step="all")
        #         ])
        #     )
        # )

        # fig.layout.font.update(size=20)

        # fig.show()
        # 저장
        plotly.offline.plot(fig, filename=f'{save_name}.html', auto_open=show_auto)
class OnePortfolio_with_Scatter(OnePortfolio):
    def __init__(self, prices, stacked_ratio, prd_ret, prd_std, st_name=None):
        # prices, stacked_ratio, prd_ret, prd_std = DD.copy(), daily_ratio.copy(), period_ret.copy(), period_std.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.dropna()

        # self.st_name = st_name if st_name else price.columns
        self.price = prices.iloc[:, [0]]
        self.prices = prices
        self.stacked_ratio = stacked_ratio.copy()
        self.prd_ret = prd_ret
        self.prd_std = prd_std

        # self.stacked_ratio = stacked_ratio.copy()
    def onereport_plotly_scatter(self, save_name='./제목없음', log_scale=True, show_auto=True, Port_name=None):
        """
        인풋 데이터는 기준가 데이터
        """
        import plotly.offline
        import plotly.express as px
        import plotly.io as pio
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        # price_ans = ans.copy()
        price_ans = self.price
        pio.renderers.default = "browser"
        port_result = Portfolio(price_ans)

        # 결과값
        st_name = port_result.st_name
        Cum_Com = port_result.compound.apply(lambda x:str(round(x*100, 2))+'%')
        CAGR = port_result.cagr.apply(lambda x:str(round(x*100, 2))+'%')
        Sharpe = port_result.sharpe.apply(lambda x:str(round(x*100, 2))+'%')
        Std = port_result.std.apply(lambda x:str(round(x*100, 2))+'%')
        MDD = port_result.mdd.apply(lambda x:str(round(x*100, 2))+'%')
        Avg_DD = port_result.dd.apply(lambda x:str(round(x.mean()*100, 2))+'%')
        Cum_Sim = port_result.cumsum.apply(lambda x:str(round(x*100, 2))+'%')

        # Plotly Table에 표시할 결과값 정리.
        Table_df = pd.DataFrame(
            {"Portfolio": st_name,
             "Compound Return": Cum_Com.tolist(),
             "CAGR": CAGR.tolist(),
             "Sharpe Ratio":Sharpe.tolist(),
             "Standard Deviation": Std.tolist(),
             "MDD": MDD.tolist(),
             "Average Drawdown": Avg_DD.tolist()
             })

        # Plotly Plot에 표시할 시계열
        Timeseries_Cum_Com = port_result.compound_sr
        Timeseries_DD = port_result.dd
        # 시계열 시작점/끝점
        TS_beg=Timeseries_Cum_Com.index.min().strftime('%Y-%m-%d')
        TS_end=Timeseries_Cum_Com.index.max().strftime('%Y-%m-%d')

        # 월별 수익률
        Ret_Montly = Timeseries_Cum_Com.assign(ym=Timeseries_Cum_Com.index.strftime('%Y-%m')).groupby('ym').apply(lambda x:((1+x.pct_change()).cumprod()-1).tail(1)).droplevel(1)

        # plot할 때의 색깔
        color_list = px.colors.qualitative.Dark24
        color_list2 = px.colors.qualitative.Pastel



        # Subplot 객체 생성
        fig = make_subplots(rows=5, cols=2,
                            # shared_xaxes=True,
                            row_width=[0.375, 0.075, 0.15, 0.2, 0.2][::-1], # row_width=[ #] in reverse order. don't ask why…!
                            subplot_titles=(f"Compound Return({TS_beg} - {TS_end})", "Drawdown", "Performance Analysis", "Monthly Return", "Asset Area Graph", u"\u03C3 - \u03BC Scatter Plot"),
                            vertical_spacing=0.06,
                            specs=[
                                   [{"type": "scatter", "colspan": 2}, None],
                                   [{"type": "scatter", "colspan": 2}, None],
                                   [{"type": "table", "colspan": 2}, None],
                                   [{"type": "bar", "colspan": 2}, None],
                                   [{"type": "scatter"}, {"type": "scatter"}],
                                   ],
                            )

        # 전체 수익률과 DD 시계열, MOnt 표시
        for i in range(len(Timeseries_Cum_Com.columns)):
            # 수익률 그래프
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
            if log_scale: # 수익률그래프 log scale 선택
                fig.update_yaxes(type="log", row=1, col=1)

            # DD 그래프
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

            # montly return Bar
            fig.add_trace(
                go.Bar(
                    x=Ret_Montly.index.astype(str),
                    y=Ret_Montly[Ret_Montly.columns[i]],
                    legendgroup=f"{st_name[i]}",
                    name=f"{st_name[i]}",
                    marker_color=color_list[i],
                    showlegend=False
                ),
                row=4, col=1
            )
        fig.layout.annotations[0].update(x=0.105)

        # 테이블 표시
        fig.add_trace(
            go.Table(
                header=dict(
                    # values=["Date", "Number<br>Transactions", "Output<br>Volume (BTC)", "Market<br>Price", "Hash<br>Rate", "Cost per<br>trans-USD", "Mining<br>Revenue-USD", "Trasaction<br>fees-BTC"],
                    # 띄어쓰기를 <br>로 마크다운처럼 쓰는건가보넹
                    values=["Portfolio", "Compound Return", "CAGR", "Sharpe Ratio", "Standard Deviation", "MDD", "Average Drawdown"], # 표 column
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[Table_df[k].tolist() for k in Table_df.columns],        # 표 셀의 값
                    align = "left"
                )
            ),
            row=3, col=1
        )
        for col_i in range(len(self.stacked_ratio.columns)):
            fig.add_trace(
                go.Scatter(
                    x=self.stacked_ratio.index, y=self.stacked_ratio.iloc[:,col_i],
                    mode='lines',
                    name=f'{self.stacked_ratio.columns[col_i]}',
                    line=dict(width=0.5, color=color_list2[col_i]),
                    stackgroup='one',
                    legendgroup='stacked line',
                    groupnorm='percent', # sets the normalization for the sum of the stackgroup
                    showlegend=True,

                ),
                row=5, col=1
            )
        color_list = px.colors.qualitative.Safe

        for i in self.prd_ret.index:
            fig.add_trace(
                go.Scatter(
                    x=self.prd_std.iloc[:, 1:].loc[i].values, y=self.prd_ret.iloc[:, 1:].loc[i].values,
                    mode="markers",
                    legendgroup=f"random",
                    # name=f'random',
                    showlegend=False,
                    line_color=color_list[0],
                    marker=dict(size=2)
                ),
                row=5, col=2
            )
        for i in self.prd_ret.index:
            fig.add_trace(
                go.Scatter(
                    x=[self.prd_std.iloc[:, 0].loc[i]], y=[self.prd_ret.iloc[:, 0].loc[i]],
                    mode="markers",
                    legendgroup=f"Portfolio",
                    # name=f'Portfolio',
                    showlegend=False,
                    line_color=color_list[1],
                    marker=dict(size=3),
                ),
                row=5, col=2
            )
        # 왼쪽 정렬 기능 찾아내는 거 실패!!
        # fig.layout.annotations[0].update(align='left')
        # fig.layout.annotations[1].update(align='left')
        # fig.layout.annotations[2].update(align='left')
        # 어쩔 수 없이 그냥 숫자로 맞춰놔야지 일단..
        fig.layout.annotations[0].update(x=0.105)
        fig.layout.annotations[1].update(x=0.025)
        fig.layout.annotations[2].update(x=0.05)
        fig.layout.annotations[3].update(x=0.035)
        fig.layout.annotations[4].update(x=0.04)
        fig.layout.annotations[5].update(x=0.6)
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="Portfolio Analysis",
            # title_text=f"{Port_name} Analysis",
            # legend=list(tracegroupgap=150),
            yaxis_tickformat='%',
            yaxis2_tickformat='%',
            yaxis3_tickformat='%',
            yaxis5_tickformat='%',
        )
        fig.update_layout(title_font_family="Arial",
                          title_font_color="Black",
                          title_font=dict(size=30)
                          )


        # fig.update_xaxes(
        #     rangeslider_visible=False,
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(count=1, label="YTD", step="year", stepmode="todate"),
        #             dict(count=1, label="1y", step="year", stepmode="backward"),
        #             dict(step="all")
        #         ])
        #     )
        # )

        # fig.layout.font.update(size=20)

        # fig.show()
        # 저장
        plotly.offline.plot(fig, filename=f'{save_name}.html', auto_open=show_auto)




