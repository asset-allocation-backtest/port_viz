import pandas as pd
import pandas_datareader as pdr
import yahoo_finance_pynterface as yahoo
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# 야후 시고저종거래량
def get_yahoo_data(name):

    df = pdr.get_data_yahoo(name, start='1971-01-01')
    df.index.name = 'date'
    return df
# 야후 수정주가
def get_data_yahoo_close(product_name):
    if type(product_name) == str:
        df = get_yahoo_data(product_name)[['Close']]
        df.columns = [product_name]
    elif type(product_name) == list:
        df = pd.DataFrame()
        for name in product_name:
            df[name] = get_yahoo_data(name)['Close']
    else:
        return None
    return df
# 네이버 시고저종거래량
def get_data_naver(company_code):
    # count=3000에서 3000은 과거 3,000 영업일간의 데이터를 의미. 사용자가 조절 가능
    url = "https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&count=4000&requestType=0".format(
        company_code)
    get_result = requests.get(url)
    bs_obj = BeautifulSoup(get_result.content, "html.parser")

    # information
    inf = bs_obj.select('item')
    columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_inf = pd.DataFrame([], columns=columns, index=range(len(inf)))

    for i in range(len(inf)):
        df_inf.iloc[i] = str(inf[i]['data']).split('|')

    df_inf.index = pd.to_datetime(df_inf['date'])

    return df_inf.drop('date', axis=1).astype(float)
# 네이버 수정주가
def get_naver_close(company_code):
    df = get_data_naver(company_code)[['Close']]

    df.columns = [company_code]
    return df
