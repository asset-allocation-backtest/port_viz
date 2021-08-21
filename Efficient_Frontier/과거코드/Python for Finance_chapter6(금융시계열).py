# 일단 필요한 모듈 ~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pandas_datareader import data as web


path = r'C:\Users\GD Park\iCloudDrive\gdpre_PycharmProjects\Python for Finance'
name = 'chapter6. 금융시계열'
folder = path + '/' + name + '/'
# os.mkdir(folder)
os.chdir(folder)
# 자 그럼 이제 들어가보즈아!
########################################################################################################################
start = '2005-01-01'
end = '2019-09-09'
df = web.DataReader('^KS11', data_source='yahoo', start=start, end=end)         # 이번에도 KOSPI지수를 가져오겠음.
df['Close'].plot(figsize=(10, 6))                       # reset_index하기 전에 행index가 날짜니깐, 바로 그려버리니깐, plot함수가 x축을 알아서 잘 잡네
plt.savefig('1.png')
########################################################################################################################
# 종가들 그리는 거 말고, 수익률을 그리는 거 해볼까. 수익률 column을 만들어 줘야지~
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))       # shift는 데이터를 한 칸 앞으로 싹 끌어오는 것이여. 그래서 수익률은 이렇게 만들면 되는데
########################################################################################################################
# shift가 어떻게 되는 건지 쉬프트시킨 것만 만들어서 보여줄께
df2 = df[['Close', 'High']]
df2['shift1'] = df['High'].shift(1)
df2['shift2'] = df['High'].shift(2)
df2['shift5'] = df['High'].shift(5)
# 그러면 로그 수익률 쉬프트 하나 시킨거랑 나누는 거 이해갔을 것이고
########################################################################################################################
# 판다스에서 그냥 바로 plot을 제공하기도 해
df[['Close', 'Return']].plot(subplots=True, style=['g', 'r'], figsize=(10, 6)) # 그냥 DataFrame에서 복수의 열을 잡고 매써드 plot 먹여버리면, 그려줌
plt.savefig('2.png')
df[['Close', 'Return']].plot(subplots=True, style=['g--', 'r-'], figsize=(10, 6)) # 다만, marker는 따로 인수를 지정하는거 없고, color랑 같이 입력하면 됨. 'k--'이런식으로
plt.savefig('3.png')
df[['Close', 'Return']].plot(style='r-', figsize=(10, 6))                # subplot은 선택하기 나름
plt.savefig('4.png')
# https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
########################################################################################################################
# 근데 너무 기간이 기니깐 쫌 그렇다.... 한 5년으로 바꿀랭!
start = '2014-09-09'
end = '2019-09-09'
df = web.DataReader('^KS11', data_source='yahoo', start=start, end=end)
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))       # shift는 데이터를 한 칸 앞으로 싹 끌어오는 것이여. 그래서 수익률은 이렇게 만들면 되는데
########################################################################################################################
# 이평선그려넣는것도 해보자. 그럼 일단 이동평균 column을 만드는게 필요함
# df['42d'] = pd.rolling_mean(df['Close'], window=42)         # 42일 이동평균
# df['252d'] = pd.rolling_mean(df['Close'], window=252)       # 252일 이동평균
# 책에서는 저렇게 쓰여져 있는데, 없어졌나 보네... 다른 방법으로 할 수 있음
df['42d'] = df['Close'].rolling(window=42).mean()
df['252d'] = df['Close'].rolling(window=252).mean()

df[['Close', '42d', '252d']].plot(figsize=(10, 6))
plt.savefig('5.png')                                # 오.. 싱기방기 동방싱기...
########################################################################################################################
# 이동표준편차... 이거 루트를 해야할테니, 루트계산할 수 있게 math를 좀 가져오자
import math
df['Mov_Vol'] = df['Return'].rolling(window=252).std() * math.sqrt(252)
df[['Close', 'Return', 'Mov_Vol']].plot(subplots=True, style=['r', 'g', 'b'], figsize=(10, 6))
plt.savefig('6.png')
########################################################################################################################
# 이번에는 회귀를 한 번 해보겠다고 하는데, 책은 독일사람이라 유로50이랑 그거의 변동성지수를 가져와서 했는데
# 나는 코스피랑 삼성전자로 해야겠다 ㅎㅎㅎ! (상관관계가 있다고 하는 걸 들은 적이 있어서, 그거 회귀분석해봐야지~)
start = '2014-09-09'
end = '2019-09-09'
kospi = web.DataReader('^KS11', data_source='yahoo', start=start, end=end)
samsung = web.DataReader('005930.KS', data_source='yahoo', start=start, end=end)
kospi['return'] = np.log(kospi['Close'] / kospi['Close'].shift(1))
samsung['return'] = np.log(samsung['Close'] / samsung['Close'].shift(1))
data = pd.DataFrame([])
data['kospi'] = kospi['return']
data['samsung'] = samsung['return']
data.plot(subplots=True, grid=True, style=['r--', 'g--'], figsize=(10, 6))      # 이렇게 보면 아무것도 안보인다
plt.savefig('7.png')
########################################################################################################################
# 이 두 값을 무슨 상관관계가 있는지 해보자!!!
# 그 상관관계를 회귀분석하겠다는 것...!
data.info() # 근데 보면, non-null인게 좀 갯수가 안맞네
# 그래서 fillna()를 좀 해줄껀데, fillna(0) 이렇게 하면, 그냥 NaN값은 0으로 바꾸라는 건데, 0을로 바꾸는 건 좀 오바고
# NaN값이 있으면 그냥 '이전의 데이터로 채워줘'하는 게 method=ffill(forward fill)
# NaN값이 있으면 그냥 '이후의 데이터로 채워줘'하는 게 method=bfill(backward fill)
data = data.fillna(method='ffill')
data.dropna(inplace=True)   # 맨 앞에 NaN값은 '이전의 데이터로 채워라'를 실행하지 못해서 생긴 NaN값 제거!
xdat = data['samsung'].values
ydat = data['kospi'].values
reg = np.polyfit(x=xdat, y=ydat, deg=1) # 그러면 베타0, 베타1 만들어졌다! 이거 이제 시각화하자!
plt.plot(xdat, ydat, 'g.')
ax = plt.axis()                             # x축의 맨 끝과 끝을 뽑아내기 위해서 이런짓을 좀 하고
x = np.linspace(ax[0]-0.01, ax[1]+0.01)     # 거기에서 약간 차이를 줘서 이쁘게 그리기 위해서 하는 짓이고
plt.plot(x, reg[1]+reg[0]*x, 'b', lw=2)     # 얍! 그려랏!
plt.xlabel('samsung_electronics return')
plt.ylabel('KOSPI return')
plt.savefig('8.png')
########################################################################################################################
# 마지막으로 이동상관계수도 살펴봐보자!
data['samsung'].rolling(window=252).corr(data['kospi']).plot(grid=True, style='b')
plt.ylabel('rolling_corr(kospi&samsung)')
plt.savefig('9.png')                        # 엥... 상관계수가 계속 올라가네? 무슨일이 있던거지...? 15년 정도로 함 봐볼까...?
########################################################################################################################
start = '2000-09-09'
end = '2019-09-09'
kospi = web.DataReader('^KS11', data_source='yahoo', start=start, end=end)
samsung = web.DataReader('005930.KS', data_source='yahoo', start=start, end=end)
kospi['return'] = np.log(kospi['Close'] / kospi['Close'].shift(1))
samsung['return'] = np.log(samsung['Close'] / samsung['Close'].shift(1))
data = pd.DataFrame([])
data['kospi'] = kospi['return']
data['samsung'] = samsung['return']
data['samsung'].rolling(window=252).corr(data['kospi']).plot(grid=True, style='b')
plt.ylabel('rolling_corr(kospi&samsung)')
# 오르락 내리락 해왔었구나아아아아아아
plt.savefig('10.png')
########################################################################################################################