# 일단 필요한 모듈 ~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 자 그럼 이제 들어가보즈아!
########################################################################################################################
# symbols = ['005930', '005380', '030200']
symbols = {'005930':'samsung', '005380':'hyundai', '030200':'KT'}
data = pd.DataFrame()
import porfolio_vis as pv
for sym in tqdm(symbols.keys()):
    data[symbols[sym]] = pv.get_data.get_naver_close(sym)

data = pd.DataFrame()
for sym in tqdm(symbols.keys()):
    data[symbols[sym]] = pv.get_data.get_naver_close(sym)
    # data[sym] = web.DataReader(sym, data_source='yahoo', start='2000-09-09', end='2019-09-09')['Close']


(data / data.iloc[0] * 100).plot(figsize=(10, 6))
# plt.savefig('1.png')                                # 간단하게 첫날 가격 대비 수익률 그래프. 근데, 현차랑 KT에 종가 빵꾸가 있네;;;....
########################################################################################################################
rets = np.log(data / data.shift(1))         # 이건 날마다 수익률 계산한거고
rets.mean() * 252           # 날마다 수익률 연율화
rets.std() * 252            # 표준편차듀~

weights = np.random.random(len(symbols))
weights = weights / np.sum(weights)             # weights = weights / np.sum(weights) 이런계산을 한 것
weights     # 즉, 랜덤하게 100%가 되도록 난수를 뽑은거

mu = np.sum(rets.mean() * weights) * 252                        # 포트폴리오 (연율화) 수익률의 합
square_sigma = np.dot(weights.T, np.dot(rets.cov(), weights))   # 포트폴리오 분산
sigma = np.sqrt(square_sigma)                                   # 이렇게 한 바뀌 돌리면, 랜덤하게 정한 weight에 대해서 -> 그 weight로 투자했을 때 포트폴리오 수익률과 표준편차를 계산하는거고
########################################################################################################################
# 위의 작업을 엄청나게 많이하면, ......
prets = []
pvols = []
for p in range(2500):
    weights = np.random.random(len(symbols))
    weights = weights / np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)

plt.figure(figsize=(10, 6))
plt.scatter(pvols, prets, c=prets/pvols, marker='.')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.title('Samsung-Hyundai-KT Portfolio')
plt.colorbar(label='sharpe ratio')                          # 그래도 이중엔 가장 효율적인 포트폴리오 weight를 가져가는게 있지 않겠냐~
# plt.savefig('2.png')                                        # 이런 논리ㅋㅋㅋㅋㅋㅋ
올퓨
########################################################################################################################
# 데이터 빵꾸 없는걸로 바꾸고 시작하자!!!!
stocks = ['005930.KS', '005380.KS', '034730.KS', '035420.KS', '010950.KS']
data = pd.DataFrame()
for i in stocks:
    data[i] = web.DataReader(i, data_source='yahoo', start='2010-01-01', end='2018-06-30')['Close']
data.columns = ['samsung', 'hyundai', 'SK', 'NAVER', 'S-Oil']           # SK도 넣어야지 ㅎㅎ

data.info() # null값들좀 봐보려고... non-null 값들 갯수 같다!

returns = np.log(data / data.shift(1))         # 이건 날마다 수익률 계산한거고
the_number_of_stock = len(stocks)              # 갯수나 그냥 지정해놓고 가자

random_expected_return = []
random_historical_volatility = []
for p in range(2500):
    weights = np.random.random(the_number_of_stock)         # 0과 1사이 숫자 중에서 그냥 아무렇게나 종목 갯수만큼의 난수를 뽑아서
    weights = weights / np.sum(weights)                     # 모두 합하면 1이 될 수 있도록 비율 숫자로 바꿔
    random_expected_return.append(np.sum(returns.mean() * weights) * 252)       # 4개 종목 랜덤 비율 포트폴리오의 기대수익률
    random_historical_volatility.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))
                                                            # 4개 종목 랜덤 비율 포트폴리오의 표준편차(risk)
random_expected_return = np.array(random_expected_return)
random_historical_volatility = np.array(random_historical_volatility)
                                                            # 그냥 이제 이름을 약자로 쓰지 않고, 이런식으로 가보자

# 위에 뺑뻉이를 돌리면, 랜덤 비중으로 투자한 포트폴리오의 수익률과, 변동성을 2500개 짝꿍들을 만들어낼 수 있네.
########################################################################################################################
# 웨이트를 딱! 하나 집어넣으면, 그 웨이트에 대한 포트폴리오의 수익률과 변동성, 거기에 sharpe ratio까지 나오는 함수를 만들어볼까
def statistics(weights, risk_free_rate=0.00):
    weights = np.array(weights)
    random_expected_return = np.sum(returns.mean() * weights) * 252
    random_historical_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([random_expected_return, random_historical_volatility, (random_expected_return-risk_free_rate)/random_historical_volatility])
# 리턴은 3개! 수익률, 변동성, 그리고 sharpe ratio(risk-free rate=0.00으로 계산(함수에 인수로 넣을 수 있게 만들어야징~))
########################################################################################################################
# 첫 번째로 할 것은 샤프지수(변동성 대비 수익률)를 최대화하는 포트폴리오가 가장 좋은거 아니겠어?
# 샤프지수를 최대로 먹게하는 weight를 찾아줘~~~ 를 해보자!
# 근데, 최대값으로 향하게하는 optimazation 소스는 없단 말이야, 최소값을 향해 찾아가는 알고리즘 밖에 없어.
# 그러면 마이너스를 붙인 다음에, 최소값을 찾아달라고하면, 그렇게 찾아낸 것에서 마이너스만 벗겨내면 최댓값을 찾았다고 할 수 있지않겠어?
# (이차함수의 맨 꼭대기를 찾기위해서, 그냥 마이너스 붙여서 아래로 볼록하게 만들어놓은 다음에 최소값을 찾으면, 마이너스 땐 다음에 최댓값을 가져오는 x를 찾았다고 할 수 있듯이?)
import scipy.optimize as sco
def min_func_sharpe(weights):           # 이 함수에 'weight'를 인자로 넣으면, 그 weight에 해당하는 sharpe ratio에 마이너스 붙여서 반환해주는 함수
    return -statistics(weights)[2]
########################################################################################################################
# 그러면 scipy의 minimize함수를 사용할 것임. 이 함수는 그야말로 함수값을 최소로 만들어주는 x를 찾아주는 것이 목표
# 실험삼아서 하나 해볼까. y = (x-5)^2 이런 함수의 최소값은 x=5이라는 사실을 알잖아? -> 이 사실을 컴퓨터가 잘 찾는지 해보자.

def test_function(x):
    return x**2 -10*x +25
x0 = np.random.random() * 100
test_sol = sco.minimize(test_function, x0, method='SLSQP')
########################################################################################################################
# 근데, 문제는 그냥 막 volatility를 최소화하는 w벡터를 찾아달라고하기엔 좀 그래;
# 몇 가지 제약조건이 있지 않겠음?
# 첫 번째로 w벡터의 원소들은 0과 1사이의 숫자여야 함.(마이너스면 공매도, 1보다 크면 레버리지 먹였단 소리니까)
# 두 번째는 w벡터의 원소들의 합은 1이라는 것!
# 이렇게 equation/inequality 제약조건이 다 있는 경우에는 SLSQP 메써드를 쓴다고

bnds = tuple((0, 1) for x in range(the_number_of_stock))           # 이게 그 첫 번째 조건을 거는 거고
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})     # 이것은 weight의 합이 1이 되어야한다는 것이고(모든 합(1)-1. 즉, 0(False))
x0 = the_number_of_stock * [1. / the_number_of_stock]        # 초기 x는 아무 숫자로 잡자( w = (역수, 역수, 역수, ...))

sol_weight_max_sharpe = sco.minimize(min_func_sharpe, x0, method='SLSQP', bounds=bnds, constraints=cons)
sol_weight_max_sharpe['x'].round(3)                # 'x' 명령어로 찾은 weight벡터를 확인할 수 있고
statistics(sol_weight_max_sharpe['x']).round(3)    # 그 weight 벡터를 우리의 함수에 집어넣으면, 수익률, 변동성, 샤프비율까지 탕탕탕! 꺼내주겠네!
########################################################################################################################
# 이번에는 샤프지수 최대화하는 거 말고, 그냥 아싸리 변동성만을 고려해서, 변동성 가장 작게 가져오는 웨이트를 찾아달라고 해보기!
def min_func_variance(weights):
    return statistics(weights)[1] ** 2
# statistics 함수의 두 번째 리턴값은 표준편차였고, 그거의 제곱 분산을 내뱉는 함수
# 분산을 최소화시킬 것이니깐!

bnds = tuple((0, 1) for x in range(the_number_of_stock))           # 이게 그 첫 번째 조건을 거는 거고
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})     # 이것은 weight의 합이 1이 되어야한다는 것이고(모든 합(1)-1. 즉, 0(False))
# 제약조건들은 weight에 관련한 것들이었으니깐, 동일해야하겠네

sol_weight_min_var = sco.minimize(min_func_variance, x0, method='SLSQP', bounds=bnds, constraints=cons)
sol_weight_min_var['x'].round(3)
statistics(sol_weight_min_var['x']).round(3)
########################################################################################################################
# 위에서 했던 짓을 좀 정리를 해보면, 일단은 랜덤하게 2500개의 weight들을 점 찍은거 부터 했었지
stocks = ['005930.KS', '005380.KS', '034730.KS', '035420.KS', '010950.KS']
data = pd.DataFrame()
for i in stocks:
    data[i] = web.DataReader(i, data_source='yahoo', start='2010-01-01', end='2018-06-30')['Close']
data.columns = ['samsung', 'hyundai', 'SK', 'NAVER', 'S-Oil']           # SK도 넣어야지 ㅎㅎ

random_expected_return = []
random_historical_volatility = []
for p in range(2500):
    weights = np.random.random(the_number_of_stock)         # 0과 1사이 숫자 중에서 그냥 아무렇게나 종목 갯수만큼의 난수를 뽑아서
    weights = weights / np.sum(weights)                     # 모두 합하면 1이 될 수 있도록 비율 숫자로 바꿔
    random_expected_return.append(np.sum(returns.mean() * weights) * 252)       # 4개 종목 랜덤 비율 포트폴리오의 기대수익률
    random_historical_volatility.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))
                                                            # 4개 종목 랜덤 비율 포트폴리오의 표준편차(risk)
random_expected_return = np.array(random_expected_return)
random_historical_volatility = np.array(random_historical_volatility)


plt.figure(figsize=(10, 6))
plt.scatter(random_historical_volatility, random_expected_return, c=random_expected_return/random_historical_volatility, marker='.')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.title('Samsung/Hyundai/SK/NAVER/S-Oil Portfolio')
plt.colorbar(label='sharpe ratio')                          # 그래도 이중엔 가장 효율적인 포트폴리오 weight를 가져가는게 있지 않겠냐~
# plt.savefig('3.png')

# 이 점들중에는 따아아아악 좋은애가 있을거란 말이야.
# 근데 가장 좋다라는게 개인의 가치관에 따라서
# 변동성 대비 수익률이 높은 포트폴리오가 가장 좋다고 하는 사람도 있을것이고
bnds = tuple((0, 1) for x in range(the_number_of_stock))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
sol_weight_max_sharpe = sco.minimize(min_func_sharpe, x0, method='SLSQP', bounds=bnds, constraints=cons)
sol_weight_max_sharpe['x'].round(3)                # 'x' 명령어로 찾은 weight벡터를 확인할 수 있고
statistics(sol_weight_max_sharpe['x']).round(3)
# 어떤 사람은 그냥 risk 자체가 가장 낮은 포트폴리오가 가장 좋다고 하는 사람도 있을 것이고
sol_weight_min_var = sco.minimize(min_func_variance, x0, method='SLSQP', bounds=bnds, constraints=cons)
sol_weight_min_var['x'].round(3)
statistics(sol_weight_min_var['x']).round(3)

# 위 그림에다가 점 찍어보면
plt.plot(statistics(sol_weight_max_sharpe['x'])[1], statistics(sol_weight_max_sharpe['x'])[0], 'r*', markersize=15)
plt.plot(statistics(sol_weight_min_var['x'])[1], statistics(sol_weight_min_var['x'])[0], 'b*', markersize=15)
# plt.savefig('4.png')
########################################################################################################################
# 여기까지 먼저 그린 이유는, 한 가지 안 그린 그림이 있음.
# 그건 바로 마코위츠 아저씨의 Efficient Frontier를 그리지 않았어!!!!!
# 저기 따아아아아악 경계선에 있는 것들만 좀 찾아야한단 말이지? 그거를 그리러 가보자!

# 지배원리 공부할 때 배웠듯이 그냥 "똑같은 기대수익률을 가져가는 포트폴리오중에서 -> 변동성이 가장 작은 것"
# 이런것만 골라 낸 것이 마코위츠 아저씨의 Efficient Frontier이니깐
# 위에서 했던 min variance를 전체에 대해서 사용하는것이 아니라,
# '각각의 수익률 수준에서~' 각각의 min-variance 포트폴리오를 찾아줘! 라고하면 Efficient Frontier를 그릴 수 있겠다.

# 아까 그림보면 수익률 최댓값이 20% 정도가 최대였으니깐
return_respectively = np.linspace(0.05, 0.2, 100)
bnds = tuple((0, 1) for x in range(the_number_of_stock))
sol_volatility=[]
for i in return_respectively:
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: statistics(x)[0] - i})
                                                            # 루프가 돌면서 이 전의 수익률보다 높은 수준에서 min-var를 찾아달라고 해야하니깐, 이런 조건이 추가되어서 루프안에 있어야 해
    sol_weight_min_var = sco.minimize(min_func_variance, x0, method='SLSQP', bounds=bnds, constraints=cons)
    sol_volatility.append(np.sqrt(sol_weight_min_var['fun']))            # 목적함수 자체가 분산이었으니깐, 루트붙이면서 꺼내며 표준편차로 바꾸자
sol_volatility = np.array(sol_volatility)
plt.plot(sol_volatility, return_respectively, 'gx')
# plt.savefig('5.png')
########################################################################################################################
# 아 근데, 민베리언스포트폴리오 밑에있는 포트폴리오는 의미없는 포트폴리오들 아니야?
# 그거 잘라내는거 마지막으로 딱 한번만 그리고 다음으로 갑시다!
# 그냥 linspace에서 자르는 곳에 장난 한번만 더 치면 될 것 같으니깐....ㅎㅎ
bnds = tuple((0, 1) for x in range(the_number_of_stock))
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
sol_weight_min_var = sco.minimize(min_func_variance, x0, method='SLSQP', bounds=bnds, constraints=cons)
mu1 = statistics(sol_weight_min_var['x'])[0]
return_respectively = np.linspace(mu1, 0.2, 100)
bnds = tuple((0, 1) for x in range(the_number_of_stock))
sol_volatility=[]
for i in return_respectively:
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'eq', 'fun': lambda x: statistics(x)[0] - i})
                                                            # 루프가 돌면서 이 전의 수익률보다 높은 수준에서 min-var를 찾아달라고 해야하니깐, 이런 조건이 추가되어서 루프안에 있어야 해
    sol_weight_min_var = sco.minimize(min_func_variance, x0, method='SLSQP', bounds=bnds, constraints=cons)
    sol_volatility.append(np.sqrt(sol_weight_min_var['fun']))
sol_volatility = np.array(sol_volatility)
plt.plot(sol_volatility, return_respectively, 'ko')
# plt.savefig('6.png')
########################################################################################################################
# Efficient Frontier 그렸으니, 이제는 New Efficient Frontier 그릴차례겠구만.
# New Efficient Frontier는 y축위에 (0, rf)에서 Efficient Frontier로 접선을 그으면 그게 New Efficient Frontier이니깐
# 두 점 딱! 잡아가꼬, -> 보간법으로 그 사이를 1차 직선으로 채우면 되겠다 그치
import scipy.interpolate as sci

# 그러면 필요한 작업이
# 첫 번째로는 Efficient Frontier만 뽑아내는 작업이 필요하고
# 도함수로도 장난칠 수 있도록 장치가 필요하겠군

# 그럼 먼저 최소의 변동성을 갖고 있는 애들이 어디에있는지 그 위치를 뽑아내주는 argmin함수를 이용하자
ind = np.argmin(sol_volatility)
Efficient_Frontier_Volatility = sol_volatility[ind:]
Efficient_Frontier_Return = return_respectively[ind:]

interpol_model = sci.splrep(Efficient_Frontier_Volatility, Efficient_Frontier_Return, k=3)      # Efficient Frontier 들만 잡아서 그 선을 보간 모형으로 만들고,
                                         # 효율적 투자'선'을 함수로 만들거잖아


def f(x):
    return sci.splev(x, interpol_model, der=0)      # 엑스를 넣어서 인터폴 모델에 맞는 y를 만들어내고
def df(x):
    return sci.splev(x, interpol_model, der=1)      # 그거의 도함수도 만들어내라는 말


def equations(p, rf=0.01):                          # p는 (a, b, x)를 집어넣을거야
    eq1 = rf - p[0]                                 # rf = a (이를 만족시키는 a이어야하고)
    eq2 = rf + p[1]*p[2] - f(p[2])                  # rf + bx = f(x) (rf+bx는 f(x) 위에 있는 점이어야 하고)        # f(x): 효율적투자선
    eq3 = p[1] - df(p[2])                           # b = df(x)      (rf+bx는 f(x)에 접접에 있어야 한다)
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [3, 3, 3])          # fsolve는 f(x)=0의 해를 찾아주는 함수라고 생각할 수 있음!
                                                        # equation이라는 함수값을 0으로 맞춰주는 x를 찾아달라구 하는 것이네!
                                                        # 근데 우리는 equation이 3개의 연립방정식이니깐, 저 식 모두 0으로 만드는 p를 찾아달라는 것!
                                                        # [3, 3, 3]은 그냥 초기에 넣어볼 (a, b, x)쌍을 넣어본 것이고
                                                        # return 하기를... 다시 함수에 넣었을 때 모두 0을 만족시키는 a, b, x를 리턴할 것이라고

np.around(equations(opt), 6)                            # 싱기방기 동방싱기하게도 정말 모두 0을 만드네

plt.figure(figsize=(10, 6))
plt.scatter(random_historical_volatility, random_expected_return, c=random_expected_return/random_historical_volatility, marker='.') # 랜덤하게 뽑은 점들
plt.plot(sol_volatility, return_respectively, 'ko')

New_x = np.linspace(0.0, 0.4)
plt.plot(New_x, opt[0] + opt[1]*New_x, lw=1.5)
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)      # 접점이여~
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe Ratio')
plt.savefig('7.png')
########################################################################################################################
# rf=0.01때 말고, 0.04, 0.1 두개만 더 그려보자
def equations(p, rf=0.04):                          # p는 (a, b, x)를 집어넣을거야
    eq1 = rf - p[0]                                 # rf = a (이를 만족시키는 a이어야하고)
    eq2 = rf + p[1]*p[2] - f(p[2])                  # rf + bx = f(x) (rf+bx는 f(x) 위에 있는 점이어야 하고)        # f(x): 효율적투자선
    eq3 = p[1] - df(p[2])                           # b = df(x)      (rf+bx는 f(x)에 접접에 있어야 한다)
    return eq1, eq2, eq3
opt2 = sco.fsolve(equations, [3, 3, 3])          # fsolve는 f(x)=0의 해를 찾아주는 함수라고 생각할 수 있음!

def equations(p, rf=0.1):                          # p는 (a, b, x)를 집어넣을거야
    eq1 = rf - p[0]                                 # rf = a (이를 만족시키는 a이어야하고)
    eq2 = rf + p[1]*p[2] - f(p[2])                  # rf + bx = f(x) (rf+bx는 f(x) 위에 있는 점이어야 하고)        # f(x): 효율적투자선
    eq3 = p[1] - df(p[2])                           # b = df(x)      (rf+bx는 f(x)에 접접에 있어야 한다)
    return eq1, eq2, eq3
opt3 = sco.fsolve(equations, [3, 3, 3])

plt.figure(figsize=(10, 6))
plt.scatter(random_historical_volatility, random_expected_return, c=random_expected_return/random_historical_volatility, marker='.') # 랜덤하게 뽑은 점들
plt.plot(sol_volatility, return_respectively, 'ko')

New_x = np.linspace(0.0, 0.4)
plt.plot(New_x, opt[0] + opt[1]*New_x, lw=1.5, label='rf=0.01', color='r')
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)      # 접점이여~

plt.plot(New_x, opt2[0] + opt2[1]*New_x, lw=1.5, label='rf=0.04', color='g')
plt.plot(opt2[2], f(opt2[2]), 'g*', markersize=15.0)

plt.plot(New_x, opt3[0] + opt3[1]*New_x, lw=1.5, label='rf=0.1', color='b')
plt.plot(opt3[2], f(opt3[2]), 'b*', markersize=15.0)

plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe Ratio')
plt.legend()
plt.title('Samsung/Hyundai/SK/NAVER/S-Oil Portfolio \n New Efficient Frontier')
plt.savefig('9.png')
########################################################################################################################