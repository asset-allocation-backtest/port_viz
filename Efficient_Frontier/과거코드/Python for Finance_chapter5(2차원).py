# 필요한 모듈 ~
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

path = r'C:\Users\GD Park\iCloudDrive\gdpre_PycharmProjects\Python for Finance'
name = 'chapter5. 데이터 시각화'
folder = path + '/' + name + '/'
# os.mkdir(folder)
os.chdir(folder)

########################################################################################################################
np.random.seed(1000) # 변수 초기화
y= np.random.standard_normal(20) # Z(0,1) 분포에서 변수 20개 추출
########################################################################################################################
x = range(len(y))                       # y를 맞춰 찍을 x를 range로 생성
plt.figure(figsize=(10, 6))             # 사이즈 원하는 걸로 하나 만들어놓고
plt.plot(x, y)                          # 여기에 그리기~ 히힛!
# plt.plot(y)만 해도 자동으로 맞춰줌
plt.savefig('1.png')
plt.show()                              # 일단 한 번 보여줘바 ㅎㅎ!
########################################################################################################################
plt.figure(figsize=(10, 6))             # 새로운 그림 그릴 판때기 하나 갖다놓고
y2 = y.cumsum()                         # 이번엔 y에 누적 합이라는 장난질을 한 번 쳐서 그려볼까
# y                                       # y가 원래 이렇게 생긴 애라면
# y2                                      # cumsum()한 애는 이렇게 생긴애고
plt.plot(x, y2)                         # 그려랏
plt.grid(True)                          # 판떼기에 모눈선을 넣는 방법은 이렇게 하는 것이고
plt.axis('tight')                       # 이거는 plt한테 '모든 자료가 보이도록 축 범위를 조밀하게 맞추라'는 명령
plt.savefig('2.png')                    # 저장스~

# plt.axis()의 명령으로 할 수 있는건 더 여러가지가 있음
"""
plt.axis()                                   현재 축의 한계 값을 반환해랏! 
plt.axis('off')                              축 선과 라벨을 없애랏
plt.axis('equal')                            가로축과 세로축의 스케일을 같게 하랏!
plt.axis('scaled')                           가로축과 세로축의 스케일을 같아지도록 크기 자체를 조정하랏!
plt.axis('tight')                            모든 자료가 보이도록 축 범위를 조밀하게 맞추라
plt.axis('image')                            모든 자료가 보이도록 축 범위를 자료의 극한값으로 조정하랏!
plt.axis([ximn, xmax, ymin, ymax])           축 범위를 주어진 리스트 값으로 하랏!!!!
"""
########################################################################################################################
plt.figure(figsize=(10, 6))             # 새로운 그림 그릴 판때기 하나 갖다놓고
plt.plot(y2)                            # 또 똑같이 하나 그려보고, 다른것도 살펴봐보자. 뭘 살펴볼거냐면
# plt.xlim(-1, 20)                      # 책에는 이렇게 나와있는데 아래의 것과 흐름을 같이 하려면 이런식으로 할 수 있겠다
plt.xlim(np.min(x)-1, np.max(x)+1)      # 책에는 이렇게 나와있는데 아래의 것과 흐름을 같이 하려면 이런식으로 할 수 있겠다
plt.ylim(np.min(y2) -1, np.max(y2)+1)   # Max와 Min 매써드를 이용해서 축의 한계를 plt.xlim()으로 설정하는 것도 있음! ㅎㅎㅎ
plt.savefig('3.png')                    # 저장스~
########################################################################################################################
plt.figure(figsize=(10, 6))             # 새로운 그림 그릴 판때기 하나 갖다놓고
# 이번에는 선들의 스타일이나 색깔, 너비 바꾸는 거 해보자
# plt(y2) 이걸로 그리면 그냥 디폴트로(약간 하늘색 -> 약간 주황색 ->)으로 그리는데, 아예 지정해서 그릴 수도 있다는 거지
plt.plot(y2, color='b', lw=1.5)          # 색깔을 color=로 지정을하고, 선의 두께는 lw=로 지정하면 됨(line width의 lw겠지 뭐)
plt.plot(y2.cumsum(), color='r', lw=3.5) # blue는 확실히 얇게 나오고, Red는 확실히 두껍게 나오는 거 보이지?
plt.grid(True)
plt.axis('tight')                        # 이건 아까 한거고
plt.xlabel('index')                      # 이건 x축 이름 설정하는 것이여~
plt.ylabel('value')                      # 그럼 이건 y축 이름 설정하는 거겠지
plt.title('value')                       # 이건 그냥 아예 제목!
plt.savefig('4.png')                     # 저장스~

# 그럼 이제 그냥 matplotlib에서 표준으로 제공하는 색깔 명령어랑, 선or점style 명령어나 알고 넘어가자
"""
color=
b: 파란색                 r: 빨강색
m: 자홍색(마젠타)         k: 검정색
g: 녹색                   c: 청록색(시안)
y: 노랑색                 w: 흰색
"""
"""
marker= 이거랑 linestyle= 이거는
캡쳐파일 참조
"""
plt.figure(figsize=(10, 6))
plt.plot(y2, color='b', lw=1.5, marker='o', linestyle='--')             # 그 옵션들은 이런식으로 집어넣으면 되는 것이여~
plt.plot(y2.cumsum(), color='r', lw=3.5, marker='d', linestyle=':')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('value')
plt.savefig('5.png')
########################################################################################################################
# 2차원
np.random.seed(1000) # 변수 초기화
y = np.random.standard_normal((20, 2))          # 이번에는 행렬을 뽑았으. 20행 2열 뽑았다고 볼 수 있는건가?
                                                # 원소 2개짜리 벡터를 20개 쌓아서 올린 행렬을 만들었으!
# y
y=y.cumsum(axis=0)                              # cumsum을 하더라도 아래로 더할건지, 옆으로 더할껀지를 정해야겠지
# y2=y.cumsum(axis=1)                             # 첫번째 차원 방향으로 더하라고 하는 건 axis=0, 두 번째 차원 방향으로 더하라고 하는 건 axis=2를 옵션으로 넣으면 됨.
                                                  # 이렇게 말하는 이유는, 나중에 차원이 n차원으로 갈 수도 있으니까;
# y2
# y
# 암튼 이제 그려보자
plt.figure(figsize=(10, 6))
plt.plot(y[:, 0], lw=3, color='r')
plt.plot(y[:, 1], lw=1, color='b')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A simple plot')
plt.savefig('6.png')
# 근데 보면 이제, 빨간선 데이터의 이름이랑, 파란선데이터 이름을 붙여준 다음에 설명해주는 작은 표를 그려주고 싶게 생기지 않음?
########################################################################################################################
# 그거는 이렇게 하면됨
plt.figure(figsize=(10, 6))
plt.plot(y[:, 0], lw=3, color='r', label='내가 red다')         # 여기에 label= 으로 데이터 이름을 설정할 수 있어
plt.plot(y[:, 1], lw=1, color='b', label='내가 blue다')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A simple plot')
plt.legend(loc=0)                                               # 라벨 설정한 거를 작은 표로 그려달라고 하는 거는 legend()를 쓰면 됨
                                                                # optional로 넣을 수 있는 건 잠시후 적어놓겠음!
plt.savefig('7.png')
# 이러면, 오류는 아니지만 빨간색 글씨 오지게 뜨면서, 한글이 깨지거든? 이럴때는 font를 지정하면 한글이 깨지지 않음
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname=r"c:\Windows\Fonts\malgun.ttf").get_name()
rc('font', family=font_name)
plt.figure(figsize=(10, 6))
plt.plot(y[:, 0], lw=3, color='r', label='내가 red다')
plt.plot(y[:, 1], lw=1, color='b', label='내가 blue다')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A simple plot')
plt.legend(loc=0)
plt.savefig('8.png')                # 한글 안깨지지롱
# optional로 넣을 수 있는 legend(loc=)이나 정리하고 넘어가자
"""
loc=
legend() 이렇게 그냥 아무것도 설정 안하면 '자동'으로 알아서 넣어달라는 거고
loc=0       가능한 최적의 위치
loc=1       오른쪽 위
loc=2       왼쪽 위
loc=3       왼쪽 아래
loc=4       오른쪽 아래
loc=5       오른쪽
loc=6       왼쪽 중앙
loc=7       오른쪽 중앙
loc=8       중앙 아래
loc=9       중앙 위
loc=10      중앙
"""
########################################################################################################################
plt.figure(figsize=(10, 6))
y[:, 1] = y[:, 1] * 100
plt.plot(y[:, 0], lw=3, color='r', label='1st')
plt.plot(y[:, 1], lw=1, color='b', label='2nd')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A simple plot')
plt.legend()
plt.savefig('9.png')            # 이렇게 스케일이 안맞아서 생기는 문제는 어떻게 해결할 수 있을까?
########################################################################################################################
# 2가지 방법이 있음
# 1. y축을 두개 놓는 법
# 2. 두개의 subplot을 놓는 법(그냥 한개의 창에 그림 2번 그리겠다는 거지)
########################################################################################################################
# 그러면 먼저 하나의 창에 2개의 y축을 놓는 거를 해보자
# 잠깐! 그 전에 음수가 깨지는 현상이 또 벌어질 수 있단말이여?
# 사실 이것도 설정을 해줘야 마이너스 부호가 깨지지 않음. 그 방법은
mpl.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 6))         # subplot이라는 애는 변수를 두개 받는 것이 유용하니 참고!
                                                # (이름을 뭐라 지정하든) 첫번째 변수는 figure제어하는 애고, 두번째 애는 axis 아니라, AXES를 제어하는 애임
np.random.seed(1000) # 변수 초기화
y = np.random.standard_normal((20, 2)).cumsum(axis=0)
y[:, 1] = y[:, 1] * 100
plt.plot(y[:, 0], lw=3, color='r', label='1st') # 첫 번째 데이터를 그리고
plt.grid(True)
plt.axis('tight')                               # 첫 번째 데이터에 해당하는 option들을 설정
plt.legend(loc=8)                                    # 첫 놈의 범례
plt.xlabel('index')
plt.ylabel('value 1st')                         # 이건 첫 번째 데이터의 y축을 붙여주는 거고
plt.title('A simple plot')
ax2 = ax.twinx()################################# 여기서 두 번째 축을 설정하고###########################################
plt.plot(y[:, 1], lw=1, color='b', label='2nd') # 그 축에 넣어서 그릴 애를 '여기에서' 그리면 됨
plt.ylabel('value 2st')                         # 여기서 설정하는 ylabel은 2번째 데이터 y축을 설정하는 것이겠네
plt.legend(loc=9)                               # 두번째 애 범례!
plt.savefig('10.png')
########################################################################################################################
# 이번에는 한 개의 창에 2개의 그래프를 각각 그리는 거를 봐보자
fig, ax = plt.subplots(2, 1, figsize=(10, 6))   # 위에서 했던 것 처럼, 이렇게 창 하나에 2개의 표를 넣어놓고 시작을 해도 되고
                                                # 2, 1의 의미는 행의 갯수, 열의 갯수임.
plt.savefig('11.png')
fig, ax = plt.subplots(2, 2, figsize=(10, 6))   # 행으로 2개, 열로 2개의 그래프 축을 놓고 시작하겠다는 것이고
plt.savefig('12.png')
fig, ax = plt.subplots(3, 6, figsize=(10, 6))   # 행으로 3개, 열로 6개의 그래프 축을 놓고 시작하겠다는 것이고
plt.savefig('13.png')
########################################################################################################################
fig = plt.figure(figsize=(10, 6))               # figure로 창을 딱 만들어 놓고
# fig = plt.subplot(2, 1)
fig = plt.subplot(211)                          # 211로 이렇게 할 수도 있어
                                                # 앞에 두 숫자는 위에서 했던 그 행의 갯수와, 열의 갯수이고
                                                # 그 뒤에 숫자는 '내가 지금 선택한 창'을 의미
                                                # 여기선 즉, 행으로 2개 열로 1개 만들었을 때, 첫번째 창을 말하는 거고
plt.savefig('14.png')

fig = plt.figure(figsize=(10, 6))
fig = plt.subplot(212)                          # 그럼 이거는 행으로 2개 열로 1개 만들었을 떄, 두번째 창을 말하는 걸꺼고
plt.savefig('15.png')

fig = plt.figure(figsize=(10, 6))
fig = plt.subplot(231)                          # 그럼 이거는 행으로 2개 열로 1개 만들었을 때, 두번째 창을 말하는 걸꺼고
plt.savefig('16.png')

fig = plt.figure(figsize=(10, 6))
fig = plt.subplot(235)                          # 행으로 2개 열로 1개 만들었을 때, 다섯번째 창을 말하는 거겠네
plt.savefig('17.png')
########################################################################################################################
# 암튼 창 만드는 건 이해갔고. ( 이제부터는 변수명 빼고 끌고 가겠습니다 )
plt.figure(figsize=(10, 6))               # figure로 창을 딱 만들어 놓고
plt.subplot(211)                          # 행으로 2개 열로 한개 만들고, 거기에서 첫 번째 창!
plt.plot(y[:, 0], lw=3, color='r', label='1st') # '그 창에' 첫 번째 데이터를 그리고
plt.grid(True)                                  # '그 창에' grid 넣고
plt.axis('tight')                               # '그 창에'
plt.legend()                                    # '그 창에'
plt.xlabel('index')                             # '그 창에'
plt.ylabel('value 1st')                         # '그 창에'
plt.title('A simple plot')                      # '그 창에'
plt.subplot(212)                                # 행으로 2개 열로 한개 만들었을 때, 거기에서 두 번째 창!
plt.plot(y[:, 1], lw=1, color='b', label='2nd') # '그 창에'
plt.ylabel('value 2st')                         # '그 창에'
plt.grid(True)                                  # '그 창에'
plt.axis('tight')                               # '그 창에'
plt.legend()                                    # '그 창에'
plt.savefig('18.png')
########################################################################################################################
# 자꾸 위에서 "~ 만들 었을 때, 몇 번째 창!" 이렇게 말했던 이유는, style을 이렇게도 만들 수 있기 때문이였음.
plt.figure(figsize=(10, 6))               # figure로 창을 딱 만들어 놓고
plt.subplot(211)
# ~ 블라블라 그림 그려주고
plt.subplot(223)
# ~ 블라블라 그림 그려주고
plt.subplot(224)
# ~ 블라블라 그림 그려주고
plt.savefig('19.png')
########################################################################################################################
# 이걸 활용하면, 하나의 창에 서로 다른 style로 그림을 그릴 수도 있겠네!
plt.figure(figsize=(10, 6))               # figure로 창을 딱 만들어 놓고
plt.subplot(121)
plt.plot(y[:, 0], lw=3, color='r', label='1st')
plt.grid(True)
plt.axis('tight')
plt.legend()
plt.xlabel('index')
plt.ylabel('value 1st')
plt.title('First Data Set')

plt.subplot(122)
# plt.bar(y[:, 1], width=3, color='r', label='2st') bar를 그리는 건, 이렇게 넣으면 안써져.....
                                                    # 막대기가 들어갈 자리를 range형태의 시퀀스로 지정을 해줘야해
                                                    # 막대기는 내가 넣을 데이터의 갯수만큼 필요한 거니깐
plt.bar(range(len(y[:,1])), y[:, 1], width=0.5, color='b', label='2st')
                                                    # 막대기의 두께는 lw가 아니라, width= 옵션으로 설정하면 됨
plt.grid(True)
plt.axis('tight')
plt.legend()
plt.xlabel('index')
plt.ylabel('value 2st')
plt.title('Second Data Set')
plt.savefig('20.png')
########################################################################################################################
# 그림을 그리는데 다른 스타일 그림도 한 번 그려보자. 점으로 확 뿌려서 찍는 scatter!
y = np.random.standard_normal((1000, 2))            # 변수 오지게 만들어놓고
plt.figure(figsize=(10, 6))
plt.scatter(y[:, 0], y[:, 1], color='r', label='1st', marker='p')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2st')
plt.title('Scatter Plot')
plt.savefig('21.png')
########################################################################################################################
# 점들마다한테 값을 줄 수 있어.
# 3차원 면으로 그릴꺼냐고? 아니아니 3차원 면은 이 뒤에서 다루게될 것 같고
# 높이로 차원을 하나 확장하는 거 말고, color bar를 이용해서 차원 하나 더 확장하는 거를 다뤄보겠음
rnd = np.random.randint(0, 10, len(y))
plt.figure(figsize=(10, 6))
plt.scatter(y[:, 0], y[:, 1], c=rnd, label='1st', marker='o')   # 3번째 차원은 c= 이라고 넣어주며
plt.colorbar()                                                  # 옆에 컬러바 하나 세워달라고 이렇게 말하면 되고
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')
plt.savefig('22.png')
########################################################################################################################
# 히스토그램도 간단하게 그려보자!!! hist()함수를 쓰면 돼!
plt.figure(figsize=(10, 6))
plt.hist(y, label=['1st', '2nd'], bins=25)      # 데이터내의 값들을 25구간으로 쪼개서 얼만큼의 빈도로 나오느냐를 카운트해서 그리는 것!
plt.grid(True)
plt.xlabel('value')
plt.ylabel('frequency')
plt.legend()
plt.title('Histogram')
plt.savefig('23.png')
########################################################################################################################
plt.figure(figsize=(10, 6))
plt.hist(y, label=['1st', '2nd'], color=['r', 'g'])               # 디폴트 bins= 값은 10이라는 것을 알 수 있음!!
plt.grid(True)
plt.xlabel('value')
plt.ylabel('frequency')
plt.legend()
plt.title('Histogram')
plt.savefig('24.png')
"""
plt.hist(x, bins=10, range=None, weights=None, cumulative=False, bottom=None,.....)
x: 리스트 혹은 ndarray를 넣으면 됨
bins: 빈도 구분값의 수
range: 빈도  구분의 위 아래 범위
normed: 전체 값의 합이 1이 되도록 정규화하는지의 여부
weights: x에 대한 가중치
cumulative: 각 빈도 구분값이 하위의 빈도구분값을 누적하는지의 여부
histtype: 옵션(문자열): bar, barstacked, step, stepfilled
align: 옵션(문자열): left, mid, right
orientation: 옵션(문자열): horizontal, vertical
rwidth: 각 막대의 상대적인 폭
log: 로그 스케일
color: 각 자료의 색
stacked: 여러개의 자료를 쌓아올려서 표시하는지의 여부
"""
########################################################################################################################
plt.figure(figsize=(10, 6))
plt.hist(y, label=['1st', '2nd'], color=['r', 'b'], stacked=True, bins=20)               # 데이터를 쌓아주네~~~~ 이것도 쓸 일 많을 수 있겠다!
plt.grid(True)
plt.xlabel('value')
plt.ylabel('frequency')
plt.legend()
plt.title('Histogram')
plt.savefig('25.png')
########################################################################################################################
# matplotlib.finance라는 라이브러리도 있다는데, 더이상 제공하지 않는다고 하네.
# 인터넷에서 그거 copy해 놓은 모듈을 구했으니! 그걸로 해보자!
import mpl_finance as mpf
from pandas_datareader import data as web
start = '2019-09-01'
end = '2019-09-09'
df = web.DataReader('^KS11', data_source='yahoo', start=start, end=end)         # data를 코스피를 가져와 볼까?
type(df) # 얘는 DataFrame이네ㅎㅎㅎ 유후!
df = df.reset_index()
""" 
최근 몇몇 파이썬 라이브러리들이 야후 파이낸스로부터 주가 자료를 가져오는 함수를 제공한다.
이러한 함수를 사용하면 금융 자료를 시각화하는 데 편하기는 하지만, 이러한 자료의 품질은 중요한 투자 결정을 하기에는 충분하지 않다
예를들어, 야후 파이낸스는 주식 분할에 의한 주가 변동을 적절히 처리하지 않고 있다. 무료로 자료를 제공하는 다른 라이브러리들도 마찬가지이다.
"""
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(bottom=0.2)                 # subplots_adjust은 subplot들의 위치를 조정해주는 놈
# subplots_adjust(left=, bottom=, right=, top=, hspace=, wspace= ) 이렇게 제어할 수 있고
# hspace, wspace는 subplot들 간의 간격을 조정하는 것
mpf.candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'], width=0.5, colorup='r', colordown='b')
plt.grid(True)
plt.savefig('26.png')
# y축은 값으로 알아서 잘 잡는데 x축은 날짜라고 잡질 못하네..
# 그거만 해결해보자
########################################################################################################################
start = '2019-08-01'
end = '2019-09-09'
df = web.DataReader('^KS11', data_source='yahoo', start=start, end=end)         # data를 코스피를 가져와 볼까?
df = df.reset_index()
import matplotlib.ticker as ticker
# ticker를 이용해서 x축을 잡을 것이여
day_list = []
name_list = []
for i, day in enumerate(df['Date']):
    if day.dayofweek == 0:          # 월요일만 잡는다는 것
        day_list.append(i)
        name_list.append(day.strftime('%Y-%m-%d'))

fig, ax = plt.subplots(figsize=(10, 6))
mpf.candlestick2_ohlc(ax, df['Open'], df['High'], df['Low'], df['Close'], width=0.5, colorup='r', colordown='b')
ax.xaxis.set_major_locator(ticker.FixedLocator(day_list))           # x축의 위치를 설정하는 것이 set_major_locatot
ax.xaxis.set_major_formatter(ticker.FixedFormatter(name_list))      # 거기에 넣을 값을 설정하는 것이 set_major_formatter
plt.grid(True)
plt.savefig('27.png')
########################################################################################################################
# 이제는 그럼 3차원을 그리는 걸 마지막으로 해서 끝내버리자!
from mpl_toolkits.mplot3d import Axes3D
strike =np.linspace(50, 150, 24)                    # 50부터 150사이의 숫자를 24등분해서 linspace만들어주는 거고
ttm = np.linspace(0.5, 2,5, 24)                     # 0.5부터 2.5사이의 숫자들을 24등분해서 space만들고
strike, ttm = np.meshgrid(strike, ttm)              # 그거 이제 평면을 만드는 것이디!

iv = (strike-100) ** 2 / (100 * strike) / ttm       # 딱딱 대응 되는 값에 iv를 계산한 것을 z값으로 만들고, 이거 이제 시각화하는 게 일이니깐,
fig = plt.figure(figsize=(10, 6))                   # 이건 판때기를 만들어주는 거고
ax = fig.gca(projection='3d')                       # 그 Axes에 대한 정보는 gca()가 말하는 건데, Axes를 3d 판떼기로 바꾸는 건 이거야
surf = ax.plot_surface(strike, ttm, iv, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0.1, antialiased=True)
                                                    # 본격적으로 surface를 그리는 것, 일단 x, y, z 값을 넣는건 너무나 당연해 보이고 이외의 option들을 아래에 기록 ㄱㄱ
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')                 # 2차원을 그릴때와 다르게, 3차원을 그릴 때 축 이름은 이렇게 넣음
fig.colorbar(surf, shrink=0.5, aspect=5)            # 컬러바는 저번에도 넣어봤으니깐~
plt.savefig('28.png')
"""
plot_surface의 인수
rstride=: 배열의 행 간격
cstride=: 배열의 열 간격
color=: 곡면 색깔
cmap=: 곡면에 사용될 칼라 맵
facecolors=: 개별 곡면 패치의 색
norm=: 값으로부터 색을 선택하는 데 사용되는 normalize 객체
vmin=: 색으로부터 표시할 최솟값
vmax=: 색으로부터 표시할 최댓값
shade= 그림자 표시 여부
"""
########################################################################################################################