import numpy as np
import pandas as pd

# 앞에서는 DataFrame객체에 대해서 그냥 간단하게 한 번 본것일 뿐이고
# 이번에는 Numpy를 DataFrame으로 변환해서 무슨 짓거리를 할 수 있는지를 봐보자.

# 일단은 numpy로 많이 했었던 난수 행렬을 좀 만들어오겠습니다.
a = np.random.standard_normal((9, 4))
print(a)                                            # 9행 4열짜리 난수 행렬을 만들었고
a = a.round(6)
print(a)                                            # 소수 6번째까지로 좀 자르고
                                                    # 이제 이 a를 DataFrame으로 만들어볼까욤?!

df = pd.DataFrame(a)
print(df)                                           # 이렇게 넣으면 콜럼과 인덱스는 자동으로 0 ~ n으로 만들어 주네요
print(type(df))
"""DataFrame()의 인수
DataFrame(data / index / columns / dtype / copy) 에서
data : ndarray/dict/DataFrame -- DataFrame자료; dict은 series, ndarray, list 포함 가능 
index : 인덱스, 디폴트는 range(n)
columns : 열제목, 디폴트는 range(n)
dtype : 자료형을 특정하는 경우에 사용, 없으면 자료에서 추정
copy : 자료를 copy하는 지 여부
"""
# 위 처럼 할당하고 나서, columns를 만들어낼 수도 있음
df.columns = [['No1', 'No2', 'No3', 'No4']]
print(df)
print("-" * 50, '구분좀')
# ---------------------------------------------------------------------------------------------------------------------

# 금융시계열 자료를 효율적으로 만질라면, '시간 인덱스'를 잘 쓸 수 있어야 하는데,
# 그럴라면 pandas에서 제공하는 date_range() 매써드를 잘 써야 해

dates = pd.date_range('2015-1-1', periods=9, freq='M')
print(dates)
# 이런식으로 DateTimeIndex객체를 생성해냄.
# 원하는대로 생성해 낼라면, 저기 괄호안에 뭐가 들어가는지 잘 알아야겠지? 그거 들어간다
"""
date_range( 함수의 인수 : start / end / periods / freq / tz / normalize / name)
start       : 문자열/datetime      -- 생성하고자 하는 날짜의 시작 시각
end         : 문자열/datetime      -- 생성하고자 하는 날짜의 종료 시각
periods     : 정수/None            -- 주기(시작 시각이나 종료시작이 None인 경우)
freq        : 문자열/DateOffset    -- 주기 문자열. 5일 간격인 경우 5D             <- 요놈의 경우 밑에다가 옵션을 다 써놓겠음
tz          : 문자열/None          -- 협정 세계시가 아닌 경우 시간대 이름
normalize   : 불리언/디폴트 None    -- 시작시각과 종료 시작을 자정으로 정규화
name        : 문자열/디폴드 None    -- 인덱스 이름

freq 인수의 값으로는 이렇게 넣으면 돼
B       : 영업일 단위 주기             C       : 변경된 영업일 단위 주기
D       : 역일(calendar day)         W       : 주 단위 주기
M       : 월 단위 주기                BM      : 영업일 월말 주기
MS      : 월초 주기                  BMS     : 영업일 월초 주기
Q       : 분기말 주기                 BQ      : 영업일 분기말 주기
QS      : 분기초 주기                BQS     : 영업일 분기초 주기              (앞)B는 영업일 / (뒤)S는 월초 라는 규칙은 볼 수 있다
A       : 연말 주기                  BA      : 영업일 연말 주기
AS      : 연초 주기                  BAS     : 엽업일 연초 주기
H       : 시간 단위 주기               T       : 분 단위 주기
S       : 초 단위 주기                 L       : 밀리초 단위 주기
U       : 마이크로초 단위 주기
"""
# 이 객체를 왜만들었느냐? 내가 가진 DataFrame의 인덱싱을 먹일라고 한 거잖아? 이제 매겨보자
df.index = dates
print(df)                       # 오오오오 인덱스 싹 날짜들로 매겼다 ㅎㅎ

print("-" * 50, '구분좀')
# ---------------------------------------------------------------------------------------------------------------------
# Numpy를 DataFrame으로 바꿔서 만지는 거, 가능한 거 알겠는데
# DataFrame을 Numpy로 바꿔서 만지는 건 가능하냐? -> ㅇㅇㅇㅇㅇㅇ 가능함

b = np.array(df).round(6)
print(b)
print(type(b))                # 가능가능

print("-" * 50, '구분좀')
# ---------------------------------------------------------------------------------------------------------------------
# 뭐 암튼 그래서 우리는 여기서 DataFrame을 배우는 거니까, 다시 df로 넘어와서 발이야.
# DataFrame으로 돌린다음에 기초적인 분석을 배워봅시다

print(df.sum(), 'sum')               # 열 방향으로 합, 평균, 누적합 이런거 할 수 있음
print(df.mean(), 'mean')
print(df.cumsum(), 'cumsum')

print(df.describe())                 # 관련한 일련의 통계값을 쭉 뽑아내주는 describe() 매써드도 있음~

print("-" * 50, '구분좀')
# ---------------------------------------------------------------------------------------------------------------------
# Numpy가 갖고있는 유니버셜 함수는 대부분 DataFrame을 넣어도 계산을 해준다고 해
print(np.sqrt(df), 'np.sqrt')            # NaN이 들어간 부분은 뭐냐? 루트가 안 씌워지는 곳이야.
                                         # 뭐 음수가 있던 자리였겠지 뭐
                                         # 이런거를 오류 내성(fault tolerance)을 가진다고 말한대
print(np.sqrt(df).sum(), 'sqrt받고 sum')  # 불완전했던 것도 다 더함으로써 완전한 것 처럼 보이게 만들구 말이야.

import matplotlib.pyplot as plt
df.cumsum().plot(lw=2.0)     # DataFrame 받고, plot을 할 수도 있네.
# plt.show()                   # 책에 이렇게 써있다. "pandas는 DataFrame 객체에 알맞게 설계된 matplotlib 라이브러리 래퍼를 제공한다."
# Run 누를때마다 그림 떠서, 비활성화 ㄱㄱ

"""
DataFrame 클래스 plot 매써드 (인수 목록: x, y, subplot, sharex, sharey, use_index, stacked, sort_columns, title, grid...)

x               : 라벨/위치. 디폴트 None               -- 열 값이 x틱인 경우만 사용
y               : 라벨/위치. 디폴트 None               -- 열 값이 y틱인 경우만 사용
subplot         : 불리언. 디폴트 False                 -- 서브플롯 사용
sharex          : 불리언. 디폴트 True                  -- x축 공유
sharey          : 불리언. 디폴트 True                  -- y축 공유
use_index       : 불리언. 디폴트 True                  -- DataFrame.index를 x틱으로 사용
stacked         : 불리언. 디폴트 False                 -- 스택 플롯(바 플롯인 경우만)
sort_columns    : 불리언. 디폴트 False                 -- 그리기 전에 열을 알파벳 순으로 재정렬
title           : 불리언. 디폴트 None                  -- 플롯 제목
grid            : 불리언. 디폴트 False                 -- 수평 및 수직 그리드 선
legend          : 불리언. 디폴트 True                  -- 라벨 범례
ax              : matplotlib axis 객체               -- 플롯에 사용될 matplotlib axis 객체
style           : 문자열 또는 리스트/사전               -- 선 플롯 스타일
kind            : line/bar/barh/ked/density          -- 플롯 유형
logx            : 불리언. 디폴트 False                 -- x축 로그 스케일
logy            : 불리언. 디폴트 False                 -- y축 로그 스케일
xticks          : 순차 자료구조. 디폴트 Index           -- x축 틱 위치
yticks          : 순차 자료구조. 디폴트 Values          -- y축 틱 위치
xlim            : 튜플, 리스트                         -- x축 상하한
ylim            : 튜플, 리스트                         -- y축 상하한
rot             : 정수, 디폴트 None                    -- x축 틱 라벨 회전
secondary_y     : 불리언/순차 자료구조, 디폴트 False     -- 두 번째 y축
mark_right      : 불리언. 디폴트 False                 -- 두 번째 y축의 자동 라벨링
colormap        : 문자열/colormap 객체, 디폴트 None     -- 플롯에 사용될 colormap 
kwds            : 키워드                              -- matplotlib로 넘길 인수
"""
# 아따 많네;;;; 그래도 많이 쓰일 각인데 이건?
print("-" * 50, '구분좀')
# ---------------------------------------------------------------------------------------------------------------------


# 자자자 위에서 다룬 클래스는 많이 언급 됐듯이 pandas DataFrame 클래스였어
# 근데, pandas에는 Series 클래스도 있어. 예를 들어, DataFrame 객체에서 하나의 열만 선택하면 Series 객체를 얻는 거라고 함
from pandas import Series
print(type(df))
print(df[['No1']])                     # 파이참에선 대괄호를 두개 써야하네!! 암튼 뽑아낼 수 있습니다!
print(type(df[['No1']]))               # 암튼 이렇게 하나의 콜럼에 대해서 뽑으면 이것은 이제 Series 객체가 됩니다여

# 이 객체는 이제 앞에서와 마찬가지로 plot매써드를 사용할 수 있어
df[['No1']].cumsum().plot(style='r', lw=2, figsize=(7, 3))      # 여기 figsize= 인수도 넣을 수 있어!
plt.xlabel('date')
plt.ylabel('value')
# plt.show()                           # 아오 귀찮아. 비활성 ㄱㄱ!
print("-" * 50, '구분좀')
# ---------------------------------------------------------------------------------------------------------------------

# 이번에는 새롭게 GroupBy 연산을 들어가네?
# 이거는 pandas에서 제공하는 강력하고 유연한 그룹 지정 기능이라고 해.
# SQL의 그룹 지정기능이나, 엑셀 피봇 테이블과 비슷하다고 함.
# 일단은 함 봐바 위 DataFrame에다가 분기 데이터를 추가해보겠슴

df['Quater'] = ['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q2', 'Q3', 'Q3', 'Q3']
print(df)                              # 추가 됐고요.

# 이제 Quater 열을 잡고 GroupBy 연산을 때려보면

# groups = df.groupby('Quater')          # GroupBy라는 매써드로 기준을 잡아서, 그러한 객체를 하나 만들어내면
# print(groups.maen())
# print(groups.max())
# print(groups.size())                   # 뭐 이런 연산이 된다고 써있는데, 왜 안되지..... 파이참에서는 좀 다르게 돌려야 하나
                                         # 그냥 일단은 넘어가자! 시간 없어.







# print(groups.mean())
# print(groups.max())
# print(groups.size())





