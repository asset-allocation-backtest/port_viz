# 필요한 모듈 불러놓고 시작하세
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

path = r'C:\Users\GD Park\iCloudDrive\gdpre_PycharmProjects\Python for Finance'
name = 'chapter9. 수학'
folder = path + '/' + name + '/'
os.mkdir(folder)
os.chdir(folder)

def f(x):
    return np.sin(x) + 0.5 * x

x = np.linspace(-2*np.pi, 2*np.pi, 50)
plt.plot(x, f(x), 'b')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('1.png')
########################################################################################################################
# 먼저 볼 것은 간단하게 해보는 회귀!
# 이전에도 한번 써본 것이지만, numpy의 polyfit이라는 함수가 최적의 파라미터를 찾아주는 역할 하는 애가 있고
# 입력값에 대한 근차값을 계산해주는 polyval이라는 함수가 있음!

# 근데 근차값이 뭐야?

reg = np.polyfit(x, f(x), deg=1)
ry = np.polyval(reg, x)

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('2.png')
plt.plot(x, reg[1]+ x*reg[0], 'k.')             # 아항 polyval이라는 애는 이렇게 하는 것 보다 좀 더 쉽게 회기값으로 그려주는 것을 말하는 거였네~
########################################################################################################################
# 근데 사인이 들어가있는 함수를 저렇게 회귀하는 것은 좀 무리가 있으니깐,
# 회귀를 5차식까지 올려버려서, 5차함수로 회귀를 하자는 것을 해보자.
reg = np.polyfit(x, f(x), deg=5)
ry = np.polyval(reg, x)
plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('3.png')                # 이전보다는 설명력이 좋은 회귀함수를 만들어낼 수 있네
########################################################################################################################
reg = np.polyfit(x, f(x), deg=7)
ry = np.polyval(reg, x)
plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('4.png')            # 7차로 올리면 더 비슷하게 따라갈 수 있고!

np.allclose(f(x), ry)           # allclose함수는 두 ndarray가 같은지를 확인하는 함수이고, -> 회귀로 y를 끌어낸 거랑 같을리가 없겠지. 그래서 False 반환해야지
np.sum((f(x) - ry) ** 2) / len(x) # 각 x에서 그 오차들을 제곱해서 다 더한다음에 갯수로 나누면, MSE(Mean Square Error)를 이렇게 확인해볼 수 있음
########################################################################################################################
# 회귀를 꼭 테일러 expansion처럼 x^n의 선형합으로 되어있는 형태로 해야하냐?
# 그건 아닐 수 있잖아?? 예를 들어 하나의 원소를 sin(x)로 둬야 해답이 나온다는 사실을 알때!! 그런 회귀는 어떻게 하느냐

# 그러면 우선 위에서 했던 방법으로 3차식 회귀를 보고가자
reg = np.polyfit(x, f(x), deg=3)
ry = np.polyval(reg, x)
plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('5.png')        # 저 방법으로 3차 회귀는 이렇게 하면 땡이었었지
########################################################################################################################
# 다른 방법으로 똑같이 3차 회귀를 하는 방법을 알아야 해 우선,
# 그래서 위에서 저렇게 하고,
# 다르게 하는 방법을 해봐서 비교를 해보자! (그럴라고 3차식 회귀를 일단 보고 지나온 것)

x = np.linspace(-2*np.pi, 2*np.pi, 50)

matrix = np.zeros((3+1, len(x)))
matrix[3, :] = x ** 3
matrix[2, :] = x ** 2
matrix[1, :] = x
matrix[0, :] = 1                    # 3차 x벡터들을 만들고

# 이제 regression을 어떻게 들어가냐면,
# numpy의 linalg에서 최소자승문제를 풀어주는 lstsq 함수를 제공해 그걸 이용
# 이 함수는 연립 선형방정식을 넣는거고 그리고나서 4개의 값을 반환하는데,
# 1.x(최소자승문제 답), 2.resid(잔차제곱합), 3.rank(랭크), 4.s(특이값(singular value))
reg = np.linalg.lstsq(matrix.T, f(x))[0]    # 지금은 최소자승의 답만 필요하니깐, 첫번째 것만 뽑아내고
ry = reg @ matrix
plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('6.png')        # 똑같이 나오네!
########################################################################################################################
# 그러면 최고차를 sin(x)로 바꾸는 걸 해보자!
x = np.linspace(-2*np.pi, 2*np.pi, 50)

matrix = np.zeros((3+1, len(x)))
matrix[3, :] = np.sin(x)
matrix[2, :] = x ** 2
matrix[1, :] = x
matrix[0, :] = 1

reg = np.linalg.lstsq(matrix.T, f(x))[0]
ry = reg @ matrix
plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, ry, 'r.', label='regression')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('7.png')        # 우왕굳

np.allclose(f(x), ry)       # True ... ㄷㄷㄷ
np.sum((f(x) - ry) ** 2) / len(x)
########################################################################################################################
# 이번엔 다차원을 한 번 해볼까?
def fm(x, y):
    return np.sin(x) + 0.25 * x + np.sqrt(y) + 0.05 * y ** 2

x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)

Z = fm(X, Y)
x = X.flatten()         # 1행, 2행, 3행 ~~ 쭈욱 있는 것들을 -> 그냥 1행으로 겹겹이 계속 열을 늘려가면서 1차원으로 만들어주는 함수가 Flatten()
y = Y.flatten()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2, cmap=plt.cm.coolwarm, cstride=2, linewidth=0.5, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('8.png')        # 이 놈을 이제부터 회기해볼 것이여

matrix = np.zeros((len(x), 6+1))
matrix[:, 6] = np.sqrt(y)
matrix[:, 5] = np.sin(x)
matrix[:, 4] = y ** 2
matrix[:, 3] = x ** 2
matrix[:, 2] = y
matrix[:, 1] = x
matrix[:, 0] = 1


import statsmodels.api as sm
model = sm.OLS(fm(x, y), matrix).fit()          # OLS는 클래스 객체를 반환하기 때문에 매써드를 한 번 더 먹여서 원하는 답으로 향해야함
model.summary() # 를 하면 summary를 볼 수 있고
# 선형회귀 답을 꺼내는 것은 params로 꺼낼 수 있대~
a = model.params

def reg_func(a, x, y):
    f6 = a[6] * np.sqrt(y)
    f5 = a[5] * np.sin(x)
    f4 = a[4] * y ** 2
    f3 = a[3] * x ** 2
    f2 = a[2] * y
    f1 = a[1] * x
    f0 = a[0] * 1
    return (f6 + f5 + f4 + f3 + f2 + f1 + f0)

RZ = reg_func(a, X, Y)                  # 위에 계산을 각 열벡터마다 해줘야하니깐, 함수를 만들어서 하는게 최선인것 같기도..?
fig = plt.figure(figsize=(10, 6))
ax = fig.gca(projection='3d')
surf1 = ax.plot_surface(X, Y, Z, rstride=2, cmap=plt.cm.coolwarm, cstride=2, linewidth=0.5, antialiased=True)
surf2 = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, label='regression')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
fig.colorbar(surf, shrink=0.5, aspect=5)
########################################################################################################################
# 보간법(interpolation) : 관측점이 주어졌을 때 두 개의 이웃하는 관측점 사이의 자료를 계산하는 보감 함수를 만드는 것.
# 근데 잇더라도 최대한 부드럽게 잇고싶단 말이지. -> 함수도 연속, 도함수도 연속, 2계도함수도 연속 이런 조건의 함수로 잇겠다는 말이 됨
# 그런 함수를 Cubic Spline함수라고 함.... 이얘기를 왜하냐면, 이용할 함수 이름이 그따구니깐... '아~ 부드럽게 하겠구나~'정도라도 받아들일 수 있도록

# 회귀랑 차이가 있다면, 회귀는 주어진 데이터가 정확하지 않다고 가정하고 -> 그 데이터를 대표할 수 있는 Sth을 끄집어 내는 작업이라면
# 보간은 주어진 데이터가 정확하다고 가정하고 -> 틈틈이 빈 공간을 채워넣는 작업을 하는 것

# 그 보간을 하는데 사용할 Scipy 패키지에서 interpolation.splrep와 interpolation.splev를 쓸 것.
# splrep는 보간 모형을 생성하고 -> 이 모형을 이용해 새로운 x에 대해서 y값을 계산해내는 것이 splev
import scipy.interpolate as spi
x = np.linspace(-2*np.pi, 2*np.pi, 25)
def f(x):
    return np.sin(x) + 0.5 * x



ipo = spi.splrep(x, f(x), k=1)      # k=1 명령으로 1차보간을 한다는 소리이고, 주어진 x와 f(x)에서 빈칸빈칸을 1차로 보간한 모형을 만들어서
iy = spi.splev(x, ipo)              # 새로운 x 구간에서 위의 모형으로 보간한 것들을 뽑아내지만... 기존것으로 뭘 보간을 해;;; 그냥 똑같이 나와야겠지

plt.plot(x, f(x), 'b', label='f(x)')
plt.plot(x, iy, 'r.', label='interpolation')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')                  # 똑같은 걸로 했으니깐, 똑같이 점 찍어야지 뭐
# 그래서 이게 잘 된것 같아 보이지만.... 하지만, 조금만 더 현미경으로 한 부분 안쪽으로 들어간다 치면
########################################################################################################################
xd = np.linspace(1.0, 3.0, 50)
iyd = spi.splev(xd, ipo)
plt.plot(xd, f(xd), 'b', label='f(x)')
plt.plot(xd, iyd, 'r.', label='interpolation')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')      # 쫌... 오차가 생기는 것을 볼 수 있음.
                        # 그래서 차수를 좀 더 올려서, 1차 함수로 보간하는게 아닌 3차함수로 보간을 한다 해보자

ipo = spi.splrep(x, f(x), k=3)      # 주어진 x와 f(x)에서 빈칸빈칸을 3차로 보간한 모형을 만들어서
iyd = spi.splev(xd, ipo)            # 새로운 xd 구간에서 위의 모형으로 보간한 것들을 뽑아내는

plt.plot(xd, f(xd), 'b', label='f(x)')
plt.plot(xd, iyd, 'r.', label='interpolation')
plt.legend()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')          # 훨씬 더 잘 따라감
"""
스플라인 보간법을 적용하면 최소자승 회귀법보다 더 정확한 근사 결과를 얻을 수 있다.
그러나, 보간법을 사용하려면 자료가 정렬되어 있어야 하고, 잡읍이 없어야하며, 다차원 문제에는 적용할 수 없다.
또한, 계산량이 더 많기 때문에 어떤 경우에는 회귀법보다 훨씬 계산 시간이 오래 걸릴 수 있다.
"""
########################################################################################################################
# 최적화. -> 그 중에서도 제약조건이 있는 최적화
# 쉽게 말하면, budget line이라는 제약조건하에서 -> 효용함수를 최대화하는 optimazation은? 이런거...!!!
import scipy.optimize as spo
from math import sqrt
def Eu(s, b):
    return -(0.5 * sqrt(s*15 + b*5) + 0.5*sqrt(s*5 + b*12))

# 제약조건
cons = {'type':'ineq', 'fun': lambda s,b: 100 - s*10 - b*10}
# 예산제한
bnds = ((0, 1000), (0, 1000))

result = spo.minimize(Eu, [5, 5], method='SLSQP', bounds=bnds, constraints=cons)
result