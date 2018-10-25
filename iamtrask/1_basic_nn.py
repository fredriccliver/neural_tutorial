## korean article : http://ddanggle.github.io/11lines
## original article : http://iamtrask.github.io/2015/07/12/basic-python-network/
## Title : "A Neural Network in 11 lines of Python (Part 1)"

# 변수,	             변수에 관한 설명
# X	                각각의 행들이 트레이닝 샘플인 입력 데이터 행
# y	                각각의 행들이 트레이닝 샘플인 결과 데이터 행
# l0	              네트워크의 첫 번째 층. 입력 데이터값들을 특징화한다. (l 은 layer 의 l, syn은 synapse 의 syn)
# l1	              네트워크의 두 번째 층. 보통 히든 레이어으로 알려져 있다.
# weight_vector	    weight들의 첫번째 레이어인 시냅스 0로 l0과 l1를 연결시킨다.
# *	                벡터의 원소별 곱셈(Elementwise multiplication). 같은 크기의 두 벡터의 상응하는 값들을 대응시켜 곱한 같은 크기의 벡터를 반환한다.
# -	                벡터의 원소별 뺄셈(Elementwise subtraction). 같은 크기의 두 벡터의 상응하는 값들을 대응시켜 뺀 벡터를 반환한다.
# x.dot(y)	        x, y가 벡터라면, 벡터의 내적(dot product)이다. 둘 다 행라면, 행의 곱이고, 만약 오직 하나만이 행라면, 벡터-행 곱이다.

#%% Data Definition
import numpy as np # 1번째 줄
from pprint import pprint

## sigmoid function 
## 숫자를 0 to 1 의 probability 값으로 변환하기 위해. 
## loss(w) = sig(w) 일 때, loss`(w) = sig(w)*(1-sig(w))
## 보통 f`(x), 즉 f(x) 도함수에서 값을 계산할 때, x 값을 집어넣어야 하지만,
## sigmoid 도함수의 특성상, x 값이 아니라, sig(x) 값으로 x에서의 미분 값을 구할 수 있다.
def nonlin(x, deriv=False): # 4번째 줄
  if (deriv==True): # 5번쨰 줄
    return x*(1-x)
  return 1/(1+np.exp(-x))

def relu(x, deriv=False):
  if (deriv==True):
    if(x>=0): return 1
    else: return 0
  else:
    if(x>=0): return x
    else: return 0


#input dataset
X = np.array([            #10번째 줄 
              [0,0,1], 
              [1,1,1], 
              [1,0,1], 
              [0,1,1], 
            ])
# pprint(X)

# 결과 데이터값 
y = np.array([[0,1,1,0]]).T # 16번째 줄 
# pprint(y)




#%% Initialize Weight
# np.random.seed(1) #20번째 줄 
# # 계산을 위한 시드 설정
# # 실험의 편의를 위해 항상 같은 값이 나오게 한다. (weight 를 랜덤으로 초기화하기 때문)
# # 하지만 초기 weight 값의 편차나 mixmax 따라 학습과정이 어떻게 다른지를 보려면 주석처리 한다.

# weights를 랜덤적으로 mean of 0으로 초기화하자.
feature_cnt = X.shape[1] ## w 는 feature 의 갯수만큼의 dimension을 갖습니다.
weight_vector = 2 * np.random.random((feature_cnt,1)) - 1 # 23번째 줄
w_ratio = 5 ## default = 1
weight_vector *= w_ratio # initial w 를 10배, 100배 해볼 수 있습니다.
print("initial W : ", weight_vector)

# # np.random.random((3,1)) 은 0~1 사이의 값 3개를 뽑아옵니다.
# # x2, -1 로 -1 < w < 1 에 확률적으로 평균은 0이 되도록 변환합니다.
# np.mean(2 * np.random.random((10000,1)) - 1 ) ## 평균은 0





##%% iteration
# 원본소스는 10000 회를 반복하지만 학습과정을 자세히 print 해보기 위해,
# 여기서는 횟수를 조정해봅니다.
err_history = [] # 각 회차마다의 에러값을 이 배열에 담아둡니다.

print_flag = 1  ## True(1) or False(0)
iterations = 10
learning_rate = 5
print("iterations : ", iterations)

for iter in range(iterations): #25번째줄 
    if(print_flag): 
      print('\r')
      print("--------- iteration", iter, " ---------")

    # "Full batch" vs "Mini Batch"
    # below is full batch learning

    # forwad propagation
    ## 학습할 데이터를 모두 가져옴. (full batch training)
    l0 = X # 28번째 줄 
    ## 현재 w(model)에 따른 prediction 값을 l1으로 가져옴.
    l1 = nonlin(np.dot(l0, weight_vector)) # 29번째 줄
    # 1 layer NN 이기 때문에, prediction 과정이라고 보면 됩니다. 

    # ## 
    # print("<< L0 >>")
    # pprint(l0)
    # print("\r")

    ##
    if(print_flag): 
      print("<< weight_vector >>")
      pprint(weight_vector)
      print("\r")
    
    # # l1 은 각 뉴런의 w 값을 모두 가지고 있는 배열입니다. 
    # # (각 뉴런 : y = wx+b)
    # print("l1 : ", l1)
    
    # 우리가 얼마나 놓쳤는가?
    ## 현재 예측 값과 실제 값의 차이를 계산.
    l1_error = y - l1 # 32번째 줄 
    err_sum = np.sum(np.abs(l1_error))
    if(print_flag): print("Total absolute error : ", err_sum , "\n")
    err_history.append(err_sum)

    # 우리가 놓친 것들과 
    # 11 의 시그모이드 경사와 곱하기.
    l1_delta = l1_error * nonlin(l1, True) # 36번째 줄
    
    # print("<< l1 >>")
    # print(l1)
    # print(nonlin(l1, True))
    # print("\r")

    

    # weight 업뎃
    weight_vector += np.dot(l0.T, l1_delta) * learning_rate
    # 업데이트 전에 learning rate 를 곱해줘도 됩니다.
    # 일반적으로 과도한 update 가 되지 않도록 1 이하의 값을 곱해주지만,
    # 여기서는 학습을 가속하기 위해 5(1이상의 값)를 곱해도 됩니다.
    
    

print ("Predictions After Traing", l1 )
# print(err_history)
print ("W :", weight_vector)


##%% Show error history
import matplotlib.pyplot as plt
plt.plot(np.abs(err_history))



##  바꿔볼 수 있는 hyper parameter 들.
# iterations, learning rate, w_ratio



### Ideal variable in trained model ###

# W : [
#   [10.83928706]
#   [-3.38957719]
#   [-3.54748419]
# ]

# Predictions After Traing [
#     [0.002800123]
#     [0.000970171]
#     [0.999319360]
#     [0.980195355]
# ]