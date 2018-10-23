## article : http://ddanggle.github.io/11lines


# 변수,	             변수에 관한 설명
# X	                각각의 행들이 트레이닝 샘플인 입력 데이터 행
# y	                각각의 행들이 트레이닝 샘플인 결과 데이터 행
# l0	              네트워크의 첫 번째 층. 입력 데이터값들을 특징화한다.
# l1	              네트워크의 두 번째 층. 보통 히든 레이어으로 알려져 있다.
# weight_vector	    weight들의 첫번째 레이어인 시냅스 0로 l0과 l1를 연결시킨다.
# *	                벡터의 원소별 곱셈(Elementwise multiplication). 같은 크기의 두 벡터의 상응하는 값들을 대응시켜 곱한 같은 크기의 벡터를 반환한다.
# -	                벡터의 원소별 뺄셈(Elementwise subtraction). 같은 크기의 두 벡터의 상응하는 값들을 대응시켜 뺀 벡터를 반환한다.
# x.dot(y)	        x, y가 벡터라면, 벡터의 내적(dot product)이다. 둘 다 행라면, 행의 곱이고, 만약 오직 하나만이 행라면, 벡터-행 곱이다.


#%%
import numpy as np # 1번째 줄

#sigmoid function
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
X = np.array([ [0,0,1], #10번째 줄 
                [0,1,1],
                [1,0,1], 
                [1,1,1]])


# 결과 데이터값
y = np.array([[0,0,1,1]]).T #16번째 줄 

# 계산을 위한 시드 설정
# 실험의 편의를 위해 항상 같은 값이 나오게 한다.
np.random.seed(1) #20번째 줄 

# weights를 랜덤적으로 mean of 0으로 초기화하자.
weight_vector = 2*np.random.random((3,1)) - 1 # 23번째 줄
# weight_vector 출력 예시
# array([[-0.5910955 ],
#        [ 0.75623487],
#        [-0.94522481]])

# 원본소스는 10000 회를 반복하지만 학습과정을 자세히 print 해보기 위해,
# 여기서는 10번만 합니다.
for iter in range(10): #25번째줄 

    # forwad propagation
    l0 = X # 25번째 줄 # 28번째 줄
    l1 = nonlin(np.dot(l0, weight_vector)) # 29번째 줄
    
    # l1 은 각 뉴런의 w 값을 모두 가지고 있는 배열입니다. 
    # (각 뉴런 : y = wx+b)
    print("l1 : ", l1)
    
    # 우리가 얼마나 놓쳤는가?
    l1_error = y - l1 # 32번째 줄 

    # 우리가 놓친 것들과 
    # 11 의 시그모이드 경사와 곱하기.
    l1_delta = l1_error * nonlin(l1, True) # 36번째 줄
    print("l1_delta :", l1_delta)

    learning_rate = 5

    # weight 업뎃
    weight_vector += np.dot(l0.T, l1_delta) # *learning_rate
    # 업데이트 전에 learning rate 를 곱해줘도 됩니다.
    # 일반적으로 과도한 update 가 되지 않도록 1 이하의 값을 곱해주지만,
    # 여기서는 학습을 가속하기 위해 5(1이상의 값)를 곱해도 됩니다.
    
    print('\r')

print ("Output After Traing")
print (l1)
