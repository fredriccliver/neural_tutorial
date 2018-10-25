#%%
import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
            [1],
            [1],
            [0]])

np.random.seed(1)

# weights를 랜덤적으로 mean of 0으로 초기화하기
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

    #  레이어 0, 1, 2 로 키우기
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # 목표로 한 값에 비해 얼마나 놓쳤을까?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print ("에러: " + str(np.mean(np.abs(l2_error))))
        
    # 목표 값들의 방향은 무엇인가?
    # 정말 확신할 수 있나요? 그렇다면 많이 바꾸면 안된다. 
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # (웨이트에 따르면) 얼마나 각각의 l1 값들은 l2에러에 기여했을까? 
    l1_error = l2_delta.dot(syn1.T) # 43번째 줄
        
    # l1 목표의 방향은 무엇인가요?
    # 정말 확신할 수 있나요? 그렇다면 많이 바꾸면 안된다. 
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)