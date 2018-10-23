## 점프 투 파이썬 : https://wikidocs.net/28
## 파이썬에 관해서는 점프 투 파이썬을 보시면 좋습니다.

class CustomClass:

    def __init__(self):
        return 

    def plus(self, x, y):
        return x+y

    def minus(self, x, y):
        return x-y

class Mathmatics:

    def square(self, below, up, debug=False):
        result = below
        if(debug==True): print("debug : ", below, up)
        for i in range(1, up):
            result = result * below
        return result

def introduce():
    return "This is test class"