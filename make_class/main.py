#%%
import os, sys
class_path = os.path.dirname(os.path.abspath('__file__'))+"/make_class/"
class_path
# '/Users/user/Projects/neural_tutorial'
#%%
sys.path.insert(1, class_path)
print('module path was inserted.')

#%%
import CustomClass as cc
print(cc.introduce())

#%%
from CustomClass import CustomClass as cc
mycc = cc()     ## jupyter 에서는 print 없이 출력가능
mycc.plus(1, 1)

#%%
from CustomClass import Mathmatics as math
math().square(2,2, debug=True)
