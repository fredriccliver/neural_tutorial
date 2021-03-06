
#%% import 및 data load
import os, sys, copy
root_path = os.path.dirname(os.path.abspath('__file__'))
# work_path = root_path + "/neural_tutorial"
data_path = root_path + "/data"
sys.path.append(root_path)



from CustomDNN.DNN import DNN
from random import seed
from pprint import pprint

seed(1)

dnn = DNN()
file_path = data_path + "/seeds_dataset.csv"

# pandas에서는 DataFrame 타입으로 data 를 처리하지면 여기서는 일반 python list 형태로 데이터를 처리합니다.
original_dataset = dnn.load_csv(file_path)
# pprint(dataset)


#%%
dataset = copy.deepcopy(original_dataset)           # To avoid copy by reference
for i in range(len(dataset[0])-1):
    dnn.str_column_to_float(dataset, i)
# convert class column to integers
dnn.str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
dnn.minmax = dnn.dataset_minmax(dataset)
dnn.normalize_dataset(dataset, dnn.minmax)
# evaluate algorithm
n_folds = 2         # how many seperates data for validation
l_rate = 0.001        # learning rate
n_epoch = 50        # learn repeat
n_hidden = 100       # neuron count (hidden)

back_propagation = dnn.back_propagation
scores = dnn.evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))




#%% hyper parameter test
l_rate = .1
hyperparam_label = ["n_epoch", "n_hidden"]
dnn.activation_function = 'sigmoid'
# elu, l_rate .0001 은 편차 높음.
# sigmoid, l_rate .1 은 편차가 낮음.
hyperparam_hist = []
for n_epoch in range(5, 40, 5):
    for n_hidden in range(5, 40, 5):
        scores = dnn.evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        hyperparam_hist.append([n_epoch, n_hidden, (sum(scores)/float(len(scores)))])
        print("n_epoch, n_hidden ", n_epoch, n_hidden)

print("hyperparam_hist : ", hyperparam_hist)


#%% l_rate 위주로 테스트
# n_hidden은 10으로 고정.
n_hidden = 10
hyperparam_label = ["n_epoch", "l_rate"]
dnn.activation_function = 'sigmoid'
hyperparam_hist = []
for n_epoch in range(5, 50, 5):
    for l_rate in [0.1 ,0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
        scores = dnn.evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        hyperparam_hist.append([n_epoch, l_rate, (sum(scores)/float(len(scores)))])
        print("n_epoch, l_rate ", n_epoch, l_rate)

print("hyperparam_hist : ", hyperparam_hist)




#%% hyperparameter-accuracy plotting
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure()
title = "l_rate:"+str(l_rate)
fig.suptitle(title)

ax = fig.add_subplot(111, projection='3d')
for i in range(0, len(hyperparam_hist)):
    ax.scatter(xs=hyperparam_hist[i][0], ys=hyperparam_hist[i][1], zs=hyperparam_hist[i][2], c='r', marker="o")
ax.set_xlabel(hyperparam_label[0])
ax.set_ylabel(hyperparam_label[1])
ax.set_zlabel('accuracy')
plt.show()

fig = plt.figure()
bx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = bx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
bx.set_xlabel(hyperparam_label[0])
bx.set_ylabel(hyperparam_label[1])
bx.set_zlabel('accuracy')
plt.show()

fig = plt.figure()
cx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = cx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
cx.set_xlabel(hyperparam_label[0])
cx.set_ylabel(hyperparam_label[1])
cx.set_zlabel('accuracy')
cx.view_init(azim=15)
plt.show()

fig = plt.figure()
cx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = cx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
cx.set_xlabel(hyperparam_label[0])
cx.set_ylabel(hyperparam_label[1])
cx.set_zlabel('accuracy')
cx.view_init(azim=-15)
plt.show()


