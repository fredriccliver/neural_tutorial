#%%
import os, sys
from pprint import pprint
import pandas as pd
import numpy
from sklearn.preprocessing import LabelBinarizer
root_path = os.path.dirname(os.path.abspath('__file__'))
data_path = root_path + "/data/titanic/"
sys.path.append(root_path)

from DNN import DNN
from random import seed
seed(1)

dnn = DNN()

dataset = pd.read_csv(data_path + "train.csv")
# dataset = dnn.load_csv(data_path + "train.csv")



dataset = dataset[['Age', 'SibSp', 'Pclass', 'Parch', 'Fare', 'Survived']]
dataset = dataset.dropna(axis=0)
dataset = dataset.values.tolist()
print(dataset)
dnn.str_column_to_int(dataset, len(dataset[0])-1)



#%%
# normalize input variables
dnn.minmax = dnn.dataset_minmax(dataset)
dnn.normalize_dataset(dataset, dnn.minmax)
# evaluate algorithm
n_folds = 2         # how many seperates data for validation
l_rate = 0.1        # learning rate
n_epoch = 50        # learn repeat
n_hidden = 10       # neuron count (hidden)

back_propagation = dnn.back_propagation
dnn.activation_function = 'sigmoid'
scores = dnn.evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))









#%% make hyperparameter history
l_rate = .01
dnn.activation_function = 'sigmoid'
hyperparam_hist = []
for n_epoch in (20, 50, 70, 100, 150, 300):
    for n_hidden in (5, 10, 15):
        scores = dnn.evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        hyperparam_hist.append([n_epoch, n_hidden, (sum(scores)/float(len(scores)))])
        print("n_epoch, n_hidden ", n_epoch, n_hidden)

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
ax.set_xlabel('n_epoch')
ax.set_ylabel('n_hidden')
ax.set_zlabel('accuracy')
plt.show()

fig = plt.figure()
bx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = bx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
bx.set_xlabel('n_epoch')
bx.set_ylabel('n_hidden')
bx.set_zlabel('accuracy')
plt.show()

fig = plt.figure()
cx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = cx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
cx.set_xlabel('n_epoch')
cx.set_ylabel('n_hidden')
cx.set_zlabel('accuracy')
cx.view_init(azim=15)
plt.show()

fig = plt.figure()
cx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = cx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
cx.set_xlabel('n_epoch')
cx.set_ylabel('n_hidden')
cx.set_zlabel('accuracy')
cx.view_init(azim=-15)
plt.show()


