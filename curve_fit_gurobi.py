import pandas as pd
import gurobipy as gb
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(r'data.xlsx')
data = df.to_numpy()

n = data.shape[0]

x = data[:, 0]
x_sqr = list(map(lambda x: x * x, data[:, 0]))
y = data[:, 1]

m = gb.Model()

a, b, c, phi, y_hat = {}, {}, {}, {}, {}

a[0] = m.addVar(vtype=gb.GRB.CONTINUOUS, name="a_")
b[0] = m.addVar(vtype=gb.GRB.CONTINUOUS, name="b_")
c[0] = m.addVar(vtype=gb.GRB.CONTINUOUS, name="c_")

for i in range(n):
    phi[i] = m.addVar(vtype=gb.GRB.CONTINUOUS, name="phi%d" % (i))
    y_hat[i] = m.addVar(vtype=gb.GRB.CONTINUOUS, name="y_hat%d" % (i))
m.update()

for i in range(n):
    m.addConstr(phi[i] >= y_hat[i] - y[i])
    m.addConstr(phi[i] >= - y_hat[i] + y[i])
m.update()

for i in range(n):
    m.addConstr(y_hat[i] == a[0] * x_sqr[i] + b[0] * x[i] + c[0])
m.update()

m.setObjective(gb.quicksum(phi[i] for i in range(n)), gb.GRB.MINIMIZE)
m.update()

m.optimize()

A, B, C, Y_hat = m.getAttr('x', a), m.getAttr(
    'x', b), m.getAttr('x', c), m.getAttr('x', y_hat)
ABC = pd.DataFrame([{'a': A[0],
                     'b': B[0],
                     'c': C[0]
                     }])
YHAT = pd.DataFrame([Y_hat]).transpose()

dF = pd.DataFrame()
dF['x'] = data[:, 0]
dF['y'] = data[:, 1]
dF['y_hat'] = np.around(YHAT.to_numpy(), decimals=3)
dF['absolute error'] = np.around(np.abs(dF['y'] - dF['y_hat']), decimals=3)

print('\nObjecitve:', m.objval,
      '\n a:', A[0],
      '\n b:', B[0],
      '\n c:', C[0],
      '\n', dF)
def Plot(data, YHAT):
    plt.plot(data[:,0], data[:,1], color = 'b', marker='x', label = 'y (real)')
    plt.plot(data[:,0], YHAT.values, color = 'r',marker='o', label = 'y_hat (predicted)')
    plt.xlabel('X'), plt.ylabel('Y'), plt.title('Real Y vs predicted Y'), plt.legend(loc=4)
    return plt.show()
Plot(data, YHAT)

# with pd.ExcelWriter('gurobi output.xlsx') as writer:
#     ABC.to_excel(writer, sheet_name='abc')
#     YHAT.to_excel(writer, sheet_name='yhat')
# YHAT = YHAT.transpose()
# print(r2_score(YHAT.values, y))
# print('\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n','\n',)
