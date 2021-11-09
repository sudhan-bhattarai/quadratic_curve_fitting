import pandas as pd
import gurobipy as gb
from sklearn.metrics import r2_score

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
# solutions =
print('\na, b, c:', A[0], B[0], C[0], '\nY_hat:', Y_hat)
print('\nObjecitve:', m.objval)
ABC = pd.DataFrame([{'a': A[0],
                     'b': B[0],
                     'c': C[0]
                     }])
YHAT = pd.DataFrame([Y_hat])
# with pd.ExcelWriter('gurobi output.xlsx') as writer:
#     ABC.to_excel(writer, sheet_name='abc')
#     YHAT.to_excel(writer, sheet_name='yhat')
YHAT = YHAT.transpose()
print(r2_score(YHAT.values, y))
