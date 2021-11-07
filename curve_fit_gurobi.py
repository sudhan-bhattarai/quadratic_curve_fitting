import pandas as pd
import gurobipy as gb

df = pd.read_excel(r'data.xlsx')
data = df.to_numpy()
n = data.shape[0]

m = gb.Model()

phi = {}

a = m.addVar(vtype= gb.GRB.CONTINUOUS, name = "a")
b = m.addVar(vtype= gb.GRB.CONTINUOUS, name = "b")
c = m.addVar(vtype= gb.GRB.CONTINUOUS, name = "c")

for i in range(n):
    phi[i] = m.addVar(vtype=gb.GRB.CONTINUOUS, name="phi%d" %(i))
m.update()
