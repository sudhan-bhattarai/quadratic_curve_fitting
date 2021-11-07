# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:38:14 2021

@author: msaboga
"""
#Packages
from pyomo.environ import*
solver = SolverFactory('gurobi')

model = ConcreteModel(name = 'curve')

model.J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

x = {1:0, 2:0.5, 3:1, 4:1.5, 5:1.9, 6:2.5, 7:3, 8:3.5, 9:4, 10:4.5, 11:5, 12:5.5, 13:6, 14:6.6, 15:7, 16:7.6, 17:8.5, 18:9, 19:10}
y = {1:1, 2:0.9, 3:0.7, 4:1.5,5:2,6:2.4,7:3.2,8:2,9:2.7,10:3.5,11:1,12:4,13:3.6,14:2.7,15:5.7,16:4.6,17:6,18:6.8,19:7.3}
          
                       
model.a = Var() 
model.b = Var() 
model.c = Var()
model.yc = Var(model.J) 
model.phi = Var(model.J)  

def obj_rule(model):
    return (sum(model.phi[j] for j in model.J))
model.obj = Objective(rule=obj_rule, sense=minimize)

@model.Constraint(model.J)
def st1(model,j):
    return model.phi[j] >= model.yc[j]-y[j] 

@model.Constraint(model.J)
def st2(model,j):
    return model.phi[j] >= -(model.yc[j]-y[j]) 

@model.Constraint(model.J)
def st3(model,j):
    return model.yc[j] == (model.a*x[j]*x[j])+(model.b*x[j])+model.c

solver.solve(model)
print ("Objective Value " + str(value(model.obj)))
print("a " + str(value(model.a)))
print("b " + str(value(model.b)))
print("c " + str(value(model.c)))
print("y predicted ")
for j in model.J: 
    print(value(model.yc[j]))
