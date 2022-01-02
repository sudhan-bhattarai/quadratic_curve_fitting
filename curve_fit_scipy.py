import pandas as pd
from scipy import optimize
from matplotlib import pyplot as plt

df = pd.read_excel(r'data.xlsx')
data = df.to_numpy()

def func(x, a, b, c):
    return a*x*x + b*x + c

popt, pcov = optimize.curve_fit(func, data[:,0], data[:,1])
print('a, b , and c are:', popt, '\n')
print('predicted y values are:', func(data[:,0], popt[0], popt[1], popt[2]))
Y_hat = func(data[:,0], popt[0], popt[1], popt[2])

def Plot_ (data, y_hat):
    plt.plot(data[:,0], data[:,1], color = 'b', marker='x', label = 'y')
    plt.plot(data[:,0], Y_hat, color = 'r',marker='o', label = 'y_hat')
    plt.legend(loc=4)
    plt.show()
