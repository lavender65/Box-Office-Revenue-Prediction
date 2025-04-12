'''Movie Box Office Revenue Prediction'''
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("cost_revenue_clean.csv")
# print(data.head())
# print(data.describe())

X = DataFrame(data,columns = ['production_budget_usd'])  #prod budget
y = DataFrame(data,columns = ['worldwide_gross_usd'])    #gross revenue

#Visualize 
plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
# plt.show()

#Linear Regression
regression = LinearRegression() #create obj for LR
regression.fit(X,y)
print(regression.coef_) #slope coefficient theta_1
print(regression.intercept_) #intercept coefficient theta_0
print(regression.score(X, y))


plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.3)

plt.plot(X,regression.predict(X), color = 'red', linewidth = 4)  #plot the LR on the previous graph

plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()


