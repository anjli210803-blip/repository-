import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
df=pd.read_excel('salary_data.xlsx')
print(df.head())
plt.scatter(df['experience'],df['salary']
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()
df.isna().sum()
#create object 
LR=LinearRegression ()
#assign target values to x
x=df.iloc[:,:1]
x.head()
#assign target values to y
y=df('salary')
y.head(2)
#train the model 
LR.fit(x,y)
test=([[1.5],[2.6],[5.8],[11],[13])
y_pred=LR.predict(test)
y_pred
plt.scatter(x,y)
plt.scatter(x,y)
plt.plot(test,y_pred,color='r')
LR.score(x,y)
