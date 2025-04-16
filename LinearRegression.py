import pandas as pd
import matplotlib.pyplot as plt
import LinearRegression

data = pd.read_csv('AI_datasheet.csv')

x = data[['Math']].values
y = data['GPA'].values

model = LinearRegression()
model.fit(x, y)

plt.scatter(x, y)
plt.plot(x, model.predict(x), color='red')
plt.xlabel('Math')
plt.ylabel('GPA')
plt.title('Linear Regression on Math vs GPA')
plt.show()
