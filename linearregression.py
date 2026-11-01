import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
db = pd.read_csv('Salary_Data.csv')
print(db.head())
x_train, x_test, y_train, y_test = train_test_split(
    db[['YearsExperience']],
    db['Salary'],
    test_size=0.2,
    random_state=0
)
linearmodel = LinearRegression()
linearmodel.fit(x_train, y_train)
y_pred_test = linearmodel.predict(x_test)
y_pred_train = linearmodel.predict(x_train)
plt.scatter(x_train, y_train, color='red', label='Actual Salary')
plt.plot(x_train, y_pred_train, color='blue', label='Regression Line')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(loc='best')
plt.show()
r_sq1 = linearmodel.score(x_train, y_train)
print('Training R-squared:', r_sq1)
y_pred1 = linearmodel.predict(pd.DataFrame({'YearsExperience': [1.1]}))
print('The salary for 1.1 years experience is:', y_pred1[0])