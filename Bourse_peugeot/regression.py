import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn import linear_model

from data_lib import *

DATASET_PATH = 'Peugeot.csv'


df = load_dataset(DATASET_PATH, separator=',')
df = df.drop(['Date'], axis=1)

trainset, testset = train_test_split(df, test_size=0.2, random_state=0)

X_train = trainset.drop('Adj Close', axis=1)
y_train = trainset['Adj Close']

X_test = testset.drop('Adj Close', axis=1)
y_test = testset['Adj Close']

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)


# The coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs
print(X_test)
plt.scatter(X_test['Close'], y_test,  color='black')
plt.plot(X_test['Close'], y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

