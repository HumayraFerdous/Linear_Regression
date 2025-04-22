import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error,r2_score
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("insurance.csv")
#print(data.head())
#print(data.shape)
sns.lmplot(x='bmi',y='charges',data=data,aspect=2,height=6)
plt.xlabel('Body Mass Index as independent variable')
plt.ylabel('Insurance Charges: as Dependent variable')
plt.title('Charge Vs BMI')
plt.show()
f = plt.figure(figsize = (12,4))
ax = f.add_subplot(121)
sns.histplot(data['charges'],bins = 50,color='r',ax=ax)
ax.set_title('Distribution of insurance charges')

ax = f.add_subplot(122)
sns.histplot(np.log10(data['charges']),bins= 40, color = 'b',ax=ax)
ax.set_title('Distribution of insurance charges in $log$ scale')
ax.set_xscale('log')
plt.show()
f = plt.figure(figsize=(14,6))
ax = f.add_subplot(121)
sns.violinplot(x='sex', y='charges',data=data,palette='Wistia',ax=ax)
ax.set_title('Violin plot of Charges vs sex')

ax = f.add_subplot(122)
sns.violinplot(x='smoker', y='charges',data=data,palette='magma',ax=ax)
ax.set_title('Violin plot of Charges vs smoker')
plt.show()

plt.figure(figsize=(14,6))
sns.boxplot(x='children', y='charges',hue='sex',data=data,palette='rainbow')
plt.title('Box plot of charges vs children')
plt.show()

plt.figure(figsize=(14,6))
sns.violinplot(x='region', y='charges',hue='sex',data=data,palette='rainbow',split=True)
plt.title('Violin plot of charges vs children')
plt.show()

f = plt.figure(figsize=(14,6))
ax = f.add_subplot(121)
sns.scatterplot(x='age',y='charges',data=data,palette='magma',hue='smoker',ax=ax)
ax.set_title('Scatter plot of Charges vs age')

ax = f.add_subplot(122)
sns.scatterplot(x='bmi',y='charges',data=data,palette='viridis',hue='smoker')
ax.set_title('Scatter plot of Charges vs bmi')
plt.show()

data[['sex','smoker','region']] = data [['sex','smoker','region']].astype('category')
print(data.dtypes)
label = LabelEncoder()
label.fit(data.sex.drop_duplicates())
data.sex = label.transform(data.sex)
label.fit(data.smoker.drop_duplicates())
data.smoker = label.transform(data.smoker)
label.fit(data.region.drop_duplicates())
data.region = label.transform(data.region)
print(data.dtypes)

plt.figure(figsize = (10,6))
sns.heatmap(data.corr(),annot=True,cmap='cool')
plt.show()
X = data[['age','sex','bmi','children','smoker','region']]
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)
print(model.score(X_test,y_test))

y_pred=model.predict(X_test)
#print(y_pred)
mse_score=np.sqrt(mean_squared_error(y_pred,y_test))
print(mse_score)
r2 = r2_score(y_pred,y_test)
print(r2)

ridge = Ridge(alpha =0.5)
ridge.fit(X_train,y_train)
print(ridge.intercept_)
print(ridge.coef_)
print(ridge.score(X_test,y_test))
y_pred2=ridge.predict(X_test)
#print(y_pred)
mse_score=np.sqrt(mean_squared_error(y_pred2,y_test))
print(mse_score)
r2 = r2_score(y_pred2,y_test)
print(r2)

lasso = Lasso(alpha=0.2, fit_intercept=True, precompute=False, max_iter=1000,
              tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
lasso.fit(X_train, y_train)
print(lasso.intercept_)
print(lasso.coef_)
print(lasso.score(X_test, y_test))
y_pred3=lasso.predict(X_test)
#print(y_pred)
mse_score=np.sqrt(mean_squared_error(y_pred3,y_test))
print(mse_score)
r2 = r2_score(y_pred3,y_test)
print(r2)

figure, (ax1,ax2,ax3) = plt.subplots(1, 3, sharex=True,figsize=(20, 10))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4,ax=ax1)
#ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  # Perfect prediction line
ax1.set_xlabel("Actual Charge")
ax1.set_ylabel("Predicted Charge")
ax1.set_title("Medical Insurance Prediction by Linear Regression")

sns.scatterplot(x=y_test, y=y_pred2, alpha=0.4,ax=ax2)
#ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  # Perfect prediction line
ax2.set_xlabel("Actual Charge")
ax2.set_title("Medical Insurance Prediction by Ridge")

sns.scatterplot(x=y_test, y=y_pred3, alpha=0.4,ax=ax3)
#ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  # Perfect prediction line
ax3.set_xlabel("Actual Charge")
ax3.set_title("Medical Insurance Prediction by Lasso")
plt.show()
