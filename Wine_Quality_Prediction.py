import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("winequality-red.csv")
#print(data.head())
#print(data.dtypes)
print(data.isnull().sum()) #data is already cleaned
data['acid_balance'] =data['fixed acidity']/data['volatile acidity']
data['sulfur_ratio'] = data['free sulfur dioxide']/data['total sulfur dioxide']
features = ['residual sugar','citric acid','sulphates','alcohol','acid_balance','sulfur_ratio']
pl.figure(figsize=(15,5))
pl.suptitle('Before Outlier Handling',y=1.02)
for i, col in enumerate(features,1):
    pl.subplot(1,len(features),i)
    sns.boxplot(y=data[col])
pl.tight_layout()
pl.show()

for col in features:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q1 + 1.5 * IQR

    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
pl.figure(figsize=(15,5))
pl.suptitle('Before Outlier Handling',y=1.02)
for i, col in enumerate(features,1):
    pl.subplot(1,len(features),i)
    sns.boxplot(y=data[col])
pl.tight_layout()
pl.show()
X = data[features]
y = data['quality']

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)

R2_Score = r2_score(y_test,y_pred)
print("r squared score: ",R2_Score)
Mean_sq_er = np.sqrt(mean_squared_error(y_test,y_pred))
data2 = data
print("Mean squared error: ",Mean_sq_er)
pl.figure(figsize=(12,8))
sns.heatmap(data2.drop('quality', axis=1).corr(method='pearson'), annot=True, cmap='Blues')
pl.title('Correlation heatmap', fontsize=18)
pl.tight_layout()
pl.show()

importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
print("\nFeature Importance:\n", importance.sort_values('Coefficient', ascending=False))

pl.figure(figsize=(10, 6))
pl.scatter(y_test, y_pred, alpha=0.5)
pl.plot([3, 8], [3, 8], 'r--')  # Perfect prediction line
pl.xlabel("Actual Quality")
pl.ylabel("Predicted Quality")
pl.title("Wine Quality Prediction")
pl.show()
