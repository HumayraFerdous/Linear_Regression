import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import shap

data = pd.read_csv("avocado.csv")
#print(data.head())
#print(data.dtypes)
data = data.drop(["Unnamed: 0","Date","4046","4225","4770"],axis = 1)
#print(data.dtypes)
#print(data.isnull().sum())
#print(data.duplicated())
data['total_volume_log']=np.log1p(data["Total Volume"])
data["Large_bags_ratio"] = data["Large Bags"]/data["Total Volume"]

features = ["type","region","year","total_volume_log","Large_bags_ratio"]
X = data[features]
y = data['AveragePrice']

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

numeric_features = ["total_volume_log","Large_bags_ratio","year"]
numeric_transformer = Pipeline([("scaler",StandardScaler())])

categorical_features  = ["type","region"]
categorical_transformer = Pipeline([("onehot",OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([("num",numeric_transformer,numeric_features),("cat",categorical_transformer,categorical_features)])

model = Pipeline([("preprocessor",preprocessor),("regressor",LinearRegression())])
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

R2_Score = r2_score(y_test,y_pred)
print("r squared score: ",R2_Score)
Mean_sq_er = np.sqrt(mean_squared_error(y_test,y_pred))
print("Mean squared error: ",Mean_sq_er)

pl.figure(figsize=(10, 6))
pl.scatter(y_test, y_pred, alpha=0.4)
pl.plot([0.5, 3.5], [0.5, 3.5], "r--")  # Perfect prediction line
pl.xlabel("Actual Price ($)")
pl.ylabel("Predicted Price ($)")
pl.title("Avocado Price Prediction Performance")
pl.show()

# Price trends by type
sns.boxplot(x="type", y="AveragePrice", data=data)
pl.title("Organic vs Conventional Avocado Prices")
pl.show()
coefs = model.named_steps["regressor"].coef_
num_features = numeric_features + list(model.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"].get_feature_names_out())
feature_importance = pd.DataFrame({"Feature": num_features,"Coefficient": coefs})
print(feature_importance)