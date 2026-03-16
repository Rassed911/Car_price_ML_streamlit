import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

url = 'https://raw.github.com/stedy/Machine-Learning-with-R-datasets/master/usedcars.csv'
df = pd.read_csv(url)
df.info()

df_encoded = pd.get_dummies(df, columns = ['model', 'color', 'transmission'])
y = df_encoded['price']
X = df_encoded.drop(columns = ['price'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Средняя ошибка: {mae:.2f} $')
