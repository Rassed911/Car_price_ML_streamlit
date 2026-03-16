import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# 1. ЗАГОЛОВОК САЙТА
st.title("🏥 Калькулятор стоимости авто")
st.write("Введите свои данные, чтобы узнать примерную стоимость авто.")

# 2. ЗАГРУЗКА И ОБУЧЕНИЕ 
@st.cache_data
def load_and_train():
    url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/usedcars.csv'
    df = pd.read_csv(url)
    df.info()

    df_encoded = pd.get_dummies(df, columns = ['model', 'color', 'transmission'])   
    y = df_encoded['price']
    X = df_encoded.drop(columns = ['price'])
    model = RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X,y)
    return model, X.columns

model,cols = load_and_train()

# 3. ИНТЕРФЕЙС (Боковая панель со слайдерами)
st.sidebar.header("Параметры авто")
year = st.sidebar.slider('Год выпуска',2000,2016,2010)
mileage = st.sidebar.slider('Пробег',0,200000, 50000)
color = st.sidebar.selectbox('Цвет',['Yellow','Gray','Silver', 'White', 'Black', 'Red','Blue', 'Green'] )
model_type = st.sidebar.selectbox('Модель автомобиля', ['SE', 'SEL', 'SES'])
trans_type = st.sidebar.selectbox('Коробка передач', ['AUTO', 'MANUAL'])

# 4. ПОДГОТОВКА ДАННЫХ ДЛЯ ПРОГНОЗА
input_df = pd.DataFrame(0, index=[0], columns=cols)

input_df['year'] = year
input_df['mileage'] = mileage
input_df[f'color_{color}'] = 1
input_df[f'model_{model_type}'] = 1
input_df[f'transmission_{trans_type}'] = 1
if st.button('Узнать цену'):
    prediction = model.predict(input_df)
    st.success(f"### Оценочная стоимость: ${prediction[0]:,.2f}")

