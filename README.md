import pandas as pd

retail = pd.read_excel('C:/Users/23X4002/Desktop/3年生秋/PBL/小売データ.xlsx')
weather= pd.read_csv('C:/Users/23X4002/Desktop/3年生秋/PBL/気象データ.csv',
                     encoding='cp932', skiprows=3)

weather['日付'] = pd.to_datetime(weather['年月日.1'], errors='coerce')
weather = weather.loc[:, ~weather.columns.str.contains(r'\.\d+$')]
weather = weather.drop(columns=['最深積雪(cm)', '平均雲量(10分比)', '年月日'])
weather = weather.drop(index=[0, 1]).reset_index(drop=True)

retail['日付'] = pd.to_datetime(retail['day'])
retail = retail.drop(columns=['day'])

print(retail.head())
print(weather.head())

# weather を retail の行数に合わせて繰り返す
weather_expanded = retail[['日付']].merge(weather, on='日付', how='left')
print(weather_expanded.head())
print(weather_expanded.shape)

merged = pd.merge(retail, weather, on='日付', how='left')
print(merged.head(20))


