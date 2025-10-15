import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from win32cryptcon import szOID_RSA_data

# ===== データの読み込み ======
retail = pd.read_excel('C:/Users/kazuki/Desktop/小売データ.xlsx')
weather = pd.read_csv('C:/Users/kazuki/Desktop/気象データ.csv',
                      encoding='cp932', skiprows=3)

# ===== データ結合と前処理 ======
weather['日付'] = pd.to_datetime(weather['年月日.1'], errors='coerce')
weather = weather.loc[:, ~weather.columns.str.contains(r'\.\d+$')]
weather = weather.drop(columns=['最深積雪(cm)', '平均雲量(10分比)', '年月日'])
weather = weather.drop(index=[0, 1]).reset_index(drop=True)

retail['日付'] = pd.to_datetime(retail['day'])
retail = retail.drop(columns=['day'])

merged = pd.merge(retail, weather, on='日付', how='left')

# 日付に基づく特徴量作成
merged['週'] = merged['日付'].dt.isocalendar().week.astype(int)
merged['月'] = merged['日付'].dt.month
merged['曜日'] = merged['日付'].dt.dayofweek  # 0:月曜, 6:日曜

# ===== 広告施策に対するエンコーディング =====
ad_cols = ['SNS', '売場施策', 'TV放映', 'プロモーション']
merged[ad_cols] = merged[ad_cols].fillna('なし')
binary_ad_cols = [col + '_有無' for col in ad_cols]
for original_col, new_col in zip(ad_cols, binary_ad_cols):
    merged[new_col] = (merged[original_col] != 'なし').astype(int)
data = merged.drop(columns=ad_cols)

# ===== 欠損値処理と外れ値処理 =====
num_cols = data.select_dtypes(include=[np.number]).columns
num_imputer = SimpleImputer(strategy='mean')
data[num_cols] = num_imputer.fit_transform(data[num_cols])

for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = np.clip(data[col], lower, upper)

# ===== 正規化（MinMax）======
exclude_cols = ['price', '売上数', '当店在庫手持週',
                'SNS_有無', '売場施策_有無', 'TV放映_有無', 'プロモーション_有無']
scale_cols = [col for col in num_cols if col not in exclude_cols]
new_scale_cols = scale_cols + ['週', '月', '曜日']

mm_scaler = MinMaxScaler()
data_mm = data.copy()
data_mm[new_scale_cols] = mm_scaler.fit_transform(data[new_scale_cols])

# ===== データの分割（時系列分割）=====
data_mm = data_mm.sort_values(by='日付').reset_index(drop=True)

# 訓練・テストデータの分割日を2024年の日付に設定
train_end_date = pd.to_datetime('2024-11-10')

# 訓練データ（〜2024年11月10日）
X_train = data_mm[data_mm['日付'] <= train_end_date].drop(columns=['売上数', '日付', 'name'])
y_train = data_mm[data_mm['日付'] <= train_end_date]['売上数']

# テストデータ（2024年11月11日以降）
X_test = data_mm[data_mm['日付'] > train_end_date].drop(columns=['売上数', '日付', 'name'])
y_test = data_mm[data_mm['日付'] > train_end_date]['売上数']

print(f"訓練データ数: {len(X_train)}")
print(f"テストデータ数: {len(X_test)}")

# ===== モデル学習と性能評価 =====
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'SVR': SVR(),
    'ANN': MLPRegressor(random_state=42, max_iter=500)
}

results = {}
for name, model in models.items():
    print(f"\n--- モデル学習: {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

results_df = pd.DataFrame(results).T
print("\n--- 性能評価結果 ---")
print(results_df)

# ===== 可視化・考察 =====
# 特徴量重要度（RandomForestRegressorの場合）
try:
    if 'RandomForestRegressor' in models:
        rf_model = models['RandomForestRegressor']
        feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(
            ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances.values, y=feature_importances.index)
        plt.title('Random Forest Feature Importance')
        plt.show()
except Exception as e:
    print(f"特徴量重要度の可視化中にエラーが発生しました: {e}")

# 実測 vs 予測のプロット（例としてRandomForestRegressor）
if 'RandomForestRegressor' in models:
    y_pred_rf = models['RandomForestRegressor'].predict(X_test)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Actual Sales')
    plt.plot(y_pred_rf, label='Predicted Sales')
    plt.title('Actual vs. Predicted Sales (Random Forest)')
    plt.xlabel('Data Point Index')
    plt.ylabel('Sales Volume')
    plt.legend()
    plt.show()
