#!/usr/bin/env python
# coding: utf-8

# # 時系列データ予測テンプレート (Time Series Forecasting Template) - Ultimate Edition
# 
# このノートブックは、2025年のデータサイエンス業界標準に基づいた時系列予測の決定版テンプレートです。
# 基本的なフローに加え、高度な欠損値補完、不均衡データ対策、**包括的なPandasデータ加工**、**テキストデータ処理**、**多様な機械学習モデル**、そして**SHAPによるモデル解釈**の実装例を含んでいます。
# 
# ## 目次
# 1. [設定とライブラリのインポート](#1.-設定とライブラリのインポート)
# 2. [データの読み込み](#2.-データの読み込み)
# 3. [データの前処理とクレンジング (Advanced)](#3.-データの前処理とクレンジング-(Advanced))
# 4. [高度なデータ加工 (Comprehensive Pandas & Text)](#4.-高度なデータ加工-(Comprehensive-Pandas-&-Text))
# 5. [探索的データ分析 (EDA)](#5.-探索的データ分析-(EDA))
# 6. [特徴量エンジニアリング](#6.-特徴量エンジニアリング)
# 7. [特徴量選択](#7.-特徴量選択)
# 8. [モデル学習と検証 (回帰 - Multi-Model)](#8.-モデル学習と検証-(回帰---Multi-Model))
# 9. [モデル解釈 (SHAP Analysis)](#9.-モデル解釈-(SHAP-Analysis))
# 10. [モデル学習と検証 (分類 - Imbalanced Data)](#10.-モデル学習と検証-(分類---Imbalanced-Data))
# 11. [最終評価と結果の保存](#11.-最終評価と結果の保存)

# ## 1. 設定とライブラリのインポート
# 必要なライブラリをインポートし、表示設定や乱数シードの固定を行います。

# In[ ]:


import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.inspection import permutation_importance
# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, VotingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.utils import resample
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

# SHAP (Model Interpretability)
try:
    import shap
except ImportError:
    print("SHAP library not found. Please install it using 'pip install shap'")

# 設定
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

# 日本語フォントの設定（環境に合わせて変更してください）
# plt.rcParams['font.family'] = 'Meiryo' 

# 乱数シードの固定
SEED = 42
np.random.seed(SEED)


# ## 2. データの読み込み
# 指定されたディレクトリ内の複数のCSVファイルを読み込み、結合します。

# In[ ]:


def load_data(data_dir, file_pattern='*.csv'):
    """
    指定ディレクトリから複数のCSVを読み込み、結合して返す関数
    """
    files = glob.glob(os.path.join(data_dir, file_pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df_list = []
    for file in files:
        print(f"Loading: {file}")
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)

    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    return combined_df

# ダミーデータ生成（不均衡データ・テキストデータを含む）
def generate_dummy_data():
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    n = len(dates)

    # テキストデータのダミー生成
    log_messages = [
        "INFO: System Normal",
        "WARNING: High Latency (120ms)",
        "ERROR: Connection Failed (Code: 503)",
        "INFO: Maintenance Started",
        "INFO: Maintenance Completed"
    ]

    df = pd.DataFrame({
        'date': dates,
        'target': np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.1, n) + np.linspace(0, 5, n),
        'feature_1': np.random.rand(n) * 100,
        'feature_2': np.random.randint(0, 10, n),
        'feature_missing': np.where(np.random.rand(n) > 0.8, np.nan, np.random.rand(n)),
        'category_col': np.random.choice(['A', 'B', 'C'], n),
        'log_message': np.random.choice(log_messages, n) # テキストカラム
    })
    return df

df = generate_dummy_data()
print(f"Data Shape: {df.shape}")
df.head()


# ## 3. データの前処理とクレンジング (Advanced)
# 基本的な欠損処理に加え、KNNImputerやIterativeImputerを用いた高度な補完を行います。

# In[ ]:


# 日付カラムの変換とインデックス設定
date_col = 'date'
if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

# 欠損値の確認
print("Missing Values Before Imputation:\n", df.isnull().sum())

# --- Advanced Imputation Strategies ---
# 数値列の抽出
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 戦略1: KNN Imputer (近傍探索による補完)
imputer_knn = KNNImputer(n_neighbors=5)
df_knn_imputed = pd.DataFrame(imputer_knn.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

# 戦略2: Iterative Imputer (多変量回帰による連鎖的な補完)
imputer_iter = IterativeImputer(max_iter=10, random_state=SEED)
df_iter_imputed = pd.DataFrame(imputer_iter.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

# ここでは Iterative Imputer の結果を採用します
df[numeric_cols] = df_iter_imputed

# カテゴリ変数のエンコーディング
df = pd.get_dummies(df, columns=['category_col'], drop_first=True)

print("Missing Values After Imputation:\n", df.isnull().sum())


# ## 4. 高度なデータ加工 (Comprehensive Pandas & Text)
# Pandasの強力な機能を用いたデータ変換、リサンプリング、集約処理、および**テキストデータ処理**の例です。

# In[ ]:


# --- Text Data Processing (New) ---

# 1. 正規表現 (Regex) による抽出
# 'log_message' からエラーコードなどを抽出する例
# パターン: 'Code: ' の後ろの数字を抽出
df['error_code'] = df['log_message'].str.extract(r'Code: (\d+)')
df['error_code'] = df['error_code'].fillna(0).astype(int) # 欠損は0として扱う

# 2. 文字列操作とフラグ作成
# 特定のキーワードが含まれているかを判定
df['is_error'] = df['log_message'].str.contains('ERROR', case=False).astype(int)
df['is_maintenance'] = df['log_message'].str.contains('Maintenance', case=False).astype(int)

# 3. テキストのクリーニング
# 小文字化、空白除去など
df['clean_message'] = df['log_message'].str.lower().str.strip()

# 4. TF-IDF Vectorization (テキストの数値化)
# 重要な単語を特徴量として抽出
tfidf = TfidfVectorizer(max_features=5, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['clean_message'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])], index=df.index)

# 元のデータフレームに結合
df = pd.concat([df, tfidf_df], axis=1)

# 不要になったテキストカラムを削除
df.drop(columns=['log_message', 'clean_message'], inplace=True)

# --- Pandas Advanced Processing ---

# 5. Resampling
df_weekly = df.resample('W').agg({
    'target': ['mean', 'sum', 'max'],
    'feature_1': 'mean'
})

# 6. Rolling Window
df['rolling_mean_7d'] = df['target'].rolling('7D').mean()
df['rolling_std_7d'] = df['target'].rolling('7D').std()

# 7. EWMA
df['ewm_mean_span7'] = df['target'].ewm(span=7).mean()

# 8. Groupby + Transform
df['month'] = df.index.month
df['target_month_demeaned'] = df.groupby('month')['target'].transform(lambda x: x - x.mean())

# 9. Differencing & Percent Change
df['target_diff'] = df['target'].diff()
df['target_pct_change'] = df['target'].pct_change()

# 10. Expanding Window
df['expanding_max'] = df['target'].expanding().max()
df['expanding_mean'] = df['target'].expanding().mean()

# 11. Binning
df['target_bin'] = pd.qcut(df['target'], q=3, labels=['Low', 'Medium', 'High'])

# 12. Merge AsOf (Example)
economic_data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', end='2024-12-31', freq='MS'),
    'gdp_growth': np.random.rand(24)
})
df_sorted = df.sort_index()
economic_data_sorted = economic_data.sort_values('date')
df_merged = pd.merge_asof(df_sorted, economic_data_sorted, 
                          left_index=True, right_on='date', 
                          direction='backward')
df_merged = df_merged.set_index(df_sorted.index)
df = df_merged.drop(columns=['date', 'target_bin'])

df.drop(columns=['month'], inplace=True)
df.dropna(inplace=True)

df.head()


# ## 5. 探索的データ分析 (EDA)
# データの傾向、季節性、相関関係を可視化します。

# In[ ]:


target_col = 'target'

plt.figure(figsize=(15, 5))
plt.plot(df.index, df[target_col], label='Target')
plt.plot(df.index, df['ewm_mean_span7'], label='EWMA (Span 7)', linestyle='--', alpha=0.8)
plt.plot(df.index, df['expanding_mean'], label='Expanding Mean', linestyle=':', alpha=0.8)
plt.title('Time Series Plot with EWMA and Expanding Mean')
plt.legend()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# ## 6. 特徴量エンジニアリング
# 時系列特有の特徴量を作成します。

# In[ ]:


def create_features(data, target_col):
    df_feat = data.copy()

    df_feat['month'] = df_feat.index.month
    df_feat['day'] = df_feat.index.day
    df_feat['dayofweek'] = df_feat.index.dayofweek

    lags = [1, 7, 30]
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)

    df_feat = df_feat.dropna()
    return df_feat

df_processed = create_features(df, target_col)
print(f"Processed Data Shape: {df_processed.shape}")


# ## 7. 特徴量選択
# Permutation Importanceを用いたより信頼性の高い特徴量重要度の算出を行います。

# In[ ]:


X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

# Permutation Importance
rf = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
rf.fit(X, y)

result = permutation_importance(rf, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
perm_sorted_idx = result.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
sns.boxplot(data=result.importances[perm_sorted_idx].T, orient='h')
plt.yticks(range(len(X.columns)), X.columns[perm_sorted_idx])
plt.title("Permutation Importance")
plt.show()

# 上位特徴量の選択
top_features = X.columns[perm_sorted_idx][:10].tolist()
X_selected = X[top_features]
print(f"Selected Features: {top_features}")


# ## 8. モデル学習と検証 (回帰 - Multi-Model)
# 多様なモデル（HistGradientBoosting, SVR, Linear Models）を比較・検証します。

# In[ ]:


tscv = TimeSeriesSplit(n_splits=5)

# 比較するモデルの定義
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED),
    'Hist Gradient Boosting': HistGradientBoostingRegressor(random_state=SEED) # 高速で欠損値も扱える
}

results = {}

print("--- Model Comparison (CV RMSE) ---")
for name, model in models.items():
    # SVRなどはスケーリングが必須のためパイプライン化
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])

    cv_scores = cross_val_score(pipeline, X_selected, y, cv=tscv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    mean_rmse = rmse_scores.mean()
    results[name] = mean_rmse
    print(f"{name}: {mean_rmse:.4f}")

# ベストモデルの選択
best_model_name = min(results, key=results.get)
print(f"\nBest Model: {best_model_name}")

# ベストモデルでの最終学習
best_model = models[best_model_name]
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', best_model)
])

split_idx = int(len(X_selected) * 0.8)
X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

final_pipeline.fit(X_train, y_train)
y_pred = final_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Final Test RMSE ({best_model_name}): {rmse:.4f}")


# ## 9. モデル解釈 (SHAP Analysis)
# SHAP (SHapley Additive exPlanations) を用いて、モデルの予測根拠を可視化します。
# ※ SHAPライブラリが必要です (`pip install shap`)

# In[ ]:


try:
    import shap

    # SHAPはTree系モデルで最も効果的かつ高速です
    # 説明用にRandomForestモデルを再学習（または既存のTreeモデルを使用）
    explainer_model = RandomForestRegressor(n_estimators=100, random_state=SEED)
    explainer_model.fit(X_train, y_train)

    # Explainerの作成
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_test)

    print("--- SHAP Summary Plot ---")
    # 特徴量重要度と影響の方向性を表示
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot')
    plt.show()

    print("--- SHAP Dependence Plot ---")
    # 最も重要な特徴量とターゲットの関係（交互作用含む）を表示
    top_feature = X_test.columns[np.abs(shap_values).mean(0).argmax()]
    shap.dependence_plot(top_feature, shap_values, X_test, show=False)
    plt.title(f'SHAP Dependence Plot: {top_feature}')
    plt.show()

    print("--- SHAP Waterfall Plot (Local Explanation) ---")
    # 特定の予測（例：最初のテストデータ）に対する各特徴量の寄与を表示
    # shap.plots.waterfall は explainer(X) の結果オブジェクトを必要とします
    # バージョン互換性のため、ここでは汎用的な force_plot を使用する場合もありますが、
    # 最新のSHAPでは waterfall が推奨されます。

    # Explanationオブジェクトの作成
    explanation = shap.Explanation(values=shap_values, 
                                   base_values=explainer.expected_value, 
                                   data=X_test, 
                                   feature_names=X_test.columns)

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], show=False)
    plt.title('SHAP Waterfall Plot (First Sample)')
    plt.show()

except ImportError:
    print("SHAP is not installed. Skipping SHAP analysis.")
except Exception as e:
    print(f"An error occurred during SHAP analysis: {e}")


# ## 10. モデル学習と検証 (分類 - Imbalanced Data)
# 不均衡データ（少数クラス）への対策として、重み付け (Class Weight) とリサンプリングを行います。

# In[ ]:


# 分類ターゲットの作成（例：大きな上昇を予測するタスク、サンプル数が少ないと仮定）
threshold = y.quantile(0.9) # 上位10%を「急騰」とする
y_class = (y.shift(-1) > threshold).astype(int).iloc[:-1]
X_class = X_selected.iloc[:-1]

print("Class Distribution:\n", y_class.value_counts(normalize=True))

# 分割
split_idx_class = int(len(X_class) * 0.8)
X_train_c, X_test_c = X_class.iloc[:split_idx_class], X_class.iloc[split_idx_class:]
y_train_c, y_test_c = y_class.iloc[:split_idx_class], y_class.iloc[split_idx_class:]

# 対策1: Class Weight Balanced
clf_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED)
clf_balanced.fit(X_train_c, y_train_c)
y_pred_balanced = clf_balanced.predict(X_test_c)

print("\n--- Balanced Class Weight Results ---")
print(classification_report(y_test_c, y_pred_balanced))

# 対策2: Upsampling Minority Class (リサンプリング)
train_data = pd.concat([X_train_c, y_train_c], axis=1)
majority = train_data[train_data[y_train_c.name] == 0]
minority = train_data[train_data[y_train_c.name] == 1]

minority_upsampled = resample(minority, 
                              replace=True,     # サンプルを重複させる
                              n_samples=len(majority),    # 多数派と同じ数まで増やす
                              random_state=SEED)

train_upsampled = pd.concat([majority, minority_upsampled])
X_train_up = train_upsampled.drop(columns=[y_train_c.name])
y_train_up = train_upsampled[y_train_c.name]

clf_resampled = RandomForestClassifier(n_estimators=100, random_state=SEED)
clf_resampled.fit(X_train_up, y_train_up)
y_pred_resampled = clf_resampled.predict(X_test_c)

print("\n--- Upsampling Results ---")
print(classification_report(y_test_c, y_pred_resampled))


# ## 11. 最終評価と結果の保存
# 回帰モデルの予測結果を可視化し、結果をCSVとして保存します。

# In[ ]:


plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test, label='Actual', alpha=0.7)
plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7, linestyle='--')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()

results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
results_df.to_csv('forecast_results_ultimate.csv')
print("Results saved to forecast_results_ultimate.csv")

