#!/usr/bin/env python
# coding: utf-8

# # 時系列データ予測テンプレート (Random Forest Edition)
# 
# このノートブックは、**ランダムフォレスト (Random Forest)** に特化した時系列予測テンプレートです。
# 強力で解釈性の高いランダムフォレストモデルを使用し、高度なデータ前処理、テキスト処理、特徴量エンジニアリング、そしてSHAPによる詳細なモデル解釈までを一貫して行います。
# 
# ## 目次
# 1. [設定とライブラリのインポート](#1.-設定とライブラリのインポート)
# 2. [データの読み込み](#2.-データの読み込み)
# 3. [データの前処理とクレンジング (Advanced)](#3.-データの前処理とクレンジング-(Advanced))
# 4. [高度なデータ加工 (Comprehensive Pandas & Text)](#4.-高度なデータ加工-(Comprehensive-Pandas-&-Text))
# 5. [探索的データ分析 (EDA)](#5.-探索的データ分析-(EDA))
# 6. [特徴量エンジニアリング](#6.-特徴量エンジニアリング)
# 7. [特徴量選択](#7.-特徴量選択)
# 8. [モデル学習と検証 (回帰 - Random Forest)](#8.-モデル学習と検証-(回帰---Random-Forest))
# 9. [モデル解釈 (SHAP Analysis)](#9.-モデル解釈-(SHAP-Analysis))
# 10. [モデル学習と検証 (分類 - Imbalanced Data)](#10.-モデル学習と検証-(分類---Imbalanced-Data))
# 11. [最終評価と結果の保存](#11.-最終評価と結果の保存)

# ## 1. 設定とライブラリのインポート
# 必要なライブラリをインポートします。ここではランダムフォレストに焦点を当てています。

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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.inspection import permutation_importance

# Models (Random Forest Only)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
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

# 乱数シードの固定
SEED = 42
np.random.seed(SEED)


# ## 2. データの読み込み
# ダミーデータを生成して使用します。

# In[ ]:


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
        'log_message': np.random.choice(log_messages, n)
    })
    return df

df = generate_dummy_data()
print(f"Data Shape: {df.shape}")
df.head()


# ## 3. データの前処理とクレンジング (Advanced)
# Iterative Imputerなどを用いた高度な欠損値補完を行います。

# In[ ]:


date_col = 'date'
if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

# 数値列の抽出と補完
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer_iter = IterativeImputer(max_iter=10, random_state=SEED)
df[numeric_cols] = imputer_iter.fit_transform(df[numeric_cols])

# カテゴリ変数のエンコーディング
df = pd.get_dummies(df, columns=['category_col'], drop_first=True)


# ## 4. 高度なデータ加工 (Comprehensive Pandas & Text)
# テキスト処理（Regex, TF-IDF）とPandasによる時系列特徴量の作成を行います。

# In[ ]:


# --- Text Data Processing ---
df['error_code'] = df['log_message'].str.extract(r'Code: (\d+)').fillna(0).astype(int)
df['is_error'] = df['log_message'].str.contains('ERROR', case=False).astype(int)
df['clean_message'] = df['log_message'].str.lower().str.strip()

tfidf = TfidfVectorizer(max_features=5, stop_words='english')
tfidf_df = pd.DataFrame(tfidf.fit_transform(df['clean_message']).toarray(), 
                        columns=[f'tfidf_{i}' for i in range(5)], index=df.index)
df = pd.concat([df, tfidf_df], axis=1)
df.drop(columns=['log_message', 'clean_message'], inplace=True)

# --- Pandas Advanced Processing ---
df['rolling_mean_7d'] = df['target'].rolling('7D').mean()
df['ewm_mean_span7'] = df['target'].ewm(span=7).mean()
df['target_diff'] = df['target'].diff()
df['expanding_max'] = df['target'].expanding().max()

df.dropna(inplace=True)


# ## 5. 探索的データ分析 (EDA)
# ターゲット変数の推移を確認します。

# In[ ]:


plt.figure(figsize=(15, 5))
plt.plot(df.index, df['target'], label='Target')
plt.plot(df.index, df['ewm_mean_span7'], label='EWMA (Span 7)', linestyle='--')
plt.legend()
plt.show()


# ## 6. 特徴量エンジニアリング
# ラグ特徴量などを作成します。

# In[ ]:


def create_features(data, target_col):
    df_feat = data.copy()
    df_feat['month'] = df_feat.index.month
    df_feat['dayofweek'] = df_feat.index.dayofweek

    for lag in [1, 7, 30]:
        df_feat[f'lag_{lag}'] = df_feat[target_col].shift(lag)

    return df_feat.dropna()

df_processed = create_features(df, 'target')


# ## 7. 特徴量選択
# Random Forestを用いたPermutation Importanceで特徴量を選別します。

# In[ ]:


X = df_processed.drop(columns=['target'])
y = df_processed['target']

rf = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
rf.fit(X, y)

result = permutation_importance(rf, X, y, n_repeats=10, random_state=SEED, n_jobs=-1)
perm_sorted_idx = result.importances_mean.argsort()[::-1]
top_features = X.columns[perm_sorted_idx][:10].tolist()
X_selected = X[top_features]

print(f"Selected Features: {top_features}")


# ## 8. モデル学習と検証 (回帰 - Random Forest)
# RandomForestRegressorを使用し、ハイパーパラメータチューニングを行います。

# In[ ]:


tscv = TimeSeriesSplit(n_splits=5)

# パラメータグリッド
param_dist = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=SEED))
])

search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=tscv, 
    scoring='neg_mean_squared_error', 
    random_state=SEED, 
    n_jobs=-1
)

search.fit(X_selected, y)
print(f"Best Parameters: {search.best_params_}")
print(f"Best CV RMSE: {np.sqrt(-search.best_score_):.4f}")

# 最終モデルの学習
best_model = search.best_estimator_
split_idx = int(len(X_selected) * 0.8)
X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.4f}")


# ## 9. モデル解釈 (SHAP Analysis)
# Random Forestモデルの予測根拠をSHAPで可視化します。

# In[ ]:


try:
    import shap
    # Pipelineからモデルを取り出す
    rf_model = best_model.named_steps['regressor']

    # Scalerを通したデータが必要なため、変換を行う
    scaler = best_model.named_steps['scaler']
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_scaled)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, show=False)
    plt.title('SHAP Summary Plot (Random Forest)')
    plt.show()
except ImportError:
    print("SHAP not installed.")


# ## 10. モデル学習と検証 (分類 - Imbalanced Data)
# RandomForestClassifierを用いた分類タスク（不均衡データ対応）です。

# In[ ]:


threshold = y.quantile(0.9)
y_class = (y.shift(-1) > threshold).astype(int).iloc[:-1]
X_class = X_selected.iloc[:-1]

split_idx_c = int(len(X_class) * 0.8)
X_train_c, X_test_c = X_class.iloc[:split_idx_c], X_class.iloc[split_idx_c:]
y_train_c, y_test_c = y_class.iloc[:split_idx_c], y_class.iloc[split_idx_c:]

# Class Weight Balanced
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED)
clf.fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)

print(classification_report(y_test_c, y_pred_c))


# ## 11. 最終評価と結果の保存
# 結果を保存します。

# In[ ]:


results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_csv('rf_forecast_results.csv')
print("Saved to rf_forecast_results.csv")

