#!/usr/bin/env python
# coding: utf-8

# # 時系列データ予測テンプレート (TimesFM-2.5 Edition)
# 
# このノートブックは、Googleが開発した時系列基盤モデル **TimesFM-2.5 (Time Series Foundation Model)** を使用した予測テンプレートです。
# 他のテンプレートと同様に、**高度なデータ前処理**、**テキストデータ処理**、**探索的データ分析 (EDA)** の完全なワークフローを含みつつ、予測モデルとしてTimesFMを採用しています。
# 
# ## 特徴
# - **Zero-Shot Forecasting**: モデルの学習（fit）を行わずに、過去のデータを与えるだけで未来を予測します。
# - **Foundation Model**: 大規模な事前学習済みモデルを利用し、高い汎化性能を期待できます。
# - **Comprehensive Workflow**: 欠損値補完やテキスト特徴量の抽出など、実務的な前処理フローを網羅しています。
# 
# ## 目次
# 1. [設定とライブラリのインストール](#1.-設定とライブラリのインストール)
# 2. [データの読み込み](#2.-データの読み込み)
# 3. [データの前処理とクレンジング (Advanced)](#3.-データの前処理とクレンジング-(Advanced))
# 4. [高度なデータ加工 (Comprehensive Pandas & Text)](#4.-高度なデータ加工-(Comprehensive-Pandas-&-Text))
# 5. [探索的データ分析 (EDA)](#5.-探索的データ分析-(EDA))
# 6. [TimesFMモデルの準備](#6.-TimesFMモデルの準備)
# 7. [予測の実行 (Zero-Shot)](#7.-予測の実行-(Zero-Shot))
# 8. [最終評価と結果の保存](#8.-最終評価と結果の保存)

# ## 1. 設定とライブラリのインストール
# TimesFMライブラリをインストールし、必要なモジュールをインポートします。

# In[ ]:


# Windows & Python 3.12 workaround: Install timesfm without dependencies to avoid lingvo/paxml errors
get_ipython().system('pip install timesfm --no-deps')
get_ipython().system('pip install utilsforecast einshape huggingface-hub accelerate jax')

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timesfm
import torch # Added for v2.5 API

# Scikit-learn modules (for preprocessing)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 設定
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

# 乱数シードの固定
SEED = 42
np.random.seed(SEED)


# ## 2. データの読み込み
# ダミーデータを生成して使用します。テキストデータや欠損値を含むリアルなデータセットを模倣します。

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
        'unique_id': 'series_1', # TimesFM用のID
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
# TimesFMは欠損のない履歴データを期待するため、Iterative Imputerを用いて高度な補完を行います。

# In[ ]:


date_col = 'date'
if date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

# 数値列の抽出
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Iterative Imputer (多変量回帰による連鎖的な補完)
imputer_iter = IterativeImputer(max_iter=10, random_state=SEED)
df[numeric_cols] = imputer_iter.fit_transform(df[numeric_cols])

# カテゴリ変数のエンコーディング
df = pd.get_dummies(df, columns=['category_col'], drop_first=True)

print("Missing Values After Imputation:\n", df.isnull().sum())


# ## 4. 高度なデータ加工 (Comprehensive Pandas & Text)
# テキスト処理（Regex, TF-IDF）とPandasによる特徴量作成を行います。
# ※ TimesFM自体は単変量予測（ターゲットのみ使用）が基本ですが、これらの特徴量はデータの理解(EDA)や、将来的に共変量として使用する場合に有用です。

# In[ ]:


# --- Text Data Processing ---

# 1. 正規表現 (Regex) による抽出
df['error_code'] = df['log_message'].str.extract(r'Code: (\d+)').fillna(0).astype(int)

# 2. 文字列操作とフラグ作成
df['is_error'] = df['log_message'].str.contains('ERROR', case=False).astype(int)

# 3. テキストのクリーニング
df['clean_message'] = df['log_message'].str.lower().str.strip()

# 4. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5, stop_words='english')
tfidf_df = pd.DataFrame(tfidf.fit_transform(df['clean_message']).toarray(), 
                        columns=[f'tfidf_{i}' for i in range(5)], index=df.index)
df = pd.concat([df, tfidf_df], axis=1)
df.drop(columns=['log_message', 'clean_message'], inplace=True)

# --- Pandas Advanced Processing ---

# 5. Rolling & EWMA
df['rolling_mean_7d'] = df['target'].rolling('7D').mean()
df['ewm_mean_span7'] = df['target'].ewm(span=7).mean()

# 6. Differencing
df['target_diff'] = df['target'].diff()

# 7. Expanding
df['expanding_max'] = df['target'].expanding().max()

df.dropna(inplace=True)
df.head()


# ## 5. 探索的データ分析 (EDA)
# データの傾向を可視化します。

# In[ ]:


plt.figure(figsize=(15, 5))
plt.plot(df.index, df['target'], label='Target')
plt.plot(df.index, df['ewm_mean_span7'], label='EWMA (Span 7)', linestyle='--', alpha=0.8)
plt.title('Time Series Plot with EWMA')
plt.legend()
plt.show()


# ## 6. TimesFMモデルの準備 (v2.5 API)
# Hugging Faceからチェックポイント (`google/timesfm-2.5-200m-pytorch`) をロードし、コンパイルします。

# In[ ]:


# GPUが使える場合は設定 (ここではCPU/GPU自動判定の例として記述しますが、基本はCPUでも動作します)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# float32の精度設定 (推奨)
torch.set_float32_matmul_precision("high")

# モデルのロード (PyTorch backend)
# 200Mパラメータモデルを使用
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

# モデルのコンパイル (設定)
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,       # 入力系列の最大長
        max_horizon=30,         # 予測ホライゾン (30日)
        normalize_inputs=True,  # 入力の正規化
        use_continuous_quantile_head=True, # 量子化ヘッドの使用
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

print("TimesFM-2.5 Model Loaded and Compiled Successfully.")


# ## 7. 予測の実行 (Zero-Shot)
# 学習データを与えて、未来の予測を行います。

# In[ ]:


# データの準備
# TimesFM v2.5の forecast メソッドは numpy array のリストを入力として受け取ります
# (Batch size x Context length)

# 訓練データ (過去データ) の準備
# 最後の30日をテスト用として除外し、それ以前をコンテキストとして使用
train_values = df['target'].iloc[:-30].values
test_values = df['target'].iloc[-30:].values
test_dates = df.index[-30:]

# 入力はリスト形式 (複数の時系列を同時に予測可能ですが、今回は1つ)
inputs = [train_values]

# 予測の実行
print(f"Forecasting for horizon: 30...")
point_forecast, quantile_forecast = model.forecast(
    horizon=30,
    inputs=inputs,
)

# 結果の確認
# point_forecast shape: (batch_size, horizon) -> (1, 30)
# quantile_forecast shape: (batch_size, horizon, quantiles) -> (1, 30, 10) (mean + 9 quantiles)

print("Point Forecast Shape:", point_forecast.shape)

# 結果をDataFrameに整形
pred_values = point_forecast[0] # 最初のバッチ(今回は1つだけ)を取り出す
forecast_df = pd.DataFrame({
    'date': test_dates,
    'timesfm_pred': pred_values
}).set_index('date')

forecast_df.head()


# ## 8. 最終評価と結果の保存
# 実測値と予測値を比較し、評価指標を算出・保存します。

# In[ ]:


# 可視化
plt.figure(figsize=(15, 6))
plt.plot(df.index[-60:], df['target'].iloc[-60:], label='Actual (History + Test)', color='blue', alpha=0.5)
plt.plot(forecast_df.index, forecast_df['timesfm_pred'], label='TimesFM Forecast', color='red', linestyle='--')
plt.axvline(x=df.index[-31], color='gray', linestyle=':', label='Forecast Start')
plt.title('TimesFM-2.5 Zero-Shot Forecasting')
plt.legend()
plt.show()

# 評価
rmse = np.sqrt(mean_squared_error(test_values, forecast_df['timesfm_pred']))
mae = mean_absolute_error(test_values, forecast_df['timesfm_pred'])

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# 保存
results_df = pd.DataFrame({'Actual': test_values, 'Predicted': forecast_df['timesfm_pred'].values}, index=test_dates)
results_df.to_csv('timesfm_forecast_results.csv')
print("Saved to timesfm_forecast_results.csv")

