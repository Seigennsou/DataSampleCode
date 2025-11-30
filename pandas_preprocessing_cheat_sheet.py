#!/usr/bin/env python
# coding: utf-8

# # Pandas 前処理大全 (Pandas Preprocessing Cheat Sheet)
# 
# このノートブックは、データ分析や機械学習の実務で頻繁に使用する **Pandasによるデータ前処理** のテクニックを網羅した「逆引き辞典（Cheat Sheet）」です。
# データの読み込みから、欠損値処理、フィルタリング、文字列操作、時系列処理、そして高度な集約まで、実用的なコードスニペットを提供します。
# 
# ## 目次
# 1. [セットアップとダミーデータの生成](#1.-セットアップとダミーデータの生成)
# 2. [データの確認 (Inspection)](#2.-データの確認-(Inspection))
# 3. [欠損値の処理 (Missing Value Handling)](#3.-欠損値の処理-(Missing-Value-Handling))
# 4. [データのフィルタリングと抽出 (Filtering & Selection)](#4.-データのフィルタリングと抽出-(Filtering-&-Selection))
# 5. [データ型の変換 (Data Type Conversion)](#5.-データ型の変換-(Data-Type-Conversion))
# 6. [文字列データの処理 (String Manipulation)](#6.-文字列データの処理-(String-Manipulation))
# 7. [時系列データの処理 (Time Series Handling)](#7.-時系列データの処理-(Time-Series-Handling))
# 8. [グループ化と集約 (Grouping & Aggregation)](#8.-グループ化と集約-(Grouping-&-Aggregation))
# 9. [データの結合 (Merging & Joining)](#9.-データの結合-(Merging-&-Joining))
# 10. [高度なデータ変換 (Advanced Transformations)](#10.-高度なデータ変換-(Advanced-Transformations))

# ## 1. セットアップとダミーデータの生成
# まずは必要なライブラリをインポートし、練習用のリッチなダミーデータを作成します。

# In[ ]:


import pandas as pd
import numpy as np
import warnings

# 警告を無視（見やすくするため）
warnings.filterwarnings('ignore')

# 表示設定（列を省略せずに表示）
pd.set_option('display.max_columns', None)

def create_dummy_data(n=1000):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'category': np.random.choice(['A', 'B', 'C', np.nan], n, p=[0.3, 0.3, 0.3, 0.1]),
        'sub_category': np.random.choice(['X', 'Y', 'Z'], n),
        'value_1': np.random.normal(100, 10, n),
        'value_2': np.random.randint(0, 100, n),
        'value_missing': np.where(np.random.rand(n) > 0.8, np.nan, np.random.rand(n) * 100),
        'text_col': np.random.choice(['Info: OK', 'Warning: Check', 'Error: 404', ' Error: 500 '], n),
        'flag': np.random.choice([True, False], n)
    })

    # 意図的に外れ値を混入
    df.loc[0, 'value_1'] = 1000

    return df

df = create_dummy_data()
print(f"Data Shape: {df.shape}")
df.head()


# ## 2. データの確認 (Inspection)
# データを読み込んだら最初に行う基本的な確認操作です。

# In[ ]:


# 基本情報の確認（型、欠損値の有無、メモリ使用量）
print("--- Info ---")
df.info()

# 統計量の要約（数値列）
print("\n--- Describe (Numeric) ---")
display(df.describe())

# 統計量の要約（カテゴリ列）
print("\n--- Describe (Object) ---")
display(df.describe(include=['O']))

# カテゴリ変数の値のカウント
print("\n--- Value Counts (Category) ---")
print(df['category'].value_counts(dropna=False)) # dropna=Falseで欠損値もカウント


# ## 3. 欠損値の処理 (Missing Value Handling)
# 欠損値（NaN）の検出、削除、補完を行います。

# In[ ]:


# 欠損値の数を確認
print("Missing Values:\n", df.isnull().sum())

# 1. 欠損値を含む行を削除
df_dropped = df.dropna(subset=['category']) # category列に欠損がある行のみ削除

# 2. 定数で埋める
df_filled_const = df.fillna({'value_missing': 0, 'category': 'Unknown'})

# 3. 平均値・中央値で埋める
df['value_missing_mean'] = df['value_missing'].fillna(df['value_missing'].mean())
df['value_missing_median'] = df['value_missing'].fillna(df['value_missing'].median())

# 4. 前方・後方埋め (Forward/Backward Fill) - 時系列データで有効
df['value_missing_ffill'] = df['value_missing'].ffill()
df['value_missing_bfill'] = df['value_missing'].bfill()

# 5. 線形補間 (Interpolation) - 時系列データで滑らかに埋める
df['value_missing_interp'] = df['value_missing'].interpolate(method='linear')

df[['value_missing', 'value_missing_ffill', 'value_missing_interp']].head(10)


# ## 4. データのフィルタリングと抽出 (Filtering & Selection)
# 条件に合うデータを抽出します。

# In[ ]:


# 1. 条件式による抽出 (Boolean Indexing)
high_value = df[df['value_1'] > 110]

# 2. 複数条件 (AND: &, OR: |)
cond_and = df[(df['category'] == 'A') & (df['value_2'] > 50)]

# 3. queryメソッド (SQLライクで可読性が高い)
cond_query = df.query('category == "A" and value_2 > 50')

# 4. isin (特定の値リストに含まれるか)
target_cats = ['A', 'C']
cond_isin = df[df['category'].isin(target_cats)]

# 5. 文字列条件 (str accessor)
error_logs = df[df['text_col'].str.contains('Error')]

print(f"High Value Count: {len(high_value)}")
print(f"Error Logs Count: {len(error_logs)}")


# ## 5. データ型の変換 (Data Type Conversion)
# 適切なデータ型への変換は、メモリ節約とエラー防止に重要です。

# In[ ]:


# 型の確認
print(df.dtypes)

# 1. astypeによる変換
df['value_2'] = df['value_2'].astype(float)
df['flag'] = df['flag'].astype(int) # True/False -> 1/0

# 2. 数値への変換 (エラーをNaNにする場合: errors='coerce')
# 例: '1,000' などの文字列を数値にする際などに便利
df['value_1'] = pd.to_numeric(df['value_1'], errors='coerce')

# 3. 日付への変換
df['date'] = pd.to_datetime(df['date'])

# 4. カテゴリ型への変換 (メモリ削減と高速化)
df['category'] = df['category'].astype('category')

print("\n--- After Conversion ---")
print(df.dtypes)


# ## 6. 文字列データの処理 (String Manipulation)
# `str` アクセサを使って、Pandas Seriesに対して文字列操作を一括適用します。

# In[ ]:


# 1. 空白除去と小文字化
df['clean_text'] = df['text_col'].str.strip().str.lower()

# 2. 文字列の分割 (Split)
# 'info: ok' -> ['info', 'ok'] -> 最初の要素を取得
df['log_level'] = df['clean_text'].str.split(':').str[0]

# 3. 正規表現による抽出 (Extract)
# 数字部分を抽出
df['error_code'] = df['text_col'].str.extract(r'(\d+)').fillna(0).astype(int)

# 4. 置換 (Replace)
df['clean_text'] = df['clean_text'].str.replace('error', 'ERR', regex=False)

df[['text_col', 'clean_text', 'log_level', 'error_code']].head()


# ## 7. 時系列データの処理 (Time Series Handling)
# Pandasの最強機能の一つである時系列処理です。

# In[ ]:


# 日付をインデックスに設定
df_ts = df.set_index('date').sort_index()

# 1. 日付要素の抽出 (dt accessor)
# インデックスでない場合は df['date'].dt.year のように使う
df_ts['year'] = df_ts.index.year
df_ts['month'] = df_ts.index.month
df_ts['weekday'] = df_ts.index.day_name()

# 2. リサンプリング (Resample)
# 日次データを月次平均に変換
monthly_avg = df_ts['value_1'].resample('M').mean()

# 3. 移動平均 (Rolling)
# 7日間の移動平均
df_ts['rolling_7d'] = df_ts['value_1'].rolling(window=7).mean()

# 4. シフト (Shift) - ラグ特徴量の作成
df_ts['lag_1'] = df_ts['value_1'].shift(1) # 1日前の値

# 5. 階差 (Diff) - 変化量
df_ts['diff_1'] = df_ts['value_1'].diff(1) # 前日との差

# 6. 変化率 (Pct Change)
df_ts['pct_change'] = df_ts['value_1'].pct_change()

df_ts[['value_1', 'rolling_7d', 'lag_1', 'diff_1']].head(10)


# ## 8. グループ化と集約 (Grouping & Aggregation)
# SQLの `GROUP BY` に相当する強力な機能です。

# In[ ]:


# 1. 基本的な集約
print("--- Mean by Category ---")
print(df.groupby('category')['value_1'].mean())

# 2. 複数の統計量を一度に算出 (agg)
agg_res = df.groupby('category').agg({
    'value_1': ['mean', 'max', 'min', 'std'],
    'value_2': 'sum'
})
display(agg_res)

# 3. Transform - グループごとの統計量を元の行に付与
# 例: カテゴリごとの平均値との差分を計算したい場合
df['cat_mean'] = df.groupby('category')['value_1'].transform('mean')
df['diff_from_cat_mean'] = df['value_1'] - df['cat_mean']

# 4. Pivot Table - クロス集計
pivot = df.pivot_table(index='category', columns='sub_category', values='value_1', aggfunc='mean')
print("\n--- Pivot Table ---")
display(pivot)


# ## 9. データの結合 (Merging & Joining)
# 複数のDataFrameを結合します。

# In[ ]:


# マスタデータの作成
category_master = pd.DataFrame({
    'category': ['A', 'B', 'C'],
    'cat_name': ['Alpha', 'Beta', 'Gamma'],
    'priority': [1, 2, 3]
})

# 1. Merge (SQLのJOIN)
# left join
df_merged = pd.merge(df, category_master, on='category', how='left')

# 2. Concat (縦または横に連結)
# データを分割
df_part1 = df.iloc[:5]
df_part2 = df.iloc[5:10]

# 縦に連結 (UNION ALL)
df_concat = pd.concat([df_part1, df_part2], axis=0)

print("Merged Shape:", df_merged.shape)
df_merged[['category', 'cat_name', 'priority']].head()


# ## 10. 高度なデータ変換 (Advanced Transformations)
# その他の便利な変換テクニックです。

# In[ ]:


# 1. Apply - 関数を各行/列に適用 (柔軟だが遅い場合がある)
def complex_logic(row):
    if row['value_1'] > 100 and row['category'] == 'A':
        return 'High-A'
    else:
        return 'Normal'

df['custom_label'] = df.apply(complex_logic, axis=1)

# 2. Clip - 外れ値の丸め込み
# 上位99%タイル以上の値を99%タイルの値に置換
upper_limit = df['value_1'].quantile(0.99)
df['value_1_clipped'] = df['value_1'].clip(upper=upper_limit)

# 3. Binning (Cut/Qcut) - 数値のカテゴリ化
# 値の範囲で分割 (等間隔)
df['val_bin'] = pd.cut(df['value_1'], bins=3, labels=['Low', 'Mid', 'High'])
# 分位数で分割 (等頻度)
df['val_qbin'] = pd.qcut(df['value_1'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 4. Explode - リストを縦持ちに展開
df_list = pd.DataFrame({'id': [1, 2], 'tags': [['apple', 'banana'], ['orange']]})
df_exploded = df_list.explode('tags')

print("--- Explode Example ---")
display(df_exploded)

print("\n--- Final Data Head ---")
df.head()

