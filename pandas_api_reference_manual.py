#!/usr/bin/env python
# coding: utf-8

# # Pandas API 究極リファレンス (Ultimate Pandas API Reference Manual)
# 
# このノートブックは、Pandasの機能を網羅的に解説した「究極のAPIリファレンス」です。
# 基本的な前処理だけでなく、入出力、高度な変形、時系列、可視化、スタイリングまで、実務で役立つ機能を辞書形式でまとめています。
# 
# ## 目次
# 1. [入出力 (Input/Output)](#1.-入出力-(Input/Output))
# 2. [データの確認と属性 (Inspection & Attributes)](#2.-データの確認と属性-(Inspection-&-Attributes))
# 3. [選択とインデックス操作 (Selection & Indexing)](#3.-選択とインデックス操作-(Selection-&-Indexing))
# 4. [データクレンジングと操作 (Data Cleaning & Manipulation)](#4.-データクレンジングと操作-(Data-Cleaning-&-Manipulation))
# 5. [変形とピボット (Reshaping & Pivoting)](#5.-変形とピボット-(Reshaping-&-Pivoting))
# 6. [データの結合 (Combining Data)](#6.-データの結合-(Combining-Data))
# 7. [グループ化と集約 (Groupby & Aggregation)](#7.-グループ化と集約-(Groupby-&-Aggregation))
# 8. [時系列データの詳細 (Time Series Deep Dive)](#8.-時系列データの詳細-(Time-Series-Deep-Dive))
# 9. [文字列操作 (String Manipulation)](#9.-文字列操作-(String-Manipulation))
# 10. [高度な機能 (Advanced / Functional)](#10.-高度な機能-(Advanced-/-Functional))
# 11. [可視化 (Visualization)](#11.-可視化-(Visualization))
# 12. [スタイリング (Styling)](#12.-スタイリング-(Styling))

# ## 0. セットアップ
# 必要なライブラリをインポートし、ダミーデータを準備します。

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

# 基本ダミーデータ
def create_dummy_df(n=100):
    np.random.seed(42)
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'value': np.random.randn(n) * 100,
        'value2': np.random.randint(0, 100, n),
        'text': np.random.choice(['foo', 'bar', 'baz'], n)
    })
    return df

df = create_dummy_df()
df.head()


# ## 1. 入出力 (Input/Output)
# 様々なフォーマットの読み書き。

# In[ ]:


# CSV
df.to_csv('temp.csv', index=False)
df_csv = pd.read_csv('temp.csv', parse_dates=['date'], encoding='utf-8')

# Excel (要 openpyxl)
# df.to_excel('temp.xlsx', sheet_name='Sheet1')
# df_excel = pd.read_excel('temp.xlsx', sheet_name='Sheet1')

# JSON
df.to_json('temp.json', orient='records')
df_json = pd.read_json('temp.json')

# Pickle (高速な一時保存)
df.to_pickle('temp.pkl')
df_pkl = pd.read_pickle('temp.pkl')

# Parquet (高速・圧縮・列指向、要 pyarrow)
# df.to_parquet('temp.parquet')
# df_parquet = pd.read_parquet('temp.parquet')

# クリップボード (Excel等からコピーしたデータを読み込む)
# df_clip = pd.read_clipboard()

print("IO examples executed.")


# ## 2. データの確認と属性 (Inspection & Attributes)
# データの概要を把握するための基本機能。

# In[ ]:


# 属性
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index}")
print(f"Dtypes:\n{df.dtypes}")

# 概要
print("\n--- Info ---")
df.info(memory_usage='deep')

# 統計量
print("\n--- Describe ---")
display(df.describe(percentiles=[0.1, 0.5, 0.9]))

# ユニーク値
print(f"Unique categories: {df['category'].unique()}")
print(f"Nunique: {df['category'].nunique()}")
print(f"Value Counts:\n{df['category'].value_counts(normalize=True)}")


# ## 3. 選択とインデックス操作 (Selection & Indexing)
# 行・列の抽出、条件抽出の完全ガイド。

# In[ ]:


# 基本的な選択
col = df['value']        # Seriesとして選択
cols = df[['value', 'category']] # DataFrameとして選択

# loc (ラベルベース)
row_loc = df.loc[0]      # 最初の行
subset_loc = df.loc[0:5, ['category', 'value']] # 範囲指定 (末尾含む)

# iloc (位置ベース)
row_iloc = df.iloc[0]
subset_iloc = df.iloc[0:5, [1, 2]] # 範囲指定 (末尾含まない)

# at / iat (スカラ値への高速アクセス)
val_at = df.at[0, 'value']
val_iat = df.iat[0, 2]

# 条件抽出 (Boolean Indexing)
cond = (df['category'] == 'A') & (df['value'] > 0)
filtered = df[cond]

# query (文字列で条件指定)
query_res = df.query('category == "A" and value > 0')

# isin
isin_res = df[df['category'].isin(['A', 'C'])]

# filter (列名やインデックス名でのフィルタ)
filter_col = df.filter(regex='val') # 'val'を含む列を選択

# where / mask (条件に合わないものをNaNまたは指定値にする)
# where: 条件Trueを残す, mask: 条件Trueを隠す(置換する)
df_where = df[['value']].where(df['value'] > 0, 0) # 0以下を0に置換

df_where.head()


# ## 4. データクレンジングと操作 (Data Cleaning & Manipulation)
# 欠損値、重複、ソート、型変換など。

# In[ ]:


# 欠損値処理
df_nan = df.copy()
df_nan.loc[0:5, 'value'] = np.nan

is_na = df_nan.isna().sum()
df_drop = df_nan.dropna() # 欠損を含む行を削除
df_fill = df_nan.fillna(0) # 0で埋める
df_interp = df_nan['value'].interpolate() # 線形補間

# 重複削除
df_dup = pd.concat([df.iloc[:2], df.iloc[:2]])
df_dedup = df_dup.drop_duplicates(subset=['date', 'category'])

# 列名変更
df_renamed = df.rename(columns={'value': 'val', 'value2': 'val2'})

# ソート
df_sorted = df.sort_values(by=['category', 'value'], ascending=[True, False])
top_3 = df.nlargest(3, 'value')

# 型変換
df['value2'] = df['value2'].astype(float)
df['date_str'] = df['date'].astype(str)
df['date_back'] = pd.to_datetime(df['date_str'])

print("Cleaning examples executed.")


# ## 5. 変形とピボット (Reshaping & Pivoting)
# データの形状を大きく変更する操作。

# In[ ]:


# Pivot Table (Long -> Wide)
# カテゴリごとの日付別平均値（日付が重複しないように月で集計）
df['month'] = df['date'].dt.to_period('M')
pivot = df.pivot_table(index='month', columns='category', values='value', aggfunc='mean')

# Melt (Wide -> Long)
# ピボットしたデータを元に戻すイメージ
melted = pivot.reset_index().melt(id_vars='month', var_name='category', value_name='value')

# Stack / Unstack
# Stack: 列をインデックス（行）に移動
# Unstack: インデックスを行から列に移動
stacked = pivot.stack()
unstacked = stacked.unstack()

# Crosstab (クロス集計)
xtab = pd.crosstab(df['category'], df['text'])

# Explode (リストを縦に展開)
df_list = pd.DataFrame({'id': [1], 'vals': [[10, 20, 30]]})
exploded = df_list.explode('vals')

print("--- Pivot ---")
display(pivot.head())
print("\n--- Melted ---")
display(melted.head())


# ## 6. データの結合 (Combining Data)
# 複数のデータフレームを結合する。

# In[ ]:


df1 = df.iloc[:5][['date', 'category', 'value']]
df2 = df.iloc[5:10][['date', 'category', 'value']]

# Concat (縦結合)
concat_v = pd.concat([df1, df2], axis=0, ignore_index=True)

# Concat (横結合)
concat_h = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

# Merge (Key結合)
master = pd.DataFrame({'category': ['A', 'B', 'C'], 'cat_name': ['Alpha', 'Beta', 'Gamma']})
merged = pd.merge(df, master, on='category', how='left')

# Join (Index結合)
df_idx1 = df1.set_index('date')
df_idx2 = df2.set_index('date')
# joined = df_idx1.join(df_idx2, lsuffix='_L', rsuffix='_R') # インデックスが重ならないと空になることが多い

# Combine First (欠損を別のDFで埋める)
df_na = df1.copy()
df_na.iloc[0, 2] = np.nan
combined = df_na.combine_first(df2.reset_index(drop=True)) # df_naの欠損をdf2で埋める

merged.head()


# ## 7. グループ化と集約 (Groupby & Aggregation)
# 高度な集約テクニック。

# In[ ]:


g = df.groupby('category')

# 複数の集約関数
agg_res = g.agg({
    'value': ['mean', 'std', 'min', 'max'],
    'value2': 'sum'
})

# Named Aggregation (列名を指定して集約)
named_agg = g.agg(
    avg_val=('value', 'mean'),
    max_val2=('value2', 'max')
)

# Transform (元の形状を維持して統計量を付与)
df['cat_mean'] = g['value'].transform('mean')
df['z_score_by_cat'] = g['value'].transform(lambda x: (x - x.mean()) / x.std())

# Filter (条件を満たすグループのみ残す)
# 例: データ数が30以上のカテゴリのみ残す
filtered_g = g.filter(lambda x: len(x) >= 30)

named_agg


# ## 8. 時系列データの詳細 (Time Series Deep Dive)
# 時系列特有の操作。

# In[ ]:


ts = df.set_index('date')['value']

# Resample (リサンプリング)
weekly_mean = ts.resample('W').mean()

# Shift (ずらす)
lag1 = ts.shift(1)
lead1 = ts.shift(-1)

# Diff (階差)
diff1 = ts.diff()

# Rolling (移動窓)
roll_mean = ts.rolling(window=7, center=True).mean()

# Expanding (累積窓)
exp_max = ts.expanding().max()

# EWM (指数加重移動平均)
ewm_mean = ts.ewm(span=10).mean()

# Date Offset
next_month = df['date'] + pd.DateOffset(months=1)

plt.figure(figsize=(10, 4))
ts.iloc[:50].plot(label='Original', alpha=0.5)
roll_mean.iloc[:50].plot(label='Rolling Mean', linestyle='--')
plt.legend()
plt.show()


# ## 9. 文字列操作 (String Manipulation)
# `str` アクセサの主要機能。

# In[ ]:


s = pd.Series([' A_foo ', 'B_bar', 'C_baz123'])

# 基本
print(s.str.lower())       # 小文字化
print(s.str.strip())       # 空白除去
print(s.str.len())         # 長さ

# 分割と結合
print(s.str.strip().str.split('_', expand=True)) # 分割してDataFrame化
print(s.str.cat(sep=','))  # 結合

# 判定
print(s.str.contains('foo')) # 含むか
print(s.str.startswith(' ')) # 始まるか

# 抽出と置換
print(s.str.extract(r'([a-z]+)')) # 正規表現抽出
print(s.str.replace(r'\d+', '', regex=True)) # 数字を除去


# ### 9.1 正規表現の詳細 (Regex Deep Dive)
# 正規表現を使った高度なパターンマッチングと抽出。

# In[ ]:


import re

s_regex = pd.Series([
    'Email: user@example.com, Phone: 090-1234-5678',
    'Email: admin@test.co.jp, Phone: 03-1234-5678',
    'No contact info'
])

# 1. 抽出 (Extract) - 名前付きグループ (?P<name>...)
# メールアドレスと電話番号を一度に抽出
pattern = r'Email: (?P<email>[\w.-]+@[\w.-]+), Phone: (?P<phone>[\d-]+)'
extracted = s_regex.str.extract(pattern)

# 2. すべて抽出 (Extract All) - 1つのセルに複数マッチする場合
s_multi = pd.Series(['A1, B2, C3', 'D4, E5'])
extracted_all = s_multi.str.extractall(r'(?P<letter>[A-Z])(?P<digit>\d)')

# 3. 置換 (Replace) - 後方参照 (Backreference)
# 電話番号のフォーマット変更: 090-1234-5678 -> ***-****-5678
# \\3 は3番目のグループの参照
masked = s_regex.str.replace(r'(\d{2,4})-(\d{2,4})-(\d{4})', r'***-****-\3', regex=True)

# 4. フィルタリング (Contains / Match)
# .co.jp を含むもの
jp_domains = s_regex[s_regex.str.contains(r'\.co\.jp', regex=True)]

print("--- Extracted ---")
display(extracted)
print("\n--- Masked ---")
display(masked)


# ## 10. 高度な機能 (Advanced / Functional)
# メソッドチェーンや高速化。

# In[ ]:


# Pipe (メソッドチェーン)
def add_prefix_to_col(data, col, prefix):
    data[col] = prefix + data[col].astype(str)
    return data

res_pipe = (
    df.copy()
    .pipe(add_prefix_to_col, 'category', 'Cat_')
    .assign(new_val = lambda x: x['value'] * 2)
    .query('value > 0')
)

# Apply (関数適用)
# axis=1で行ごとに適用（遅いので注意）
df['val_cat'] = df.apply(lambda row: row['value'] * row['value2'], axis=1)

# Eval (高速計算)
# 文字列で式を書くとNumExprを使って高速に計算される
df.eval('val_sum = value + value2', inplace=True)

res_pipe.head()


# ## 11. 可視化 (Visualization)
# Pandas組み込みのプロット機能。

# In[ ]:


df_plot = df.iloc[:50]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Line Plot
df_plot.plot(x='date', y='value', ax=axes[0, 0], title='Line Plot')

# Histogram
df['value'].plot.hist(bins=20, ax=axes[0, 1], title='Histogram')

# Scatter Plot
df.plot.scatter(x='value', y='value2', c='category', cmap='viridis', ax=axes[1, 0], title='Scatter Plot')

# Box Plot
df.boxplot(column='value', by='category', ax=axes[1, 1])
plt.suptitle('') # デフォルトのタイトルを消す
axes[1, 1].set_title('Box Plot')

plt.tight_layout()
plt.show()


# ## 12. スタイリング (Styling)
# DataFrameの表示を見やすく装飾する（Jupyter上で有効）。

# In[ ]:


style_df = df.head(10)[['category', 'value', 'value2']]

(
    style_df.style
    .format({'value': '{:.2f}', 'value2': '{}円'})
    .background_gradient(subset=['value'], cmap='coolwarm')
    .bar(subset=['value2'], color='lightblue')
    .highlight_max(subset=['value'], color='yellow')
)

