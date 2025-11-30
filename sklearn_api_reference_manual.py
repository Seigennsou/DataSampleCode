#!/usr/bin/env python
# coding: utf-8

# # Scikit-Learn API 究極リファレンス (Ultimate Sklearn API Reference Manual)
# 
# このノートブックは、Pythonの標準的な機械学習ライブラリである **Scikit-learn (sklearn)** の主要機能を網羅した「究極のAPIリファレンス」です。
# データの前処理から、モデルの構築、評価、チューニング、そしてパイプライン化まで、実務で頻繁に使用するコードパターンを辞書形式でまとめています。
# 
# ## 目次
# 1. [データの前処理 (Data Preparation)](#1.-データの前処理-(Data-Preparation))
# 2. [特徴量の選択と削減 (Feature Selection & Reduction)](#2.-特徴量の選択と削減-(Feature-Selection-&-Reduction))
# 3. [モデル選択とデータ分割 (Model Selection & Split)](#3.-モデル選択とデータ分割-(Model-Selection-&-Split))
# 4. [教師あり学習 - 回帰 (Regression)](#4.-教師あり学習---回帰-(Regression))
# 5. [教師あり学習 - 分類 (Classification)](#5.-教師あり学習---分類-(Classification))
# 6. [教師なし学習 - クラスタリング (Clustering)](#6.-教師なし学習---クラスタリング-(Clustering))
# 7. [モデル評価 (Model Evaluation)](#7.-モデル評価-(Model-Evaluation))
# 8. [ハイパーパラメータチューニング (Hyperparameter Tuning)](#8.-ハイパーパラメータチューニング-(Hyperparameter-Tuning))
# 9. [パイプライン (Pipelines & Composites)](#9.-パイプライン-(Pipelines-&-Composites))
# 10. [モデルの保存と読み込み (Model Persistence)](#10.-モデルの保存と読み込み-(Model-Persistence))

# ## 0. セットアップ
# 必要なライブラリとダミーデータを準備します。

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# ダミーデータの生成 (回帰用と分類用)
from sklearn.datasets import make_regression, make_classification

# 回帰データ
X_reg, y_reg = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)

# 分類データ
X_clf, y_clf = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)

print("Setup Complete.")


# ## 1. データの前処理 (Data Preparation)
# モデルに入力する前のデータ変換。

# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# --- スケーリング (Scaling) ---
# 標準化 (平均0, 分散1)
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X_reg)

# 正規化 (0-1の範囲)
scaler_mm = MinMaxScaler()
X_mm = scaler_mm.fit_transform(X_reg)

# Robust Scaler (外れ値に強い、四分位範囲を使用)
scaler_rob = RobustScaler()
X_rob = scaler_rob.fit_transform(X_reg)

# --- エンコーディング (Encoding) ---
cats = [['Male', 'S'], ['Female', 'M'], ['Female', 'L']]

# One-Hot Encoding (ダミー変数化)
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cats_ohe = ohe.fit_transform(cats)

# Label Encoding (ターゲット変数のラベル化: 'A', 'B' -> 0, 1)
le = LabelEncoder()
y_le = le.fit_transform(['cat', 'dog', 'cat'])

# Ordinal Encoding (順序特徴量: 'S', 'M', 'L' -> 0, 1, 2)
oe = OrdinalEncoder(categories=[['S', 'M', 'L']]) # 順序を指定可能
cats_oe = oe.fit_transform([['S'], ['L'], ['M']])

# --- 欠損値補完 (Imputation) ---
X_nan = X_reg.copy()
X_nan[0, 0] = np.nan

# 単純な補完 (平均、中央値、最頻値)
imp_mean = SimpleImputer(strategy='mean')
X_imp = imp_mean.fit_transform(X_nan)

# KNN補完 (近傍点の平均)
imp_knn = KNNImputer(n_neighbors=5)
X_knn = imp_knn.fit_transform(X_nan)

print("Preprocessing examples executed.")


# ## 2. 特徴量の選択と削減 (Feature Selection & Reduction)
# 重要な特徴量を選び、次元を削減する。

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# --- 特徴量選択 (Selection) ---
# 統計的検定 (F値) で上位K個を選択
selector_k = SelectKBest(score_func=f_regression, k=5)
X_kbest = selector_k.fit_transform(X_reg, y_reg)

# RFE (再帰的特徴量削減)
estimator = RandomForestRegressor(n_estimators=10, random_state=42)
selector_rfe = RFE(estimator, n_features_to_select=5, step=1)
X_rfe = selector_rfe.fit_transform(X_reg, y_reg)

# SelectFromModel (モデルの重要度に基づく選択)
selector_sfm = SelectFromModel(estimator, threshold='median')
X_sfm = selector_sfm.fit_transform(X_reg, y_reg)

# --- 次元削減 (Reduction) ---
# PCA (主成分分析)
pca = PCA(n_components=0.95) # 分散の95%を説明できる次元数まで削減
X_pca = pca.fit_transform(X_reg)

print(f"Original Shape: {X_reg.shape}")
print(f"PCA Shape: {X_pca.shape}")


# ## 3. モデル選択とデータ分割 (Model Selection & Split)
# 学習用とテスト用のデータ分割、交差検証の準備。

# In[ ]:


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, TimeSeriesSplit

# 単純な分割 (Hold-out)
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# K-Fold (回帰用)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X_reg):
    pass # ここで学習・検証を行う

# Stratified K-Fold (分類用: クラス比率を維持)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# TimeSeriesSplit (時系列用: 未来のデータを検証に使わない)
tscv = TimeSeriesSplit(n_splits=5)
for train_index, val_index in tscv.split(X_reg):
    pass

print("Split examples executed.")


# ## 4. 教師あり学習 - 回帰 (Regression)
# 数値を予測するモデル。

# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 線形回帰
lr = LinearRegression()
lr.fit(X_train, y_train)

# 正則化付き線形回帰 (Ridge: L2, Lasso: L1)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# サポートベクター回帰 (SVR)
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# ランダムフォレスト (並列化可能: n_jobs=-1)
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)

# 勾配ブースティング (GBDT)
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

print("Regression models initialized.")


# ## 5. 教師あり学習 - 分類 (Classification)
# クラス（ラベル）を予測するモデル。

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ロジスティック回帰 (線形分類)
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)

# サポートベクターマシン (SVC)
# probability=Trueで確率を出力可能にする（遅くなるので注意）
svc = SVC(kernel='rbf', C=1.0, probability=True)

# K近傍法 (KNN)
knn = KNeighborsClassifier(n_neighbors=5)

# 決定木
dt = DecisionTreeClassifier(max_depth=5)

# ランダムフォレスト分類
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

print("Classification models initialized.")


# ## 6. 教師なし学習 - クラスタリング (Clustering)
# データのグループ化。

# In[ ]:


from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# K-Means (K平均法)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_km = kmeans.fit_predict(X_reg)

# DBSCAN (密度ベース)
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_db = dbscan.fit_predict(X_reg)

# 階層的クラスタリング
agg = AgglomerativeClustering(n_clusters=3)
clusters_agg = agg.fit_predict(X_reg)

print(f"KMeans Labels: {np.unique(clusters_km)}")


# ## 7. モデル評価 (Model Evaluation)
# 予測精度の測定。

# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import silhouette_score

# --- 回帰評価 ---
y_pred_reg = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
print(f"Regression R2: {r2:.4f}")

# --- 分類評価 ---
y_pred_clf = log_reg.predict(X_test)
y_prob_clf = log_reg.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred_clf)
f1 = f1_score(y_test, y_pred_clf)
roc_auc = roc_auc_score(y_test, y_prob_clf)
cm = confusion_matrix(y_test, y_pred_clf)

print(f"Classification Accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_clf))

# --- クラスタリング評価 ---
sil = silhouette_score(X_reg, clusters_km)
print(f"Silhouette Score: {sil:.4f}")


# ## 8. ハイパーパラメータチューニング (Hyperparameter Tuning)
# 最適なパラメータの探索。

# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# パラメータグリッド
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5]
}

# Grid Search (全探索)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best Params: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# Randomized Search (ランダム探索 - 高速)
rand_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=5, # 試行回数
    cv=3,
    random_state=42
)
rand_search.fit(X_train, y_train)


# ## 9. パイプライン (Pipelines & Composites)
# 前処理とモデルを連結して一つの推論器として扱う。

# In[ ]:


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

# --- Column Transformer ---
# 数値列とカテゴリ列で異なる前処理を適用
num_cols = [0, 1, 2] # 列インデックスまたは列名
cat_cols = [3]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ]
)

# --- Pipeline ---
# 前処理 -> 特徴量選択 -> モデル
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=5)),
    ('classifier', LogisticRegression())
])

# 学習と予測
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print("Pipeline executed successfully.")


# ## 10. モデルの保存と読み込み (Model Persistence)
# 学習済みモデルの永続化。

# In[ ]:


import joblib

# 保存
joblib.dump(pipe, 'model_pipeline.pkl')

# 読み込み
loaded_pipe = joblib.load('model_pipeline.pkl')

# 再利用
result = loaded_pipe.score(X_test, y_test)
print(f"Loaded Model Score: {result:.4f}")

