#!/usr/bin/env python
# coding: utf-8

# # 環境構築 (Install Dependencies)
# 
# このノートブックは、時系列予測テンプレートシリーズ（Ultimate, Random Forest, TimesFM）を実行するために必要なすべてのライブラリを一括でインストールするためのものです。
# 
# ## 注意事項 (Windows Python 3.12 Users)
# WindowsかつPython 3.12環境では、`timesfm` の依存ライブラリである `lingvo` (paxml) のインストールでエラーが発生することがあります。
# そのため、以下では **TimesFMを依存関係なしでインストールし、必要なライブラリのみを手動で追加する** 手順を採用しています。

# ### 1. 基本ライブラリのインストール

# In[ ]:


get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn shap')


# ### 2. TimesFMのインストール
# 以下の2つの方法のいずれかを選択してください。
# 
# #### オプション A: PyPIからインストール (Stable)
# 手軽にインストールしたい場合におすすめです。

# In[ ]:


# TimesFM本体のみをインストール (--no-deps で lingvo エラーを回避)
get_ipython().system('pip install timesfm --no-deps')

# 必要な依存ライブラリを個別インストール
get_ipython().system('pip install utilsforecast einshape huggingface-hub accelerate jax')


# #### オプション B: Gitからインストール (Latest / v2.5 API)
# 最新の機能 (TimesFM 2.5 APIなど) を使用する場合におすすめです。

# In[ ]:


# 既存のディレクトリがある場合は削除 (必要に応じて)
# import shutil
# shutil.rmtree('timesfm')

get_ipython().system('git clone https://github.com/google-research/timesfm.git')
get_ipython().run_line_magic('cd', 'timesfm')

# Windows/Python 3.12 workaround: 依存関係なしでインストール
get_ipython().system('pip install -e . --no-deps')

# 依存ライブラリの手動インストール (Torch backend)
get_ipython().system('pip install torch utilsforecast einshape huggingface-hub accelerate jax')

# 元のディレクトリに戻る
get_ipython().run_line_magic('cd', '..')


# ### 3. インストールの確認

# In[ ]:


try:
    import timesfm
    print("TimesFM imported successfully!")
except ImportError as e:
    print(f"TimesFM import failed: {e}")
    print("Note: If you see an error related to 'paxml', please ensure you are using the PyTorch backend in the notebook.")

