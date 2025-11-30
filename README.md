# データ分析・時系列予測 テンプレート集 (DataSampleCode)

このリポジトリは、データ分析、機械学習、特に**時系列予測**の実務で使用できる高品質なJupyter Notebookテンプレートと、Pandas/Sklearnの速査マニュアル（Cheat Sheet）をまとめたものです。

すべてのノートブックは**日本語**で記述されており、コピー＆ペーストですぐに実務に適用できるように設計されています。

## 📂 収録コンテンツ

### 📈 時系列予測テンプレート (Time Series Forecasting)

| ファイル名 | 説明 |
| :--- | :--- |
| **`time_series_forecast_template.ipynb`** | **Ultimate Edition**。前処理、特徴量エンジニアリング、複数モデル比較、SHAP解析まで網羅した決定版テンプレート。 |
| **`timesfm_forecast_template.ipynb`** | Googleの基盤モデル **TimesFM 2.5** を使用したZero-Shot予測のテンプレート。 |
| **`sklearn_time_series_forecast_template.ipynb`** | **Scikit-learn** の回帰モデル（LightGBM等）を使って時系列予測を行うためのテンプレート。再帰的予測（Recursive Forecasting）の実装を含む。 |
| **`random_forest_forecast_template.ipynb`** | **Random Forest** に特化したシンプルかつ強力な予測テンプレート。 |

### 📚 速査マニュアル・リファレンス (Cheat Sheets & References)

| ファイル名 | 説明 |
| :--- | :--- |
| **`pandas_api_reference_manual.ipynb`** | **Pandas API 究極リファレンス**。入出力、変形、時系列、可視化、正規表現など、Pandasの機能を網羅した辞書。 |
| **`pandas_preprocessing_cheat_sheet.ipynb`** | **Pandas 前処理大全**。実務で頻出するデータ加工・クレンジングのパターン集。 |
| **`sklearn_api_reference_manual.ipynb`** | **Scikit-learn API 究極リファレンス**。前処理からモデル構築、評価、パイプラインまで、MLフロー全体をカバー。 |

### 🛠️ 環境構築 (Setup)

| ファイル名 | 説明 |
| :--- | :--- |
| **`install_dependencies.ipynb`** | 必要なライブラリを一括インストールするためのノートブック。Windows/Python 3.12環境でのTimesFMインストール回避策も含む。 |
| **`requirements.txt`** | 依存ライブラリ一覧。 |

## 🚀 使い方

1. リポジトリをクローンします。
   ```bash
   git clone https://github.com/Seigennsou/DataSampleCode.git
   ```
2. `install_dependencies.ipynb` を開き、セルを実行してライブラリをインストールします。
3. 目的に合ったテンプレート（`.ipynb`）を開いて使用してください。
   - テキストエディタで閲覧したい場合は、同名の `.py` ファイルを参照してください。

## ⚠️ 注意事項

- **TimesFM** の使用には、十分なメモリ（またはGPU）が推奨されます。
- Windows環境で `lingvo` 関連のエラーが出る場合は、`install_dependencies.ipynb` に記載されている回避策（`--no-deps` インストール）を使用してください。

---
Created by Antigravity
