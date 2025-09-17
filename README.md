# westa-ocr

## **1\. 概要**

本プロジェクトは、GoogleのGemini APIを活用し、PDFファイル（受発注伝票などを想定）から指定された項目を抽出し、構造化データとして出力する業務用Webアプリケーションです。  
Dockerコンテナ上での稼働を前提として設計されています。

## **2\. システム構成**

* **オーケストレーション**: Docker (単一コンテナ)
* **フロントエンド**:  
  * **フレームワーク**: Vue.js 3 (CDN経由)  
  * **UI**: Bootstrap 5  
  * **配信**: Flask による静的ファイル配信
* **バックエンド**:  
  * **フレームワーク**: Flask (Python)  
  * **AIモデル**: Google Gemini 1.5 Pro  
* **主要ライブラリ**:  
  * **Python**: google-generativeai, flask, PyPDF2, pillow, concurrent.futures, python-dotenv

## **3\. 機能**

* PDFファイルをアップロードし、バックエンドでページごとに並列OCR処理を実行します。  
* Gemini APIの **Structured Output** を利用し、高精度で指定された項目を抽出します。  
* 抽出結果は画面上にテーブルで表示され、JSON形式でダウンロード可能です。

## **4\. 抽出項目**

Gemini APIを利用して、PDFから以下の項目を抽出します。

| キー (JSON) | 日本語名称 | データ型 |
| :---- | :---- | :---- |
| pdf\_id | PDFのID | string |
| delivery\_location | 納入場所 | string |
| customer\_name | 得意先 | string |
| customer\_order\_number | 得意先注文番号 | string |
| order\_date | 受注日 | string |
| shipping\_date | 出荷予定日 | string |
| customer\_delivery\_date | 顧客納期 | string |
| customer\_item\_code | 得意先品目コード | string |
| internal\_item\_code | 自社品目コード | string |
| product\_name | 受注商品名称 | string |
| quantity | 受注数 | number |
| unit | 単位 | string |
| notes | 受注記事 | string |

## **5\. 開発環境の起動手順**

### **前提条件**

* Docker がインストールされていること。
* コンテナ起動時に読み込む環境変数を `data/.env` に記述していること。
  * `cp .env.example data/.env` でひな形をコピーし、値を編集してください。
* ユーザーアップロード、ジョブ履歴、応答履歴を保存するホスト側ディレクトリを用意し、コンテナの`/data`としてバインドすること。
  * `data` 配下には `uploads`、`jobs`、`responses` ディレクトリを作成してください。必要に応じて `.env` で `UPLOAD_DIR` などのパスを上書きできます。

**.env ファイルの例:**

GEMINI\_API\_KEY="YOUR\_GEMINI\_API\_KEY"

### **起動コマンド**

プロジェクトのルートディレクトリで以下のコマンドを実行します。

\# ビルド
docker build -t westa-ocr .

\# 永続化ディレクトリの作成
mkdir -p data/uploads data/jobs data/responses

\# コンテナ起動
docker run \
  -v $(pwd)/data:/data \
  -p 5000:5000 \
  --name westa-ocr \
  westa-ocr

起動後、Webブラウザで http://localhost:5000 にアクセスしてください。

アップロードされたファイルはコンテナ内の`/data/uploads`に保存され、バインドマウントされた`data/uploads`ディレクトリから参照できます。`data/.env` に記述した環境変数もコンテナ起動時に自動で読み込まれます。

### **停止・削除コマンド**

docker stop westa-ocr

docker rm westa-ocr
