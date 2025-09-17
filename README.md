# westa-ocr

## 1. 概要

本プロジェクトは、GoogleのGemini APIを活用して、PDFファイルから情報を抽出し、構造化データとして出力する業務用Webアプリケーションです。
ユーザーはPDFをアップロードするだけで、請求書や報告書などのドキュメントから必要なデータを自動で抽出し、JSONやCSV形式でダウンロードできます。



## 2. システム構成

- **フロントエンド**: Vue.js 3
- **バックエンド**: Flask (Python)
- **UIフレームワーク**: Bootstrap 5
- **AIモデル**: Google Gemini API (Gemini 1.5 Proなど)
- **主要ライブラリ**:
    - **Python**: `google-generativeai`, `flask`, `PyPDF2`, `pillow`, `concurrent.futures`
    - **JavaScript**: `vue`, `axios`, `bootstrap`

## 3. 機能要件

### 3.1. PDFアップロード機能
- [ ] ユーザーは単一のPDFファイルをWebインターフェース経由でアップロードできる。
- [ ] ドラッグ＆ドロップによるアップロードに対応する。

### 3.2. OCR処理機能
- [ ] バックエンドはアップロードされたPDFをページごとに画像へ変換する。
- [ ] **並列処理**: 各ページを個別のタスクとして、並列でGemini APIへ送信し処理時間を短縮する。
- [ ] **構造化データ抽出**: Gemini APIの**Structured Output**機能を活用し、プロンプトで定義されたJSONスキーマに基づき、高精度なデータ抽出を行う。
- [ ] 処理中は、ユーザーに進捗状況が分かるようなUI（スピナー等）を表示する。

### 3.3. 結果の表示とダウンロード
- [ ] ページごとに抽出されたデータを画面上のテーブルに一覧表示する。
- [ ] 全ページの処理結果をまとめた単一のファイルを生成する。
- [ ] **ダウンロード形式**: ユーザーは抽出結果を**JSON**または**CSV**形式でダウンロードできる。

## 4. 画面設計

画面は単一ページで構成され、以下の要素を含みます。
1.  **ファイル選択エリア**: PDFをアップロードするためのエリア。
2.  **プロンプト設定エリア(任意)**: 抽出したいデータの形式を指示するテキストエリア。
3.  **実行ボタン**: OCR処理を開始するボタン。
4.  **結果表示エリア**: 抽出されたデータがテーブル形式で表示される。
5.  **ダウンロードボタン**: 結果をファイルとして保存する。

詳細は後述の「画面サンプル」を参照。

## 5. API設計 (Gemini API)

### プロンプト例
以下のようなプロンプトとレスポンススキーマを定義して、Gemini APIに送信する。

**プロンプト**:
```
この請求書の画像から、以下の情報をJSON形式で抽出してください。
- invoice_id: 請求書番号 (文字列)
- issue_date: 発行日 (YYYY-MM-DD形式の文字列)
- total_amount: 合計金額 (数値)
- items: 品目リスト (各品目はnameとpriceを持つオブジェクトの配列)
```

**レスポンススキーマ (JSON)**:
```json
{
  "invoice_id": "string",
  "issue_date": "string",
  "total_amount": "number",
  "items": [
    {
      "name": "string",
      "price": "number"
    }
  ]
}
```

## 6. 開発環境構築

### バックエンド (Flask)
```bash
# 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 必要なライブラリのインストール
pip install flask google-generativeai PyPDF2 pillow python-dotenv

# 環境変数の設定
echo "GEMINI_API_KEY='YOUR_API_KEY'" > .env

# サーバーの起動
flask run
```

### フロントエンド (Vue.js)
```bash
# プロジェクトの作成 (初回のみ)
npm create vue@latest

# 依存関係のインストール
cd <project-name>
npm install

# 開発サーバーの起動
npm run dev
```
