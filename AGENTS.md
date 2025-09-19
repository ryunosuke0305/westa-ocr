# AGENTS 記述

このリポジトリに含まれる主なファイルおよびディレクトリとその役割を以下にまとめる。

## ルートディレクトリ
- `Dockerfile`: アプリケーションをコンテナ化するための設定を定義する。
- `README.md`: プロジェクトの概要やセットアップ手順を説明するドキュメント。
- `cloudbuild.yaml`: Google Cloud Build 用のビルド構成ファイル。
- `requirements.txt`: Python 依存パッケージの一覧。
- `app/`: アプリケーション本体のソースコード。
- `docs/`: プロジェクトに関する追加ドキュメント。
- `tests/`: 自動テストコード。

## `app/` ディレクトリ配下
- `__init__.py`: `app` パッケージを初期化する空ファイル。
- `admin.py`: 管理系処理（例: スタッフ向け機能）のエンドポイントやロジックを定義。
- `auth.py`: 認証・認可に関するユーティリティやロジックを提供。
- `file_fetcher.py`: 外部ストレージからのファイル取得処理を担当。
- `gemini.py`: Gemini 関連の API 連携や処理を実装。
- `logging_config.py`: ログ設定を定義するモジュール。
- `main.py`: FastAPI アプリケーションのエントリーポイント。
- `models.py`: Pydantic モデルやデータ構造を定義。
- `pdf_utils.py`: PDF 処理に関するユーティリティ関数。
- `repository.py`: データアクセス層の処理をまとめたリポジトリ実装。
- `settings.py`: 環境変数などの設定値を管理。
- `templates/`: テンプレートファイルを格納。
- `webhook.py`: Webhook 関連のエンドポイントおよび処理。
- `worker.py`: 非同期ジョブやバックグラウンド処理を実装。

## `docs/` ディレクトリ配下
- `google_apps_script_reference.md`: Google Apps Script 連携に関するリファレンスドキュメント。

## `tests/` ディレクトリ配下
- `test_file_fetcher.py`: `file_fetcher` モジュールのテスト。
- `test_repository.py`: `repository` モジュールのテスト。

---
**運用ポリシー**: 新しいファイルやディレクトリを追加した場合、また既存の役割に変更が生じた場合は、本 `AGENTS.md` を必ず更新して最新の構成情報を反映すること。
