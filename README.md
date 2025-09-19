# westa-ocr

**目的**: GAS/AppSheet からの大容量 OCR ジョブを**非同期化**し、Gemini API への呼び出しを**1ページ単位**で実行。結果は **Webhook** で GAS (`parseMultiPageDataFromLLM`) へ通知する。  
**前提**: 単一の Docker コンテナで動作。**永続化データは `/data`**（ホストへバインドマウント）に保存。

---

## 1. コンポーネント概要

- **HTTP API（/jobs 他）**: GAS からジョブを登録。即 ACK を返す。
- **ワーカー（同一プロセス内）**: キューから 1 ジョブずつ取り出して順次処理。Google Drive から PDF を取得し、ページごとに Gemini を呼び出し、結果を Webhook へ転送する。
- **永続ストレージ**: `/data` に SQLite / ジョブ状態 / 一時ファイル（必要に応じて）を保存。

> ※ 現状の実装は 1 ワーカーで順次処理。将来的な並列化やレート制御は環境変数で切り替えられるよう拡張予定。

---

## 2. データフロー（時系列）

```
GAS(ProcessOrder_test)
   └─ POST /jobs  ───────────────▶  Relay API（即時ACK）
                                  └─ (enqueue job)
Worker（同一コンテナ内）
   └─ Google Drive から PDF 取得（fileId 認証付き）
   └─ PDF をページ単位に分割（PDF or 画像）
   └─ 各ページを Gemini へ送信
   └─ ページごとに Webhook POST（PAGE_RESULT）
   └─ 全ページ終了で Webhook POST（JOB_SUMMARY）
GAS(doPost) で検証・保存・集計
```

### 2.1 ジョブ処理ロジック（GAS リクエスト受付〜Webhook 返信）

1. **GAS からのリクエスト受信**: `POST /jobs` に `orderId` や `fileId`、マスタ CSV、Webhook 設定が届く。Bearer トークンを検証し、SQLite へ永続化したうえでジョブ ID を払い出す。
2. **Google Drive から PDF 取得**: バックグラウンドワーカーが `fileId` を取り出し、サービスアカウントのアクセストークンを用いて Drive API (`files.get?alt=media`) からバイナリをダウンロードする。HTTP/HTTPS や `file://` にも対応。
3. **ページ単位へ分割**: 取得した PDF を `split_pdf` でページごとに切り出す。ページ番号は **1 から** の連番で管理する。
4. **Gemini へ問い合わせ**: 各ページを `GeminiClient.generate` で送信。プロンプトにマスタ CSV（`shipCsv` / `itemCsv`）を追記し、Gemini から得たテキストとメタ情報を保存する。API キーが未設定の場合はシミュレーションレスポンスを返す。
5. **Webhook へ転送**: `WebhookDispatcher.send` がページごとの結果 (`event=PAGE_RESULT`) を GAS の Webhook（Apps Script doPost）へ POST。Bearer トークンをヘッダに付与し、HTTP 302 にも追従する。
6. **完了通知**: 全ページ処理後に `event=JOB_SUMMARY` を送信。処理済み枚数・スキップ数・ページ単位のエラー情報、および最終的なジョブステータス（`DONE` / `ERROR`）をまとめて通知する。

---

## 3. 受け取る情報（GAS → Relay）

### 3.1 エンドポイント: `POST /jobs`

- **目的**: ジョブ登録（即時 200 ACK）。
- **認証**: `Authorization: Bearer <RELAY_TOKEN>` のみ。
- **Content-Type**: `application/json`

#### リクエストボディ（厳密仕様）
```jsonc
{
  "orderId": "string (必須)",              // GASの「注文書ID」。冪等性キーの片割れ
  "fileId": "string (必須)",               // Google Drive のファイルID（本体は送らない）
  "prompt": "string (必須)",               // {current_date} 展開済みのプロンプト
  "pattern": "string (任意)",              // マスタのフィルタ条件など
  "masters": {
    "shipCsv": "string (必須, CSV)",       // 納入場所マスタ（ヘッダ含むCSV）
    "itemCsv": "string (必須, CSV)"        // 品目マスタ（ヘッダ含むCSV）
  },
  "webhook": {
    "url": "https://script.google.com/... (必須)",
    "token": "string (必須)"                 // Relay→GAS の Bearer トークン（空文字不可）
  },
  "gemini": {
    "model": "string (既定: gemini-2.5-flash)",
    "temperature": 0.1,
    "topP": 0.9,
    "topK": 32,
    "maxOutputTokens": 65536
  },
  "options": {
    "splitMode": "pdf|image",              // 既定: pdf。現状 image は未対応
    "dpi": 300,                            // splitMode=image のとき有効
    "concurrency": 3                       // 予約フィールド（現状は未使用）
  },
  "idempotencyKey": "string (任意)"        // 無指定時は server が orderId を採用
}
```

#### レスポンス（200）
```json
{ "job_id": "job_2025-09-18_00001", "correlation_id": "abc123", "status": "RECEIVED" }
```

#### エラーレスポンス（例）
```json
{ "error": { "code": "INVALID_ARGUMENT", "message": "fileId is required" } }
```

> **備考**: `fileId` にアクセスするため、**サービスアカウント**へ対象フォルダ/ファイルの **Viewer** 権限を付与しておくこと。

---

## 4. 渡す情報（Relay → GAS Webhook）

### 4.1 共通ヘッダ
- `Content-Type`: `application/json`
- `Authorization`: `Bearer <WEBHOOK_TOKEN>` （設定時）

> **備考**: Google Apps Script の `doPost(e)` では `Authorization` ヘッダーを直接参照できないため、
> ペイロード本体にも `token` フィールドを含めて同じ値を渡します。

### 4.2 ページごとの結果: `event=PAGE_RESULT`
```jsonc
{
  "event": "PAGE_RESULT",
  "jobId": "job_2025-09-18_00001",
  "orderId": "abc123",
  "pageIndex": 3,                          // 1-based でも 0-based でも可。以下の規約に従う
  "isNonOrderPage": false,
  "rawText": "--- PAGE 3 ---\n【得意先】...\n【注文明細】...",
  "meta": {
    "model": "gemini-2.5-flash",
    "durationMs": 8420,
    "tokensInput": 0,
    "tokensOutput": 0
  },
  "idempotencyKey": "abc123:3",            // orderId:pageIndex 形式を推奨（pageIndex は 1 始まり）
  "token": "<WEBHOOK_TOKEN>"               // Apps Script での検証用（ヘッダーと同値）
}
```

### 4.3 ジョブ完了サマリ: `event=JOB_SUMMARY`
```jsonc
{
  "event": "JOB_SUMMARY",
  "jobId": "job_2025-09-18_00001",
  "orderId": "abc123",
  "totalPages": 7,
  "processedPages": 6,                      // isNonOrderPage を除く
  "skippedPages": 1,
  "errors": [],                             // ページ単位の失敗やタイムアウトがあれば配列で返す
  "status": "DONE",                        // 任意。ジョブの最終ステータス（DONE / ERROR）
  "token": "<WEBHOOK_TOKEN>"               // Apps Script での検証用（ヘッダーと同値）
}
```

> **ページ番号規約**: `pageIndex` は **1-based**。UI 表示と内部記録を一致させたい場合はこの値をそのまま利用する。

---

## 5. 永続化モデル（SQLite）

### 5.1 `jobs` テーブル
| column            | type    | note                                                            |
|-------------------|---------|-----------------------------------------------------------------|
| job_id            | TEXT PK | `job_{UTCタイムスタンプ}_{ランダム英数字}`                       |
| order_id          | TEXT    | GAS 側の注文書 ID                                                |
| file_id           | TEXT    | Drive ファイル ID / URL / `file://` など                         |
| prompt            | TEXT    | Gemini へ渡すプロンプト                                          |
| pattern           | TEXT    | フィルタ条件。未指定可                                           |
| masters_json      | TEXT    | `shipCsv` / `itemCsv` を格納した JSON                            |
| webhook_url       | TEXT    | GAS Webhook URL                                                 |
| webhook_token     | TEXT    | GAS Webhook 用 Bearer トークン                                   |
| gemini_json       | TEXT    | Gemini 設定（任意）                                              |
| options_json      | TEXT    | `splitMode` など追加オプション（任意）                           |
| idempotency_key   | TEXT    | `orderId` など。UNIQUE 制約                                      |
| status            | TEXT    | `RECEIVED` / `ENQUEUED` / `PROCESSING` / `DONE` / `ERROR`         |
| last_error        | TEXT    | 最終エラー概要（任意）                                           |
| total_pages       | INTEGER | 総ページ数（処理後に更新）                                       |
| processed_pages   | INTEGER | Webhook 送信まで成功したページ数                                 |
| skipped_pages     | INTEGER | エラー等でスキップしたページ数                                   |
| created_at        | TEXT    | ISO8601（`datetime('now')`）                                      |
| updated_at        | TEXT    | ISO8601（`datetime('now')`）                                      |

### 5.2 `job_pages` テーブル
| column            | type    | note                                                   |
|-------------------|---------|--------------------------------------------------------|
| job_id            | TEXT    | FK → `jobs.job_id`                                     |
| page_index        | INTEGER | 1 から始まるページ番号                                 |
| status            | TEXT    | `DONE` / `ERROR`                                        |
| is_non_order_page | INTEGER | 現状は常に 0（非注文書ページ検出は未実装）             |
| raw_text          | TEXT    | Gemini からの生テキスト                                |
| error             | TEXT    | 失敗時のメッセージ                                     |
| meta_json         | TEXT    | Gemini メタデータ（JSON 文字列）                       |
| created_at        | TEXT    | ISO8601                                                 |
| updated_at        | TEXT    | ISO8601                                                 |

`JobRepository` がすべての読み書きを担い、マルチスレッドでも安全なように `RLock` で直列化している。起動時には `RECEIVED` / `ENQUEUED` / `PROCESSING` のジョブを再キューイングし、処理を継続可能な状態に戻す。

---

## 6. 主要モジュールと外部サービス

- **FileFetcher**: `http(s)://`、`file://`、`local:`、および Google Drive のファイル ID に対応。Drive 利用時はサービスアカウント認証（`DRIVE_SERVICE_ACCOUNT_JSON`）を読み込み、必要に応じてトークンをリフレッシュする。
- **GeminiClient**: `models/{model}:generateContent` を呼び出し、ページ PDF・プロンプト・マスタ CSV を 1 リクエストにまとめる。API キー未設定時はシミュレーションレスポンスを返すので、ローカル検証が容易。
- **WebhookDispatcher**: Apps Script など 302 を返すエンドポイントにも対応できるよう `follow_redirects=True` で POST。Bearer トークンを自動付与し、失敗時は例外で呼び出し元に通知する。
- **JobWorker**: 1 スレッドで順次処理し、ページ結果を都度 SQLite と Webhook に記録。途中失敗時は `_handle_initial_failure` でサマリを送信し、ジョブを `ERROR` として終了させる。

---

## 7. 環境変数

`app.settings.get_settings()` が読み込む主な変数は以下の通り。

| env                        | default            | 備考 |
|---------------------------|--------------------|------|
| RELAY_TOKEN               | (必須)             | GAS→Relay 認証用 Bearer トークン |
| DATA_DIR                  | `/data`            | SQLite などのベースディレクトリ |
| SQLITE_PATH               | `/data/relay.db`   | SQLite ファイルパス |
| TMP_DIR                   | `/data/tmp`        | 予約（現状未使用） |
| WORKER_IDLE_SLEEP         | `1.0`              | ワーカーがキュー待機するときの sleep 秒数 |
| GEMINI_API_KEY            | なし               | 未設定だとシミュレーション動作 |
| GEMINI_MODEL              | `gemini-2.5-flash` | 既定モデル名 |
| WEBHOOK_TIMEOUT           | `30.0`             | Webhook POST タイムアウト（秒） |
| REQUEST_TIMEOUT           | `60.0`             | Drive / Gemini への HTTP タイムアウト（秒） |
| LOG_LEVEL                 | `INFO`             | Python ロガーのレベル |
| DRIVE_SERVICE_ACCOUNT_JSON| なし               | Drive 認証情報のパス。設定時は `FileFetcher` が利用 |

---

## 8. API エンドポイント詳細

### 8.1 `POST /jobs`
- Bearer 認証必須。
- `JobRequest` のバリデーションに失敗すると 400。`webhook.token` が空文字の場合も 400。
- 成功時は `JobResponse` を返し、`status` は常に `RECEIVED`。
- `idempotencyKey` が重複していた場合は既存ジョブをそのまま返す。

### 8.2 `GET /jobs/{job_id}`
- Bearer 認証必須。
- `JobDetail` を返却。例:
```json
{
  "jobId": "job_20250201T120000_abcd12",
  "orderId": "abc123",
  "status": "PROCESSING",
  "fileId": "1AbCdEf...",
  "prompt": "...",
  "pattern": null,
  "masters": {"shipCsv": "...", "itemCsv": "..."},
  "webhookUrl": "https://script.google.com/...",
  "webhookToken": "******",
  "createdAt": "2025-02-01T12:00:00",
  "updatedAt": "2025-02-01T12:00:05",
  "totalPages": 7,
  "processedPages": 6,
  "skippedPages": 1,
  "lastError": null,
  "pages": [
    {"pageIndex": 1, "status": "DONE", "isNonOrderPage": false, "rawText": "...", "error": null, "meta": {"model": "gemini-2.5-flash"}}
  ]
}
```

### 8.3 `GET /healthz`
- 無認証。単に `{ "status": "ok" }` を返す。

---

## 9. ワーカー処理の擬似コード

```
on POST /jobs:
  authenticate()
  validate(payload)
  job_id = generate_id()
  insert_job(status=RECEIVED)
  mark_enqueued(job_id)
  enqueue(job_id)
  return JobResponse(job_id, correlation_id=orderId, status=RECEIVED)

worker_loop():
  while running:
    job_id = queue.get(timeout=idle_sleep)
    row = repository.get_job(job_id)
    if row is None or row.status not in {RECEIVED, ENQUEUED, PROCESSING}:
      continue
    repository.update_job_status(PROCESSING)
    try:
      file_bytes = file_fetcher.fetch(file_id)
      pages = split_pdf(file_bytes)
    except Exception as exc:
      handle_initial_failure(job_row, exc)
      continue
    for page in pages:
      try:
        result = gemini.generate(...)
        repository.record_page_result(status="DONE", raw_text=result.text)
        webhook_dispatcher.send(PAGE_RESULT)
      except Exception as exc:
        repository.record_page_result(status="ERROR", error=str(exc))
        collect_error(exc)
    repository.update_job_counters(...)
    send_summary_and_finalize()
```

---

## 10. 運用メモ

- Docker 実行例:
  ```bash
  docker run -d --name relay \
    -p 5000:5000 \
    -e RELAY_TOKEN="REPLACE_ME" \
    -e GEMINI_API_KEY="REPLACE_ME" \
    -e DRIVE_SERVICE_ACCOUNT_JSON="/data/sa.json" \
    -v /opt/relay/data:/data \
    -v /opt/relay/sa.json:/data/sa.json:ro \
    relay-image:latest
  ```
  SQLite を含む `/data` を必ず永続化すること。
- 確認ポイント:
  - `POST /jobs` が 200 を返し、レスポンスの `job_id` を取得できること。
  - Drive 上のファイル権限が適切で、ワーカーが PDF を取得できること。
  - GAS Webhook で `PAGE_RESULT` / `JOB_SUMMARY` が到達すること。
  - ジョブの途中で例外が発生した場合でも、`JOB_SUMMARY` が `status=ERROR` で送信されること。

---

## 11. GAS との連携メモ

- 送信例（Bearer トークン必須）
  ```js
  const body = {
    orderId,
    fileId,
    prompt: replacedPrompt,
    pattern: PATTERN,
    masters: { shipCsv, itemCsv },
    webhook: { url: WEBHOOK_URL, token: WEBHOOK_TOKEN },
    gemini: { model: "gemini-2.5-flash", temperature: 0.1, topP: 0.9 },
    options: { splitMode: "pdf" }
  };
  UrlFetchApp.fetch(RELAY_URL, {
    method: "post",
    contentType: "application/json",
    headers: { Authorization: `Bearer ${RELAY_TOKEN}` },
    payload: JSON.stringify(body)
  });
  ```
- 受信側（GAS Webhook）は `event` に応じて処理を分岐し、`idempotencyKey` で重複排除できる。

---

以上が現状の実装と README の整合内容。
