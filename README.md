# westa-ocr

**目的**: GAS/AppSheet からの大容量 OCR ジョブを**非同期化**し、Gemini API への呼び出しを**1ページ単位**で実行。結果は **Webhook** で GAS (`parseMultiPageDataFromLLM`) へ通知する。  
**前提**: 単一の Docker コンテナで動作。**永続化データは `/data`**（ホストへバインドマウント）に保存。

---

## 1. コンポーネント概要

- **HTTP API（/jobs 他）**: GAS からジョブを登録。即 ACK を返す。
- **ワーカー（同一プロセス内）**: ジョブキューをポーリングし、PDF 取得 → ページ分割 → Gemini 呼び出し → Webhook 送信。
- **永続ストレージ**: `/data` に SQLite / ジョブ状態 / 一時ファイル（必要に応じて）を保存。

> 並列数（concurrency）、レート制御、リトライは環境変数で調整可能。

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
    "url": "https://script.google.com/... (必須)"
  },
  "gemini": {
    "model": "string (既定: gemini-2.5-flash)",
    "temperature": 0.1,
    "topP": 0.9,
    "maxOutputTokens": 65536
  },
  "options": {
    "splitMode": "pdf|image",              // 既定: pdf（ページPDFのまま送る）
    "dpi": 300,                            // splitMode=image のとき有効
    "concurrency": 3                       // ページ並列数（環境変数で上限）
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
  "idempotencyKey": "abc123:3"             // orderId:pageIndex 形式を推奨
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
  "errors": []                              // ページ単位の失敗やタイムアウトがあれば配列で返す
}
```

> **ページ番号規約**: `pageIndex` は **0-based** を推奨（内部配列と一致）。UI表示が1-basedなら GAS 側で +1。

---

## 5. 内部モデル（永続化; `/data` 配下）

### 5.1 SQLite ファイル
- パス: `/data/relay.db`（バインドマウント先に作成）

#### テーブル: `jobs`
| column        | type      | note                                               |
|---------------|-----------|----------------------------------------------------|
| job_id        | TEXT PK   | `job_yyyy-mm-dd_nnnnn`                             |
| order_id      | TEXT      | GAS の注文書ID                                     |
| file_id       | TEXT      | Drive fileId                                       |
| status        | TEXT      | RECEIVED / PROCESSING / DONE / ERROR               |
| total_pages   | INTEGER   | 検出総ページ数                                     |
| created_at    | INTEGER   | unix seconds                                       |
| updated_at    | INTEGER   | unix seconds                                       |

#### テーブル: `pages`
| column        | type      | note                                               |
|---------------|-----------|----------------------------------------------------|
| job_id        | TEXT      | FK -> jobs.job_id                                  |
| page_index    | INTEGER   | 0-based                                            |
| status        | TEXT      | QUEUED / RUNNING / DONE / ERROR                    |
| retries       | INTEGER   | 既定 0                                             |
| raw_text      | TEXT      | Geminiの結果（text/plain）                         |
| is_non_order  | INTEGER   | 0/1                                                |
| error         | TEXT      | 失敗時の短縮メッセージ                             |
| updated_at    | INTEGER   | unix seconds                                       |
| PRIMARY KEY(job_id, page_index) |           | 冪等性確保                           |

#### テーブル: `events`（オプション）
- Webhook送信ログ（重複対策＆監査）

### 5.2 一時ファイル
- `/data/tmp/<job_id>/` にページPDF/PNGを一時保存（必要時のみ）。  
- ジョブ完了後に削除（`RETENTION_SECS` を超えたらクリーンアップ）。

---

## 6. 外部連携

### 6.1 Google Drive 取得
- サービスアカウントで Drive API `files.get` を使用。  
- PDF バイト列を取得し、ページ分解。権限は**対象フォルダ共有**で最小化。

### 6.2 Gemini API 呼び出し（ページ単位）
- URL 例: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...`
- `parts`:
  - `inline_data`: `application/pdf`（1ページPDF）または `image/png`（画像化した1枚）
  - `text`: `prompt + "\n\n" + shipCsv + "\n\n" + itemCsv`
- 429/5xx は指数バックオフ。`maxRetries`・`initialBackoffMs` 環境変数で調整。

---

## 7. セキュリティ

- **入力認証（GAS→Relay）**: Bearer Token のみ。
- **冪等性**: `idempotencyKey`（`orderId:pageIndex`）で重複処理を抑止。
- **秘密情報**: `GEMINI_API_KEY` は**コンテナ環境変数**で注入し、ログに出さない。

---

## 8. 環境変数（コンテナ）

| env                          | default             | 説明 |
|-----------------------------|---------------------|------|
| PORT                        | 5000                | HTTP待受 |
| RELAY_TOKEN                 | (必須)              | GAS→Relay の Bearer |
| WEBHOOK_USER_AGENT          | relay/1.0           | 監査用 UA |
| GEMINI_API_KEY              | (必須)              | Gemini 認証 |
| GEMINI_MODEL                | gemini-2.5-flash    | 既定モデル |
| MAX_CONCURRENCY             | 3                   | ページ並列上限 |
| RATE_LIMIT_RPS              | 5                   | 1秒あたり送信上限 |
| BACKOFF_MAX_RETRIES         | 5                   | リトライ回数 |
| BACKOFF_INITIAL_MS          | 1000                | 初期待機(ms) |
| SQLITE_PATH                 | /data/relay.db      | 永続 DB |
| TMP_DIR                     | /data/tmp           | 一時ファイル |
| RETENTION_SECS              | 600                 | 一時ファイル保持 |
| DRIVE_SERVICE_ACCOUNT_JSON  | /data/sa.json       | サービスアカウント鍵（マウント） |

> `DRIVE_SERVICE_ACCOUNT_JSON` は `/data` に配置（読み取り権限のみ）。

---

## 9. API 仕様（詳細）

### 9.1 `POST /jobs`
- バリデーション: 必須項目（`orderId`, `fileId`, `masters.shipCsv`, `masters.itemCsv`, `webhook.url`, 認証）。
- 正常系: 200 + `job_id` を返して、**非同期で処理開始**。
- 代表的 4xx:
  - 400: 必須フィールド欠落 / 正規表現違反
  - 401/403: 認証失敗
- 代表的 5xx:
  - 502: Drive/Gemini 一時障害（内部リトライ後でも失敗）

### 9.2 `GET /jobs/{job_id}`（任意）
- レスポンス例
```json
{
  "job_id": "job_2025-09-18_00001",
  "order_id": "abc123",
  "status": "PROCESSING",
  "total_pages": 7,
  "done": 4,
  "error": 0
}
```

### 9.3 `POST /healthz`
- 200 OK（DB疎通・キュー深さを返すと尚良）

---

## 10. 処理アルゴリズム（擬似コード）

```
on POST /jobs (req):
  assert auth ok
  validate body
  job_id = newJobId()
  upsert jobs(status=RECEIVED)
  enqueue(job_id)
  return {200, job_id, correlation_id=orderId}

workerLoop():
  while true:
    job = dequeue()
    if not job: sleep(500ms); continue
    try:
      set job.status=PROCESSING
      bytes = driveFetch(fileId)
      pages = splitPdf(bytes or images)
      total = len(pages)
      for i, page in enumerate(pages):
        with rate_limit & semaphore:
          try:
            resp = callGemini(page, prompt, masters)
            upsert pages(status=DONE, raw_text=resp, is_non_order=detectNonOrder(resp))
            postWebhook(PAGE_RESULT, i, resp)
          except e:
            upsert pages(status=ERROR, error=short(e))
            retry with backoff (<= BACKOFF_MAX_RETRIES)
      summarize = calcSummary()
      postWebhook(JOB_SUMMARY, summarize)
      set job.status=DONE
    except e:
      set job.status=ERROR
      postWebhook(JOB_SUMMARY, errors=[short(e)])
```

---

## 11. ロギングと監視

- すべてのログに `jobId`, `orderId`, `pageIndex` を含める。
- 重要イベント（受信、Drive取得、ページ化、Gemini送信、Webhook送信、リトライ、完了）。
- メトリクス（任意）: 1ページ処理時間、429件数、平均リトライ回数。

---

## 12. Docker 実行例

**ホスト側**に以下を用意:
- `/opt/relay/data` … 永続ディレクトリ（空で可）
- `/opt/relay/sa.json` … Service Account キー（Drive用）

```bash
docker run -d --name relay \
  -p 5000:5000 \
  -e RELAY_TOKEN="REPLACE_ME" \
  -e GEMINI_API_KEY="REPLACE_ME" \
  -e GEMINI_MODEL="gemini-2.5-flash" \
  -e SQLITE_PATH="/data/relay.db" \
  -e TMP_DIR="/data/tmp" \
  -e DRIVE_SERVICE_ACCOUNT_JSON="/data/sa.json" \
  -v /opt/relay/data:/data \
  -v /opt/relay/sa.json:/data/sa.json:ro \
  relay-image:latest
```

> **注意**: 永続化は **必ず `/data` にバインド**。ログを stdout/stderr に出し、ホスト側で収集。

---

## 13. テスト・検証チェックリスト

- [ ] `POST /jobs` で 200 / job_id を受領できる
- [ ] Drive から `fileId` を取得できる（権限OK）
- [ ] ページ分割が正しい（ページ数一致）
- [ ] `PAGE_RESULT` が GAS に届き、`parseMultiPageDataFromLLM` で解析可能
- [ ] `JOB_SUMMARY` で「注文一覧」のページ数・ステータスが更新される
- [ ] 429/5xx リトライが機能
- [ ] `/data` に DB/一時ファイルが生成・削除される

---

## 14. GAS 側インターフェース（参考: 送信と受信）

### 送信（GAS → Relay）
```js
const body = {
  orderId,
  fileId,                   // Drive URL から抽出
  prompt: replacedPrompt,   // {current_date} 埋め込み後
  pattern: PATTERN,
  masters: { shipCsv, itemCsv },
  webhook: { url: GAS_WEBAPP_URL },
  gemini: { model: "gemini-2.5-flash", temperature: 0.1, topP: 0.9, maxOutputTokens: 65536 },
  options: { splitMode: "pdf", concurrency: 3 }
};
UrlFetchApp.fetch(RELAY_URL, {
  method: "post",
  contentType: "application/json",
  headers: { Authorization: `Bearer ${RELAY_TOKEN}` },
  payload: JSON.stringify(body)
});
```

### 受信（Relay → GAS Webhook）
- `event=PAGE_RESULT` を受け取ったら `parseMultiPageDataFromLLM(rawText)` を呼び、
  「明細一覧」に append。`orderId:pageIndex` をキーに冪等チェック。  
- `event=JOB_SUMMARY` で「注文一覧」のページ数・ステータス更新。

---

## 15. 非機能要件

- **可用性**: 単一コンテナだが、**プロセス内ワーカー**はクラッシュ耐性あり（未送信Webhookは再試行）。
- **性能**: 1 ジョブあたり並列 3 ページ想定、平均 5–15 秒/ページ（モデル・ページ密度依存）。
- **保守性**: 設定は環境変数化、コードは API 層とワーカー層を分離。

---

以上。単一コンテナ・/data 永続の要件で実運用に必要な仕様を網羅しています。
