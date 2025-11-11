# Google Apps Script リファレンス

本リポジトリのリレーサーバーと連携する Google Apps Script（GAS）のコードを参考情報として保存します。

```javascript
/**
 * AppSheetからトリガーされ、未処理の注文書PDFをGemini APIで読み込み、
 * スプレッドシートに注文情報と明細情報を記録する。
 * PDF全体を一度にAPIに渡し、ページごとの解析をLLMに指示する。
 *
 * @param {string} PROMPT Gemini APIに渡すプロンプト文字列。
 * @param {string} PATTERN マスタフィルタ用のパターン
 */

/* =========================================================
   新フロー（リレー経由）: 既存関数は残し、変更箇所は別名で追加
   ========================================================= */

/**
 * 新フロー（推奨）: Bearerトークンのみで認証
 *  - GASはリレーサーバーに「メタ情報のみ」をPOSTして即終了（ACKを待つだけ）
 *  - ページごとのGemini実行→Webhook通知はリレーが担当
 * @param {string} PROMPT
 * @param {string} PATTERN
 */
function ProcessOrder_relay(PROMPT, PATTERN) {
  const SPREADSHEET_ID = '193JwqYPME-f33UdyfBxEDkCCWimxppnC2CE9L7PrqoQ';
  const SHEET_ORDERS_NAME = '注文一覧';
  const SHEET_SHIPMASTER_NAME = '納入場所マスタ';
  const SHEET_ITEMMASTER_NAME = '品目マスタ';

  const RELAY_URL = PropertiesService.getScriptProperties().getProperty('RELAY_URL'); // 例: https://relay.example.com/jobs
  const RELAY_TOKEN = PropertiesService.getScriptProperties().getProperty('RELAY_TOKEN'); // Bearer（必須）
  const WEBHOOK_URL = PropertiesService.getScriptProperties().getProperty('WEBHOOK_URL'); // このGASのWebアプリURL(/exec)
  const WEBHOOK_TOKEN = PropertiesService.getScriptProperties().getProperty('WEBHOOK_TOKEN'); // リレー→GAS Bearer

  if (!RELAY_URL) throw new Error('RELAY_URL が未設定です（スクリプトプロパティ）。');
  if (!RELAY_TOKEN) throw new Error('RELAY_TOKEN が未設定です（スクリプトプロパティ）。');
  if (!WEBHOOK_URL) throw new Error('WEBHOOK_URL が未設定です（スクリプトプロパティ）。');
  if (!WEBHOOK_TOKEN) throw new Error('WEBHOOK_TOKEN が未設定です（スクリプトプロパティ）。');

  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  const sheetOrders = spreadsheet.getSheetByName(SHEET_ORDERS_NAME);
  const sheetShipMaster = spreadsheet.getSheetByName(SHEET_SHIPMASTER_NAME);
  const sheetItemMaster = spreadsheet.getSheetByName(SHEET_ITEMMASTER_NAME);

  // 注文一覧のヘッダー
  const ordersData = sheetOrders.getDataRange().getValues();
  const header = ordersData[0];
  const orderIdIndex = header.indexOf("注文書ID");
  const pdfUrlIndex = header.indexOf("注文書ファイル");
  const statusIndex = header.indexOf("処理ステータス");

  // マスタCSV生成（PATTERNでフィルタ）
  const masterShipData = sheetShipMaster.getDataRange().getValues();
  const masterShipHeader = masterShipData[0];
  const masterShipFiltered = masterShipData.filter((row, idx) => idx > 0 && row[1] === PATTERN);
  const shipCsv = [masterShipHeader, ...masterShipFiltered]
    .map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(","))
    .join("\n");

  const masterItemData = sheetItemMaster.getDataRange().getValues();
  const masterItemHeader = masterItemData[0];
  const masterItemFiltered = masterItemData.filter((row, idx) => idx > 0 && row[1] === PATTERN);
  const itemCsv = [masterItemHeader, ...masterItemFiltered]
    .map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(","))
    .join("\n");

  for (let i = 1; i < ordersData.length; i++) {
    const row = ordersData[i];
    const status = row[statusIndex];
    if (status) continue; // 空のみ対象

    try {
      const orderId = row[orderIdIndex];
      const pdfUrl = row[pdfUrlIndex];
      if (!pdfUrl) {
        console.warn(`Row ${i + 1}: PDFのURLが空のためスキップします。`);
        continue;
      }

      const fileId = extractDriveFileId(pdfUrl);
      const today = Utilities.formatDate(new Date(), "JST", "yyyyMMdd");
      const replacedPrompt = PROMPT.replace('{current_date}', today);

      const body = {
        orderId,
        fileId,
        prompt: replacedPrompt,
        pattern: PATTERN,
        masters: { shipCsv, itemCsv },
        webhook: { url: WEBHOOK_URL, token: WEBHOOK_TOKEN }, // ← Tokenのみ
        gemini: { model: "gemini-2.5-flash", temperature: 0.1, topP: 0.9, maxOutputTokens: 65536 },
        options: { splitMode: "pdf", concurrency: 3 },
        idempotencyKey: `${orderId}`
      };

      const options = {
        method: 'post',
        contentType: 'application/json',
        payload: JSON.stringify(body),
        headers: makeAuthHeadersForRelayTokenOnly_(RELAY_TOKEN),
        muteHttpExceptions: true
      };

      console.log(RELAY_URL);
      console.log(options);
      const resp = UrlFetchApp.fetch(RELAY_URL, options);
      const code = resp.getResponseCode();
      const text = resp.getContentText();
      console.log(text);
      if (code >= 200 && code < 300) {
        // 受付済みに更新
        sheetOrders.getRange(i + 1, statusIndex + 1).setValue("受付済");
      } else {
        console.error(`Relay error ${code}: ${text}`);
        sheetOrders.getRange(i + 1, statusIndex + 1).setValue(`エラー: Relay ${code}`);
      }
    } catch (err) {
      console.error(`行 ${i + 1} の処理中にエラー: ${err.message}`);
      sheetOrders.getRange(i + 1, statusIndex + 1).setValue(`エラー: ${err.message}`);
    }
  }
}


/**
 * Webhook受け口（Bearerトークン検証のみ）
 * リレーサーバーからの PAGE_RESULT / JOB_SUMMARY を受け取り、
 * 「明細一覧」「注文一覧」を更新する。
 *
 * デプロイ: 「ウェブアプリとして導入」し、WEBHOOK_URL にそのURL(/exec)を設定。
 */
function doPost(e) {
  try {
    // Bearerトークン検証
    const ok = verifyTokenFromBody_(e);
    if (!ok) {
      logToSheet("認証に失敗");
      return ContentService.createTextOutput(JSON.stringify({ error: 'unauthorized' }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const raw = e.postData && e.postData.contents ? e.postData.contents : '';
    const payload = raw ? JSON.parse(raw) : {};
    const event = payload.event;

    if (event === 'PAGE_RESULT') {
      return handlePageResult_(payload);
    } else if (event === 'JOB_SUMMARY') {
      return handleJobSummary_(payload);
    } else {
      return ContentService.createTextOutput(JSON.stringify({ ok: true, ignored: true }))
        .setMimeType(ContentService.MimeType.JSON);
    }
  } catch (err) {
    logToSheet(err.message);
    return ContentService.createTextOutput(JSON.stringify({ error: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}


/**
 * PAGE_RESULT ハンドラ
 * payload: { orderId, pageIndex, isNonOrderPage, rawText, ... }
 */
function handlePageResult_(payload) {
  const SPREADSHEET_ID = '193JwqYPME-f33UdyfBxEDkCCWimxppnC2CE9L7PrqoQ';
  const SHEET_DETAILS_NAME = '明細一覧';

  logToSheet("ページ処理開始");
  logToSheet(payload);
  // 冪等チェック
  const orderId = payload.orderId;
  const pageIndex = payload.pageIndex;
  const isNonOrderPage = !!payload.isNonOrderPage;
  const idemKey = `PAGE_RECEIVED_${orderId}_${pageIndex}`;

  const props = PropertiesService.getScriptProperties();
  if (props.getProperty(idemKey) === '1') {
    // 既に処理済み
    logToSheet("重複したリクエストを検出");
    return ContentService.createTextOutput(JSON.stringify({ ok: true, duplicate: true }))
      .setMimeType(ContentService.MimeType.JSON);
  }

  if (isNonOrderPage) {
    // 注文書ではないページ → 既読フラグだけ付与
    props.setProperty(idemKey, '1');
    logToSheet("注文書ではないページを検出");
    return ContentService.createTextOutput(JSON.stringify({ ok: true, skipped: true }))
      .setMimeType(ContentService.MimeType.JSON);
  }

  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  const sheetDetails = spreadsheet.getSheetByName(SHEET_DETAILS_NAME);

  const rawText = payload.rawText || '';
  logToSheet(rawText);
  const pages = parseMultiPageDataFromLLM(rawText); // 1ページ分でも既存関数を流用可

  pages.forEach(pageData => {
    if (pageData.isNonOrderPage) return;
    const orderDetails = pageData['注文明細'] || [];
    orderDetails.forEach(detail => {
      const newRow = [
        generateRandomID(),                       // 明細ID
        orderId,                                  // 親ID
        detail["受注伝票番号"] || "",
        detail["納入場所"] || "",
        detail["得意先"] || "",
        detail["得意先注文番号"] || "",
        detail["受注日"] || "",
        detail["出荷予定日"] || "",
        detail["顧客納期"] || "",
        detail["得意先品目コード"] || "",
        detail["自社品目コード"] || "",
        detail["受注商品名称"] || "",
        detail["受注数"] || "",
        detail["単位"] || "",
        detail["受注単価"] || "",
        detail["納品書記事"] || "",            // ★ 新規列（受注単価と受注記事の間）
        detail["受注記事"] || "",
      ];
      sheetDetails.appendRow(newRow);
      // 明細一覧の最後列に更新時刻を記録
      const r = sheetDetails.getLastRow();
      const c = sheetDetails.getLastColumn();
      sheetDetails.getRange(r, c)
      .setValue(Utilities.formatDate(
        new Date(), Session.getScriptTimeZone() || 'Asia/Tokyo', 'yyyy/MM/dd HH:mm:ss'
      ));
    });
  });

  props.setProperty(idemKey, '1');
  return ContentService.createTextOutput(JSON.stringify({ ok: true }))
    .setMimeType(ContentService.MimeType.JSON);
}


/**
 * JOB_SUMMARY ハンドラ
 * payload: { orderId, totalPages, processedPages, skippedPages, errors: [] }
 */
function handleJobSummary_(payload) {
  const SPREADSHEET_ID = '193JwqYPME-f33UdyfBxEDkCCWimxppnC2CE9L7PrqoQ';
  const SHEET_ORDERS_NAME = '注文一覧';

  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  const sheetOrders = spreadsheet.getSheetByName(SHEET_ORDERS_NAME);

  const ordersData = sheetOrders.getDataRange().getValues();
  const header = ordersData[0];
  const orderIdIndex = header.indexOf("注文書ID");
  const statusIndex = header.indexOf("処理ステータス");
  const pageCountIndex = header.indexOf("ページ数");

  const orderId = payload.orderId;
  const processedPages = Number(payload.processedPages || 0);
  const errors = payload.errors || [];

  // 注文書ID 行を検索
  for (let i = 1; i < ordersData.length; i++) {
    const row = ordersData[i];
    if (row[orderIdIndex] === orderId) {
      if (pageCountIndex > -1) {
        sheetOrders.getRange(i + 1, pageCountIndex + 1).setValue(processedPages);
      }
      if (errors.length > 0) {
        sheetOrders.getRange(i + 1, statusIndex + 1).setValue(`エラー: ${String(errors[0]).substring(0, 120)}`);
      } else {
        sheetOrders.getRange(i + 1, statusIndex + 1).setValue("読込完了");
      }
      break;
    }
  }

  deletePageReceivedForOrder_(orderId);

  return ContentService.createTextOutput(JSON.stringify({ ok: true }))
    .setMimeType(ContentService.MimeType.JSON);
}


/* ========= 認証（Bearerのみ）/ ヘルパー ========= */

/**
 * リレーへ送るときのヘッダー生成（Bearerのみ）
 */
function makeAuthHeadersForRelayTokenOnly_(RELAY_TOKEN) {
  const headers = {};
  headers['Authorization'] = `Bearer ${RELAY_TOKEN}`;
  return headers;
}

/**
 * Webhookのトークン検証（JSONボディの token を検査）
 * - スクリプトプロパティ WEBHOOK_TOKEN と一致したらOK
 */
function verifyTokenFromBody_(e) {
  const WEBHOOK_TOKEN = PropertiesService.getScriptProperties().getProperty('WEBHOOK_TOKEN');
  if (!WEBHOOK_TOKEN) {
    console.error('WEBHOOK_TOKEN not set.');
    return false;
  }
  try {
    const raw = e && e.postData && e.postData.contents ? e.postData.contents : '';
    if (!raw) return false;
    const payload = JSON.parse(raw);
    const token = payload && payload.token ? String(payload.token) : '';
    return token === WEBHOOK_TOKEN;
  } catch (err) {
    console.error('verifyTokenFromBody_ parse error:', err);
    return false;
  }
}


/**
 * Google ドライブのファイルURLからファイルIDを抽出
 */
function extractDriveFileId(fileUrl) {
  const m = String(fileUrl).match(/[-\w]{25,}/);
  if (!m) throw new Error(`無効なGoogleドライブURLです: ${fileUrl}`);
  return m[0];
}


/* ========= 既存ユーティリティ群（そのまま残す） ========= */

/**
 * Gemini APIからのページ毎に構造化されたレスポンス(テキスト)を解析し、
 * ページごとのデータオブジェクトの配列を返す
 * @param {string} text - Gemini APIからのレスポンス文字列
 * @returns {Array<Object>} 解析されたページごとのデータ配列
 */
function parseMultiPageDataFromLLM(text) {
  const allPagesData = [];
  // "--- PAGE X ---" を区切り文字として、各ページのブロックに分割
  const pageBlocks = text.split(/--- PAGE \d+ ---/).filter(block => block.trim() !== '');

  pageBlocks.forEach(block => {
    const pageResult = {};

    // 注文書ではないページの判定
    if (block.includes("このページは注文書ではありません。")) {
      allPagesData.push({ isNonOrderPage: true });
      return; // 次のブロックへ
    }
    
    // 正規表現でヘッダー項目を抽出
    pageResult["得意先"] = extractUsingRegex(block, /【得意先】(.*?)(?:\n|【|$)/);
    pageResult["受注日"] = extractUsingRegex(block, /【受注日】(.*?)(?:\n|【|$)/);
    pageResult["得意先注文番号"] = extractUsingRegex(block, /【得意先注文番号】(.*?)(?:\n|【|$)/);

    const detailRows = [];
    const detailBlockMatch = block.match(/【注文明細】([\s\S]*)/);
    
    if (detailBlockMatch) {
      const detailBlock = detailBlockMatch[1].trim();
      const rows = detailBlock.split("\n").filter(row => row.trim() !== ''); // 空行を除外

      rows.forEach(row => {
        const columns = row.split(",").map(col => col.trim());
        // ★ 列数チェックを15列に変更（「納品書記事」を追加）
        if (columns.length >= 15) {
          detailRows.push({
            "受注伝票番号":     columns[0]  || "",
            "納入場所":         columns[1]  || "",
            "得意先":           columns[2]  || "",
            "得意先注文番号":   columns[3]  || "",
            "受注日":           columns[4]  || "",
            "出荷予定日":       columns[5]  || "",
            "顧客納期":         columns[6]  || "",
            "得意先品目コード": columns[7]  || "",
            "自社品目コード":   columns[8]  || "",
            "受注商品名称":     columns[9]  || "",
            "受注数":           columns[10] || "",
            "単位":             columns[11] || "",
            "受注単価":         columns[12] || "",
            "納品書記事":       columns[13] || "", // ★ 追加
            "受注記事":         columns[14] || "",
          });
        } else {
          console.log(`列数が15未満のため、この明細行をスキップしました: ${row}`);
        }
      });
    }
    pageResult["注文明細"] = detailRows;

    if (pageResult["得意先"] || detailRows.length > 0) {
      allPagesData.push(pageResult);
    }
  });

  return allPagesData;
}


/**
 * テキストと正規表現を受け取り、マッチした部分文字列を返す
 * @param {string} text - 検索対象のテキスト
 * @param {RegExp} regex - 正規表現
 * @returns {string} マッチした文字列、見つからない場合は空文字
 */
function extractUsingRegex(text, regex) {
  const match = text.match(regex);
  return match && match[1] ? match[1].trim() : "";
}

/**
 * 10桁のランダムな英数字IDを生成する
 */
function generateRandomID() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let randomId = '';
  for (let i = 0; i < 10; i++) {
    randomId += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return randomId;
}

//doPostログ出力用
function logToSheet(logMessage) {
  const SPREADSHEET_ID = '193JwqYPME-f33UdyfBxEDkCCWimxppnC2CE9L7PrqoQ';
  var ss = SpreadsheetApp.openById(SPREADSHEET_ID);
  var logSheet = ss.getSheetByName("log");

  // If the "log" sheet doesn't exist, create it
  if (!logSheet) {
    logSheet = ss.insertSheet("log");
  }

  // Set the locale to Japan Standard Time for consistent timestamping
  var timestamp = Utilities.formatDate(new Date(), Session.getScriptTimeZone(), "yyyy/MM/dd HH:mm:ss");

  // Insert a new row at the very top (row 1)
  logSheet.insertRows(1, 1);

  // Write the timestamp to Column A (A1) and the log message to Column B (B1) of the new top row
  logSheet.getRange("A1").setValue(timestamp);
  logSheet.getRange("B1").setValue(logMessage);
}

function deletePageReceivedForOrder_(orderId) {
  const props = PropertiesService.getScriptProperties();
  const all = props.getProperties();
  let count = 0;
  const prefix = `PAGE_RECEIVED_${orderId}_`;
  for (const k in all) {
    if (k.startsWith(prefix)) {
      props.deleteProperty(k);
      count++;
    }
  }
  logToSheet(`cleanup: deleted ${count} PAGE_RECEIVED_* keys for order ${orderId}`);
}


```
