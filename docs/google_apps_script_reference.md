# Google Apps Script リファレンス

本リポジトリのリレーサーバーと連携する Google Apps Script（GAS）のコードを参考情報として保存します。

```javascript
/*
 * 以下のコードは、AppSheet からトリガーされる注文書 PDF の処理および
 * リレー経由での Gemeni 呼び出しに対応する Google Apps Script の実装です。
 * 参考情報として保管しており、本リポジトリから直接実行されることはありません。
 */

/**
 * AppSheetからトリガーされ、未処理の注文書PDFをGemini APIで読み込み、
 * スプレッドシートに注文情報と明細情報を記録する。
 * PDF全体を一度にAPIに渡し、ページごとの解析をLLMに指示する。
 *
 * ※この関数は旧フロー（GAS→Gemini 直呼び）のまま残してあります。
 *   新フローでは ProcessOrder_test_relay を使用してください。
 *
 * @param {string} PROMPT Gemini APIに渡すプロンプト文字列。
 * @param {string} PATTERN マスタフィルタ用のパターン
 */
function ProcessOrder_test(PROMPT, PATTERN) {
  // 定数定義
  const SPREADSHEET_ID = '193JwqYPME-f33UdyfBxEDkCCWimxppnC2CE9L7PrqoQ';
  const SHEET_ORDERS_NAME = '注文一覧'; // 注文一覧シート
  const SHEET_DETAILS_NAME = '明細一覧'; // 明細一覧シート
  const SHEET_SHIPMASTER_NAME = '納入場所マスタ'; //納入場所マスタシート
  const SHEET_ITEMMASTER_NAME = '品目マスタ'; //品目マスタシート
  const API_KEY = PropertiesService.getScriptProperties().getProperty('GEMINI_API_KEY');

  // スプレッドシートとシートを取得
  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  const sheetOrders = spreadsheet.getSheetByName(SHEET_ORDERS_NAME);
  const sheetDetails = spreadsheet.getSheetByName(SHEET_DETAILS_NAME);
  const sheetShipMaster = spreadsheet.getSheetByName(SHEET_SHIPMASTER_NAME);
  const sheetItemMaster = spreadsheet.getSheetByName(SHEET_ITEMMASTER_NAME);

  // ヘッダー行から各列のインデックスを取得
  const ordersData = sheetOrders.getDataRange().getValues();
  const header = ordersData[0];
  const orderIdIndex = header.indexOf("注文書ID");
  const pdfUrlIndex = header.indexOf("注文書ファイル");
  const statusIndex = header.indexOf("処理ステータス");
  const pageCountIndex = header.indexOf("ページ数");

  // マスタから対象パターンでフィルタした納入場所一覧を取得
  const masterShipData = sheetShipMaster.getDataRange().getValues();
  const masterShipHeader = masterShipData[0];
  const masterShipFiltered = masterShipData.filter((row, idx) => idx > 0 && row[1] === PATTERN);
  const masterShip = [masterShipHeader, ...masterShipFiltered]
    .map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(","))
    .join("\n");

  // マスタから対象パターンでフィルタした品目一覧を取得
  const masterItemData = sheetItemMaster.getDataRange().getValues();
  const masterItemHeader = masterItemData[0];
  const masterItemFiltered = masterItemData.filter((row, idx) => idx > 0 && row[1] === PATTERN);
  const masterItem = [masterItemHeader, ...masterItemFiltered]
    .map(row => row.map(v => `"${String(v).replace(/"/g, '""')}"`).join(","))
    .join("\n");
  
  // "注文一覧"シートの2行目から最終行までループ
  for (let i = 1; i < ordersData.length; i++) {
    const row = ordersData[i];
    const status = row[statusIndex];

    // 処理ステータスが空の場合のみ実行
    if (!status) {
      try {
        const pdfUrl = row[pdfUrlIndex];
        if (!pdfUrl) {
          console.warn(`Row ${i + 1}: PDFのURLが空のためスキップします。`);
          continue;
        }
        
        const orderId = row[orderIdIndex]; // 親となる注文書IDを取得
        const pdfBlob = fetchPdfBlobFromUrl(pdfUrl); // GoogleドライブのURLからPDFファイルを取得

        // PDF全体を一度だけAPIに渡し、ページ毎の解析を指示
        const today = Utilities.formatDate(new Date(), "JST", "yyyyMMdd");
        const replacedPrompt = PROMPT.replace('{current_date}', today);
        const extractedText = callGeminiApiApendMaster(pdfBlob, replacedPrompt, masterShip, masterItem, API_KEY);
        
        // ページ毎に構造化されたテキストを解析する
        const allPagesData = parseMultiPageDataFromLLM(extractedText);

        // AIが解析したページ数を「注文一覧」シートに書き込む
        const processedPageCount = allPagesData.filter(p => !p.isNonOrderPage).length; // 注文書ではないページはカウントしない
        if (pageCountIndex > -1) {
          sheetOrders.getRange(i + 1, pageCountIndex + 1).setValue(processedPageCount);
        }

        // 解析されたページデータごとにループ
        allPagesData.forEach(pageData => {
          if (pageData.isNonOrderPage) {
            return; // 注文書ではないページはスキップ
          }

          const orderDetails = pageData["注文明細"] || [];

          // 解析した注文明細を「明細一覧」シートに書き込む
          orderDetails.forEach(detail => {
            // 【変更後】のカラム定義に合わせて新しい行を作成
            const newRow = [
              generateRandomID(),         // 明細ID (乱数)
              orderId,                    // 注文書ID (親ID)
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
              detail["受注記事"] || "",
            ];
            sheetDetails.appendRow(newRow);
          });
        });

        // 処理が成功したらステータスを更新
        sheetOrders.getRange(i + 1, statusIndex + 1).setValue("読込完了");

      } catch (error) {
        console.error(`行 ${i + 1} の処理中にエラーが発生しました: ${error.toString()}`);
        sheetOrders.getRange(i + 1, statusIndex + 1).setValue(`エラー: ${error.message}`);
      }
    }
  }
}


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
function ProcessOrder_test_relay(PROMPT, PATTERN) {
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

      const resp = UrlFetchApp.fetch(RELAY_URL, options);
      const code = resp.getResponseCode();
      const text = resp.getContentText();
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
 * リレー→GASのHTTPヘッダーに Authorization: Bearer WEBHOOK_TOKEN を付与する想定。
 */
function doPost(e) {
  try {
    // Bearerトークン検証
    const ok = verifyBearerTokenForWebhook_(e);
    if (!ok) {
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
    console.error('doPost error:', err.stack || err);
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

  // 冪等チェック
  const orderId = payload.orderId;
  const pageIndex = payload.pageIndex;
  const isNonOrderPage = !!payload.isNonOrderPage;
  const idemKey = `PAGE_RECEIVED_${orderId}_${pageIndex}`;

  const props = PropertiesService.getScriptProperties();
  if (props.getProperty(idemKey) === '1') {
    // 既に処理済み
    return ContentService.createTextOutput(JSON.stringify({ ok: true, duplicate: true }))
      .setMimeType(ContentService.MimeType.JSON);
  }

  if (isNonOrderPage) {
    // 注文書ではないページ → 既読フラグだけ付与
    props.setProperty(idemKey, '1');
    return ContentService.createTextOutput(JSON.stringify({ ok: true, skipped: true }))
      .setMimeType(ContentService.MimeType.JSON);
  }

  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  const sheetDetails = spreadsheet.getSheetByName(SHEET_DETAILS_NAME);

  const rawText = payload.rawText || '';
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
        detail["受注記事"] || "",
      ];
      sheetDetails.appendRow(newRow);
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
 * WebhookのBearerトークン検証
 * - 受信ヘッダー Authorization: Bearer <token> を検査
 * - スクリプトプロパティ WEBHOOK_TOKEN と一致したらOK
 */
function verifyBearerTokenForWebhook_(e) {
  const WEBHOOK_TOKEN = PropertiesService.getScriptProperties().getProperty('WEBHOOK_TOKEN');
  if (!WEBHOOK_TOKEN) {
    console.error('WEBHOOK_TOKEN not set.');
    return false;
  }
  const auth =
    (e && e.parameter && e.parameter['Authorization']) ||
    (e && e.headers && (e.headers['Authorization'] || e.headers['authorization'])) ||
    '';
  if (!auth || !auth.toString().startsWith('Bearer ')) return false;
  const token = auth.toString().substring('Bearer '.length).trim();
  return token && token === WEBHOOK_TOKEN;
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
        if (columns.length >= 14) {
          detailRows.push({
            "受注伝票番号":     columns[0] || "",
            "納入場所":         columns[1] || "",
            "得意先":           columns[2] || "",
            "得意先注文番号":   columns[3] || "",
            "受注日":           columns[4] || "",
            "出荷予定日":       columns[5] || "",
            "顧客納期":         columns[6] || "",
            "得意先品目コード": columns[7] || "",
            "自社品目コード":   columns[8] || "",
            "受注商品名称":     columns[9] || "",
            "受注数":           columns[10] || "",
            "単位":             columns[11] || "",
            "受注単価":         columns[12] || "",
            "受注記事":         columns[13] || "",
          });
        } else {
          console.log(`列数が14未満のため、この明細行をスキップしました: ${row}`);
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
 * Google ドライブのファイルURLからファイルIDを抽出し、PDFのBlobオブジェクトを取得する
 * （旧フロー用に残置。新フローでは extractDriveFileId のみ使用）
 */
function fetchPdfBlobFromUrl(fileUrl) {
  const fileIdMatch = fileUrl.match(/[-\w]{25,}/);
  if (!fileIdMatch) {
    throw new Error(`無効なGoogleドライブURLです: ${fileUrl}`);
  }
  const fileId = fileIdMatch[0];
  const file = DriveApp.getFileById(fileId);
  return file.getBlob();
}


/**
 * Gemini APIを呼び出し、PDFの内容をテキストとして抽出する（旧フロー用に残置）
 */
function callGeminiApiApendMaster(pdfBlob, prompt, masterShip, masterItem, apiKey) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
  const pdfData = Utilities.base64Encode(pdfBlob.getBytes());

  const payload = {
    contents: [
      {
        parts: [
        { inline_data: { mime_type: 'application/pdf', data: pdfData } },
        { text: prompt + "\n\n" + masterShip  + "\n\n" + masterItem }
        ]
      }
    ],
    generationConfig: {
      temperature: 0.1,
      topP: 0.9,
      maxOutputTokens: 65536,
      response_mime_type: "text/plain",
    }
  };

  const options = {
    method: 'post',
    contentType: 'application/json',
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  };

  console.log("リクエスト開始");
  const response = UrlFetchApp.fetch(url, options);
  console.log(response);
  const responseCode = response.getResponseCode();
  const responseBody = response.getContentText();

  if (responseCode === 200) {
    const parsedResponse = JSON.parse(responseBody);
    if (parsedResponse.candidates && parsedResponse.candidates[0].content && parsedResponse.candidates[0].content.parts.length > 0) {
      console.log(parsedResponse);
      return parsedResponse.candidates[0].content.parts[0].text;
    } else {
      console.error("API Response Body:", responseBody);
      throw new Error("APIからのレスポンスにテキストが含まれていません。");
    }
  } else {
    console.error("API Error Response:", responseBody);
    throw new Error(`APIリクエストエラー: ${responseCode}. Response: ${responseBody}`);
  }
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
```
