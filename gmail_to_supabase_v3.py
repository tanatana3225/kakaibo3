"""
Step 2改訂版: 正規表現でまず抽出を試み、失敗したときだけClaudeにフォールバックする。

事前準備は gmail_to_supabase.py と同じ。
"""

import base64
import json
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from supabase import create_client

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# 決済方法ごとに送信元アドレスが違うので、送信元でどのパーサー・ラベルを使うか判定する。
# 「口座引落し」と「デビット」はユーザー側では区別不要なので、同じ payment_type にまとめている。
SENDER_CONFIG = {
    "statement@vpass.ne.jp": {"payment_type": "クレジット", "parser": "olive_style"},
    "smbc-debit@smbc-card.com": {"payment_type": "デビット・引き落とし", "parser": "olive_style"},
    "SMBC_service@dn.smbc.co.jp": {"payment_type": "デビット・引き落とし", "parser": "account_withdrawal"},
}

SEARCH_QUERY = "(" + " OR ".join(f"from:{addr}" for addr in SENDER_CONFIG) + ") newer_than:7d"

# カテゴリはここで自由に増減できる。AIはこの中から必ず1つを選ぶ。
CATEGORY_LIST = ["食費", "交通費", "買い物", "交際費", "サブスク", "光熱費", "医療費", "その他"]

_categorize_client: Anthropic | None = None


def categorize_with_ai(merchant: str) -> str:
    """店舗名から、CATEGORY_LISTの中のどれに該当するかをAIに判定させる。
    分類専用なので、安価なHaikuモデルを使う。
    """
    global _categorize_client

    if not merchant:
        return "その他"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  → ANTHROPIC_API_KEYが未設定のため、カテゴリ分類をスキップ（その他に設定）")
        return "その他"

    if _categorize_client is None:
        _categorize_client = Anthropic(api_key=api_key)

    try:
        response = _categorize_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            system=(
                "あなたは家計簿の店舗名をカテゴリに分類するアシスタントです。"
                f"次のカテゴリの中から、最も当てはまるものを1つだけ選んで、そのカテゴリ名だけを返してください。"
                f"他の言葉は一切含めないでください。\n\nカテゴリ一覧: {', '.join(CATEGORY_LIST)}"
            ),
            messages=[{"role": "user", "content": merchant}],
        )
        category = response.content[0].text.strip()
        if category in CATEGORY_LIST:
            return category
        print(f"  → AIの分類結果「{category}」が一覧にないため「その他」にします")
        return "その他"
    except Exception as e:
        print(f"  → カテゴリ分類でエラー: {e}。「その他」にします")
        return "その他"


def parse_olive_style(text: str) -> dict | None:
    """statement@vpass.ne.jp / smbc-debit@smbc-card.com 共通のテンプレート
    例:
    ◇利用日：2026/08/16 14:23
    ◇利用先：BUSHIKAKUNAVI
    ◇利用取引：買物
    ◇利用金額：5,544円
    """
    date_match = re.search(r"◇利用日\s*[：:]\s*([\d/]+\s+[\d:]+)", text)
    merchant_match = re.search(r"◇利用先\s*[：:]\s*(.+)", text)
    amount_match = re.search(r"◇利用金額\s*[：:]\s*([\d,]+)\s*円", text)

    if not (date_match and merchant_match and amount_match):
        # 「利用日」の記載自体がない = 購入通知ではない別種のメール（お支払い金額のお知らせ等）
        return None

    return {
        "used_at": date_match.group(1).replace("/", "-"),
        "amount": int(amount_match.group(1).replace(",", "")),
        "merchant": merchant_match.group(1).strip(),
    }


def parse_account_withdrawal(text: str) -> dict | None:
    """SMBC_service@dn.smbc.co.jp のテンプレート
    例: 出金日：2026年08月05日 / 出金額：5,000円 / 内容：PAYPAY
    """
    date_match = re.search(r"出金日\s*[：:]\s*(\d+)年(\d+)月(\d+)日", text)
    amount_match = re.search(r"出金額\s*[：:]\s*([\d,]+)\s*円", text)
    merchant_match = re.search(r"内容\s*[：:]\s*(.+)", text)

    if not (date_match and amount_match and merchant_match):
        return None

    year, month, day = date_match.groups()
    used_at = f"{year}-{int(month):02d}-{int(day):02d}"  # 例: "2026-08-09"

    return {
        "used_at": used_at,
        "amount": int(amount_match.group(1).replace(",", "")),
        "merchant": re.sub(r"\s+", " ", merchant_match.group(1)).strip(),
    }


# 送信元アドレスの設定と対応付ける、本文パーサーの実体
PARSER_FUNCTIONS = {
    "olive_style": parse_olive_style,
    "account_withdrawal": parse_account_withdrawal,
}


def parse_known_sender(sender_address: str, text: str) -> dict | None:
    """送信元アドレスから payment_type と使うべきパーサーを決め、本文を解析する。"""
    config = SENDER_CONFIG.get(sender_address)
    if not config:
        return None  # 未登録の送信元 → AIフォールバックへ

    parser_fn = PARSER_FUNCTIONS[config["parser"]]
    extracted = parser_fn(text)
    if not extracted:
        return None  # テンプレートは分かっているのに本文が読み取れなかった → AIフォールバックへ

    extracted["payment_type"] = config["payment_type"]
    extracted["category"] = categorize_with_ai(extracted["merchant"])
    return extracted


AI_FALLBACK_SYSTEM_PROMPT = """あなたは日本のクレジットカード・銀行の利用通知メールを解析するアシスタントです。
渡されたメール本文から、以下の情報をJSON形式で抽出してください。出力はJSONオブジェクト1つだけ、前置き不要。

{{
  "used_at": "利用日時",
  "amount": 利用金額を整数で,
  "merchant": "利用先・店舗名",
  "payment_type": "クレジット / デビット / 口座引落し のいずれか。不明なら「不明」",
  "category": "次のいずれか: {categories}"
}}

読み取れない項目はnull。カード利用通知でなければ {{"error": "not_a_transaction_email"}} とだけ返す。
""".format(categories=", ".join(CATEGORY_LIST))


def parse_with_ai_fallback(raw_text: str, known_payment_type: str | None = None) -> dict:
    """未登録の送信元、または既知の送信元だがフォーマットが変わったメールをAIに解析させる。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "AIフォールバックが必要ですが、ANTHROPIC_API_KEYが.envに設定されていません。"
        )

    print("  → Claudeで解析中...")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        system=AI_FALLBACK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": raw_text}],
    )
    text = response.content[0].text.strip().replace("```json", "").replace("```", "")
    parsed = json.loads(text)

    # 送信元アドレスから決済方法が確定している場合は、AIの推測より優先して上書きする
    if known_payment_type and not parsed.get("error"):
        parsed["payment_type"] = known_payment_type

    return parsed


def parse_email(sender_address: str, raw_text: str) -> dict | None:
    config = SENDER_CONFIG.get(sender_address)

    if config:
        result = parse_known_sender(sender_address, raw_text)
        if result:
            print(f"  → 正規表現で解析成功（送信元: {sender_address}）")
            return result
        print("  → 登録済みの送信元だが正規表現にマッチせず。AIにフォールバックします")
        return parse_with_ai_fallback(raw_text, known_payment_type=config["payment_type"])

    print(f"  → 未登録の送信元（{sender_address}）。AIにフォールバックします")
    return parse_with_ai_fallback(raw_text)


def get_gmail_service():
    creds = None

    # GitHub Actions上で動く場合、環境変数からtoken.jsonの中身が渡ってくる。
    # ローカルの.envではなく、GitHub Secretsの値がそのまま入っている想定。
    token_json_content = os.environ.get("GOOGLE_TOKEN_JSON")

    if token_json_content:
        creds = Credentials.from_authorized_user_info(json.loads(token_json_content), SCOPES)
    elif os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # ローカル実行時のみ、更新されたトークンをファイルに保存しておく
            if not token_json_content:
                with open("token.json", "w") as token_file:
                    token_file.write(creds.to_json())
        else:
            # ここに来るのは「まだ一度もログインしたことがない」ローカル初回実行の時だけのはず。
            # GitHub Actions上でここに来た場合は、token.jsonの中身が古すぎる/間違っている可能性が高い。
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token_file:
                token_file.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def get_plain_text_body(payload):
    if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
    if payload.get("mimeType") == "text/html" and "data" in payload.get("body", {}):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
    for part in payload.get("parts", []):
        result = get_plain_text_body(part)
        if result:
            return result
    return None


def save_to_supabase(supabase_client, gmail_message_id: str, raw_text: str, parsed: dict):
    if parsed.get("error"):
        print(f"  → スキップ: {parsed['error']}")
        return

    row = {
        "gmail_message_id": gmail_message_id,
        "used_at": parsed.get("used_at"),
        "amount": parsed.get("amount"),
        "merchant": parsed.get("merchant"),
        "payment_type": parsed.get("payment_type"),
        "category": parsed.get("category"),
        "raw_email": raw_text,
    }
    supabase_client.table("transactions").upsert(row, on_conflict="gmail_message_id").execute()
    print(f"  → 保存: {parsed.get('merchant')} ¥{parsed.get('amount')} [{parsed.get('category')}]")


def get_already_processed_ids(supabase_client) -> set:
    """既にSupabaseに保存済みのgmail_message_idを取得しておく。
    これにより、同じメールを毎回AIやDBに送り直す無駄を防ぐ。
    """
    result = supabase_client.table("transactions").select("gmail_message_id").execute()
    return {row["gmail_message_id"] for row in result.data if row["gmail_message_id"]}


def main():
    gmail_service = get_gmail_service()
    supabase_client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])

    already_processed = get_already_processed_ids(supabase_client)
    print(f"既に処理済み: {len(already_processed)}件")

    print(f"検索条件: {SEARCH_QUERY}")
    results = gmail_service.users().messages().list(
        userId="me", q=SEARCH_QUERY, maxResults=50
    ).execute()
    messages = results.get("messages", [])

    if not messages:
        print("該当するメールが見つかりませんでした。")
        return

    # 処理済みのものはこの時点で除外しておく（Gmail本文取得やAI解析を一切させない）
    new_messages = [m for m in messages if m["id"] not in already_processed]
    print(f"{len(messages)}件中、未処理の{len(new_messages)}件を処理します。\n")

    if not new_messages:
        print("新しいメールはありませんでした。")
        return

    for msg_meta in new_messages:
        msg = gmail_service.users().messages().get(
            userId="me", id=msg_meta["id"], format="full"
        ).execute()

        body = get_plain_text_body(msg["payload"])
        if not body:
            print(f"[{msg_meta['id']}] 本文が取得できずスキップ")
            continue

        headers = msg["payload"].get("headers", [])
        from_header = next((h["value"] for h in headers if h["name"] == "From"), "")
        # "田中様 <statement@vpass.ne.jp>" のような表記からアドレス部分だけ取り出す
        address_match = re.search(r"[\w.+-]+@[\w.-]+", from_header)
        sender_address = address_match.group(0) if address_match else ""

        print(f"[{msg_meta['id']}] 処理中... (差出人: {sender_address})")
        try:
            parsed = parse_email(sender_address, body)
        except json.JSONDecodeError:
            print("  → AIの応答をJSONとして解釈できませんでした。スキップ")
            continue
        except Exception as e:
            print(f"  → 解析に失敗しました: {e}")
            print("  → このメールの本文（デバッグ用）:")
            print("  " + "-" * 40)
            print(body)
            print("  " + "-" * 40)
            continue

        if parsed is None:
            continue

        save_to_supabase(supabase_client, msg_meta["id"], body, parsed)

    print("\n完了。SupabaseのTable Editorで確認してみてください。")


if __name__ == "__main__":
    main()
