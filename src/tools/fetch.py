import html2text  # HTMLをプレーンテキストに変換するライブラリ
import requests  # HTTPリクエストを扱うためのライブラリ
from langchain_core.tools import tool  # LangChainのデコレーター、ツールとして関数を登録
from pydantic import BaseModel, Field  # Pydanticを用いたデータ検証
from readability import Document  # HTMLドキュメントを読みやすくするライブラリ

TIMEOUT_ERROR_DICT = {
    "status": 500,
    "page_content": {
        "error_message": "タイムアウトによりページが取得できませんでした。他のページを試してください。"
    },
}

PAGE_ANALYSIS_ERROR_DICT = {
    "status": 500,
    "page_content": {
        "error_message": "ページの解析に失敗しました。他のページを試してください。"
    },
}


class FetchDDGPageInput(BaseModel):
    url: str = Field()  # 取得するウェブページのURL
    page_num: int = Field(
        default=0, description="取得するページ番号。0以上の整数で指定します。"
    )


@tool(args_schema=FetchDDGPageInput)
def fetch_ddg_page(url, page_num=0, timeout_sec=10):
    """
    指定されたURLからウェブページのコンテンツを取得するツール。
    取得したコンテンツは、タイトルと本文に分けて返されます。
    返されるデータは、状態コード、ページ内容（タイトル、内容、次ページ有無）を含みます。

    Parameters
    ----------
    url : str
        取得するページのURL。
    page_num : int, optional
        取得するページ番号、デフォルトは0。
    timeout_sec : int, optional
        リクエストのタイムアウト秒数、デフォルトは10秒。

    Returns
    -------
    Dict[str, Any]:
        - status : int
            HTTPステータスコードまたは内部エラーコード。
        - page_content : dict
            - title : str
                ページのタイトル。
            - content : str
                ページの内容（指定されたページ番号のチャンク）。
            - has_next : bool
                次のページ番号が存在するかどうか。
    """
    try:
        response = requests.get(url, timeout=timeout_sec)  # URLからコンテンツを取得
        response.encoding = "utf-8"  # エンコーディングをUTF-8に設定
    except requests.exceptions.Timeout:
        return TIMEOUT_ERROR_DICT

    if response.status_code != 200:
        return {
            "status": response.status_code,
            "page_content": {
                "error_message": "ページが取得できませんでした。他のページを試してください。"
            },
        }

    try:
        doc = Document(response.text)  # レスポンスからDocumentオブジェクトを生成
        title = doc.title()  # ページのタイトルを取得
        html_content = doc.summary()  # 読みやすい要約を生成
        content = html2text.html2text(html_content)  # HTMLをテキストに変換

        return {
            "status": response.status_code,
            "page_content": {
                "title": title,
                "content": content[page_num * 3000 : (page_num + 1) * 3000],
                "has_next": False,
            },
        }
    except:
        return PAGE_ANALYSIS_ERROR_DICT
