from itertools import islice

from ddgs import DDGS
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SearchDDGInput(BaseModel):
    query: str = Field(description="検索したいキーワードを入力してください")


@tool(args_schema=SearchDDGInput)
def search_ddg(query, max_result_num=5):
    """
    DuckDuckGo検索を行うツールです。
    指定されたクエリでDuckDuckGoを使ってウェブ検索を行い、結果を返します。
    返される情報には、ページのタイトル、概要（スニペット）、そしてURLが含まれます。

    Parameters
    ----------
    query : str
        検索を行うためのクエリ文字列。
    max_result_num : int
        返される結果の最大数。

    Returns
    -------
    List[Dict[str, str]]:
        検索結果のリスト。各辞書オブジェクトには以下が含まれます。
        - title: タイトル
        - snippet: ページの概要
        - url: ページのURL

    この関数は、プログラミングに関連する質問など、特定の質問に最適な言語で検索を行うことを推奨します。
    また、検索結果だけでは十分でない場合は、実際のページ内容を取得するために追加のツールを使用することをお勧めします。
    """

    # [1] Web検索を実施
    res = DDGS().text(query, region="jp-jp", safesearch="off", backend="lite")

    # [2] 結果のリストを分解して戻す
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        }
        for r in islice(res, max_result_num)
    ]
