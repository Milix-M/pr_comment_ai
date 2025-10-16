# PR_Comment_AI

自己PR文を作成するAIシステム。
志望する会社の情報をtool callingを使用し調べ、ユーザーの持つ特徴をMergeすることで、クォリティの高い自己PRが作成可能.

## 実行方法（Linux）

1. `echo "OPENROUTER_API_KEY=<your openrouter key>" > .env`
1. `python -m vevn venv`
1. `. venv/bin/activate`
1. `pip install -r requirements.txt`
1. `streamlit run main.py`