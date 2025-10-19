from dotenv import load_dotenv
load_dotenv()

stress_template = """
あなたは親の育児ストレスを軽減するための専門家です。
育児疲れやストレス管理に関する実践的なアドバイスを提供します。
親自身の心身の健康を保つための方法を教えます。

質問：{input}
"""

nutrition_template = """
あなたは子どもの栄養に詳しいアドバイザーです。
子どもの健康な発育を支える食事や栄養バランスについてアドバイスを提供します。
食事の習慣や偏食に関する質問にも丁寧に答えます。

質問：{input}
"""

sleep_template = """
あなたは子どもの睡眠習慣に詳しい専門家です。
子どもの夜泣きや睡眠不足に関する解決策を提供し、健全な睡眠を促すためのアドバイスを行います。
親が子どもの睡眠問題に対処できるようサポートします。

質問：{input}
"""

balance_template = """
あなたは働く親のための育児と仕事の両立に詳しいアドバイザーです。
仕事と育児のバランスを保つための実践的なアドバイスを提供し、時間管理や家族とのコミュニケーションをサポートします。

質問：{input}
"""

management_template = """
あなたは経営に詳しいアドバイザーです。
企業経営に役立つ情報を提供し、サポートします。

質問：{input}
"""
prompt_infos = [
    {
        "name": "stress",
        "description": "親の育児ストレスを軽減するための専門家です",
        "prompt_template": stress_template
    },
    {
        "name": "nutrition",
        "description": "子どもの栄養に詳しい専門家です",
        "prompt_template": nutrition_template
    },
    {
        "name": "sleep",
        "description": "子どもの睡眠習慣に詳しい専門家です",
        "prompt_template": sleep_template
    },
    {
        "name": "balance",
        "description": "働く親のための育児と仕事の両立に詳しい専門家です",
        "prompt_template": balance_template
    },
    {
        "name": "management",
        "description": "経営の専門家です",
        "prompt_template": management_template
    },
]
import os
import streamlit as st

# langchain 関連の import を安全に試みる
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    _LANGCHAIN_AVAILABLE = True
except Exception as _e:
    # 実行環境に langchain 等がない場合は None をセットし、後で親切なエラーを出す
    ChatOpenAI = None
    PromptTemplate = None
    LLMChain = None
    _LANGCHAIN_AVAILABLE = False
    _LANGCHAIN_IMPORT_ERROR = _e
    # langchain が利用できない場合は後続処理で明示的にエラーを出す


# Streamlit 実行時に langchain が無い場合は親切なメッセージを出して終了
if "__name__" in globals() and (__name__ == "__main__" or "streamlit" in os.environ.get("SERVER_SOFTWARE", "")):
    if not _LANGCHAIN_AVAILABLE:
        st.write("# 必要なパッケージが見つかりません")
        st.error(
            "このアプリを実行するには `langchain` と関連パッケージが必要です。\n"
            "`requirements.txt` を確認し、`pip install -r requirements.txt` を実行してください。\n"
            f"詳細: {_LANGCHAIN_IMPORT_ERROR}"
        )
        # ここで以降の UI をレンダリングしないように終了
        st.stop()


# --- LLM 初期化ヘルパー ---
def get_llm():
    """LangChain の ChatOpenAI インスタンスを返す。インポートエラーや API キー未設定時は例外を送出する。

    Streamlit Community Cloud を使う場合は、`OPENAI_API_KEY` をシークレットに設定してください。
    """
    if not _LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "langchain パッケージのインポートに失敗しました。`requirements.txt` に langchain 等があるか確認し、\n"
            f"インストールエラー: {_LANGCHAIN_IMPORT_ERROR}"
        )

    # 環境変数にキーがない場合は Streamlit Secrets を試す
    if "OPENAI_API_KEY" not in os.environ:
        try:
            openai_key = st.secrets.get("OPENAI_API_KEY")
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
        except Exception:
            # 取得できない場合は実行時に ChatOpenAI がエラーを出す
            pass

    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
def ask_expert(input_text: str, expert_name: str) -> str:
    """入力テキストと専門家名を受け取り、該当するプロンプトテンプレートで LLM に問い合わせて回答を返す。

    Args:
        input_text: ユーザー入力のテキスト
        expert_name: prompt_infos に定義した name のいずれか

    Returns:
        LLM の応答テキスト
    """
    llm = get_llm()

    # 該当するテンプレートを探す
    selected = None
    for p in prompt_infos:
        if p["name"] == expert_name:
            selected = p
            break

    if selected is None:
        # デフォルトのプロンプトを用いる
        prompt = PromptTemplate(input_variables=["input"], template="{input}")
    else:
        prompt = PromptTemplate(template=selected["prompt_template"], input_variables=["input"])

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input_text)
    return result


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="育児アドバイザー", layout="centered")
    st.title("育児／経営アドバイザーチャット")

    st.markdown("選択した専門家になりきって質問に回答します。OpenAI の API キーは環境変数 `OPENAI_API_KEY` または Streamlit Secrets に設定してください。")

    expert_names = [p["name"] for p in prompt_infos]
    expert_labels = {p["name"]: p["description"] for p in prompt_infos}

    # ラジオで専門家選択
    choice = st.radio("専門家を選択してください", options=expert_names, format_func=lambda x: f"{x} — {expert_labels.get(x, '')}")

    user_input = st.text_area("質問を入力してください", height=150)

    if st.button("送信"):
        if not user_input.strip():
            st.warning("質問を入力してください。")
        else:
            with st.spinner("回答を取得しています..."):
                try:
                    answer = ask_expert(user_input, choice)
                    st.subheader("回答")
                    st.write(answer)
                except Exception as e:
                    st.error(f"LLM への問い合わせ中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()