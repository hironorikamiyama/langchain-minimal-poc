
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたはPythonとKubernetesに詳しい技術学習アシスタントです。"
                "回答は日本語で、簡潔に説明してください。",
            ),
            ("human", "{question}"),
        ]
    )


def main() -> None:
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEYが設定されていません")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        temperature=0,
    )

    chain = build_prompt() | model

    result = chain.invoke(
        {"question": "KubernetesのServiceの役割を3行で説明してください。"}
    )
    print(result.text)


if __name__ == "__main__":
    main()
