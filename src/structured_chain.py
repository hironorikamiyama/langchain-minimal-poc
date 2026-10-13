
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class TechnicalExplanation(BaseModel):
    topic: str = Field(
        min_length=1,
        description="技術トピックの名称",
    )

    summary: str = Field(
        min_length=1,
        max_length=100,
        description="100文字以内の概要",
    )

    key_points: list[str] = Field(
        min_length=3,
        max_length=3,
        description="重要なポイント3つ",
    )



def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "技術について日本語で説明してください。"
                "summaryは100文字以内、key_pointsは必ず3つとしてください。",
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
    )

    structured_model = model.with_structured_output(
        TechnicalExplanation,
        method="json_schema",
    )

    chain = build_prompt() | structured_model

    result = chain.invoke(
        {"question": "KubernetesのServiceについて説明してください。"}
    )

    if result is None:
        raise RuntimeError("構造化出力を取得できませんでした")

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
