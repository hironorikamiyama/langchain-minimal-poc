
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from src.rag_comparison.retrieval import retrieve_top1


EMBEDDING_MODEL = "gemini-embedding-001"
TARGET_DIM = 768
SIMILARITY_THRESHOLD = 0.65

DOCUMENTS = [
    "開発チームのリモートワーク上限は週3日までとする。月曜と金曜は原則出社日とする。",
    "書籍購入費は月額5,000円まで事前申請なしで経費精算できる。5,000円を超える場合は事前の承認が必要。",
    "本番環境へのデプロイは火曜日と木曜日の午前10時にのみ実施可能。緊急デプロイにはテックリードの承認が必要。",
    "社内ツールのパスワードは90日ごとに更新が必要であり、16文字以上の英数記号混ざりで設定しなければならない。",
]

QUERIES = [
    "本番環境へのデプロイは何曜日の何時にできますか？",
    "5000円を超える技術書を買いたい場合、先に何かする必要がありますか？",
    "有給休暇は年間何日取得できますか？",
]


def build_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "以下の参照情報のみに基づいて回答してください。"
                "参照情報に明確な答えがない場合は、"
                "「参照情報からは回答できません」とだけ回答してください。\n"
                "参照情報:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )


def main():
    load_dotenv()

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("Gemini APIキーが設定されていません")

    # 比較基準のrag_poc.pyではEmbeddingのtask_typeは未指定。
    # LangChain側も明示的にUNSPECIFIEDを使用する。
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=TARGET_DIM,
    )

    # 元のrag_poc.pyに合わせて生成モデルを設定。
    model = ChatGoogleGenerativeAI(
        model=os.getenv("RAG_LLM_MODEL", "gemini-3.6-flash")
    )

    chain = build_prompt() | model

    print("文書をEmbeddingしています...")
    document_vectors = embeddings.embed_documents(
        DOCUMENTS,
        task_type="SEMANTIC_SIMILARITY",
    )

    for question in QUERIES:
        print("=" * 60)
        print(f"質問: {question}")

        query_vector = embeddings.embed_query(
            question,
            task_type="SEMANTIC_SIMILARITY",
        )

        result = retrieve_top1(
            query_vector=query_vector,
            document_vectors=document_vectors,
            documents=DOCUMENTS,
            threshold=SIMILARITY_THRESHOLD,
        )

        print(
            f"Top-1: Doc {result.document_id}, "
            f"Score: {result.similarity:.4f}"
        )

        if not result.accepted:
            print("検索層の判定: REJECT")
            print("LLM呼び出し: スキップ")
            continue

        print("検索層の判定: ACCEPT")

        response = chain.invoke(
            {
                "context": result.document,
                "question": question,
            }
        )

        print(f"LLMの回答: {response.text}")


if __name__ == "__main__":
    main()
