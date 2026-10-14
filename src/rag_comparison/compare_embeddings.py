
import os

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.rag_comparison.langchain_rag import DOCUMENTS, QUERIES
from src.rag_comparison.retrieval import retrieve_top1


EMBEDDING_MODEL = "gemini-embedding-001"
TARGET_DIM = 768
TASK_TYPE = "SEMANTIC_SIMILARITY"
THRESHOLD = 0.65


def create_clients():
    load_dotenv()

    # 両実装で同一のAPIキーを使用する
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEYが設定されていません")

    direct_client = genai.Client(api_key=api_key)

    langchain_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=TARGET_DIM,
    )

    return direct_client, langchain_embeddings


def embed_direct(client, text: str) -> list[float]:
    """Gemini API直接版のEmbeddingを取得する。"""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=TASK_TYPE,
            output_dimensionality=TARGET_DIM,
        ),
    )

    return response.embeddings[0].values


def embed_langchain(embeddings, text: str) -> list[float]:
    """LangChain経由でEmbeddingを取得する。"""
    return embeddings.embed_query(
        text,
        task_type=TASK_TYPE,
    )


def print_result(label: str, result) -> None:
    status = "ACCEPT" if result.accepted else "REJECT"

    print(
        f"{label:<12} "
        f"Doc {result.document_id} "
        f"Score={result.similarity:.6f} "
        f"判定={status}"
    )


def main():
    direct_client, langchain_embeddings = create_clients()

    print("=== 文書Embedding生成 ===")

    # Gemini API直接版
    direct_document_vectors = [
        embed_direct(direct_client, doc)
        for doc in DOCUMENTS
    ]

    # LangChain版
    langchain_document_vectors = langchain_embeddings.embed_documents(
        DOCUMENTS,
        task_type=TASK_TYPE,
    )

    print(
        f"直接版: {len(direct_document_vectors)}件 "
        f"LangChain版: {len(langchain_document_vectors)}件"
    )

    print("\n=== Embeddingベクトル差分 ===")

    for index, (direct, langchain) in enumerate(
        zip(direct_document_vectors, langchain_document_vectors),
        start=1,
    ):
        direct_array = np.asarray(direct, dtype=np.float32)
        langchain_array = np.asarray(langchain, dtype=np.float32)

        if direct_array.shape != langchain_array.shape:
            raise ValueError(
                f"Doc {index}: Embeddingの次元数が一致しません"
            )

        max_diff = float(
            np.max(np.abs(direct_array - langchain_array))
        )

        print(f"Doc {index}: 最大絶対差={max_diff:.8f}")

    print("\n=== 質問ごとの検索結果比較 ===")

    for question in QUERIES:
        print("-" * 70)
        print(f"質問: {question}")

        direct_query_vector = embed_direct(
            direct_client,
            question,
        )

        langchain_query_vector = embed_langchain(
            langchain_embeddings,
            question,
        )

        direct_result = retrieve_top1(
            query_vector=direct_query_vector,
            document_vectors=direct_document_vectors,
            documents=DOCUMENTS,
            threshold=THRESHOLD,
        )

        langchain_result = retrieve_top1(
            query_vector=langchain_query_vector,
            document_vectors=langchain_document_vectors,
            documents=DOCUMENTS,
            threshold=THRESHOLD,
        )

        print_result("Gemini直接", direct_result)
        print_result("LangChain", langchain_result)

        score_difference = abs(
            direct_result.similarity - langchain_result.similarity
        )

        same_document = (
            direct_result.document_id == langchain_result.document_id
        )

        same_decision = (
            direct_result.accepted == langchain_result.accepted
        )

        print(f"類似度の絶対差: {score_difference:.8f}")
        print(f"Top-1一致: {same_document}")
        print(f"判定一致: {same_decision}")


if __name__ == "__main__":
    main()
