
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    document_id: int
    document: str
    similarity: float
    accepted: bool


def normalize(vector: list[float] | np.ndarray) -> np.ndarray:
    """ベクトルをL2正規化する。"""
    arr = np.asarray(vector, dtype=np.float32)

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("1次元の非空ベクトルが必要です")

    if not np.all(np.isfinite(arr)):
        raise ValueError("ベクトルに非有限値が含まれています")

    norm = np.linalg.norm(arr)

    if norm == 0:
        raise ValueError("ゼロベクトルは正規化できません")

    return arr / norm


def retrieve_top1(
    query_vector: list[float] | np.ndarray,
    document_vectors: list[list[float] | np.ndarray],
    documents: list[str],
    threshold: float = 0.65,
) -> RetrievalResult:
    """正規化内積でTop-1を選択し、閾値判定する。"""
    if not documents or len(document_vectors) != len(documents):
        raise ValueError("文書とベクトルの件数が不正です")

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("閾値は-1～1で指定してください")

    query = normalize(query_vector)
    vectors = [normalize(vector) for vector in document_vectors]

    if any(vector.shape != query.shape for vector in vectors):
        raise ValueError("Embeddingの次元数が一致しません")

    similarities = [float(np.dot(query, vector)) for vector in vectors]

    index = int(np.argmax(similarities))
    score = similarities[index]

    return RetrievalResult(
        document_id=index + 1,
        document=documents[index],
        similarity=score,
        accepted=score >= threshold,
    )
