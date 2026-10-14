
import numpy as np
import pytest

from src.rag_comparison.retrieval import (
    normalize,
    retrieve_top1,
)


def test_normalize():
    result = normalize([3.0, 4.0])

    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_retrieve_top1_accepted():
    result = retrieve_top1(
        query_vector=[1.0, 0.0],
        document_vectors=[
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        documents=[
            "無関係な文書",
            "関連する文書",
        ],
        threshold=0.65,
    )

    assert result.document_id == 2
    assert result.document == "関連する文書"
    assert result.similarity == pytest.approx(1.0)
    assert result.accepted is True


def test_retrieve_top1_rejected():
    result = retrieve_top1(
        query_vector=[1.0, 0.0],
        document_vectors=[[0.0, 1.0]],
        documents=["無関係な文書"],
        threshold=0.65,
    )

    assert result.similarity == pytest.approx(0.0)
    assert result.accepted is False


def test_reject_dimension_mismatch():
    with pytest.raises(ValueError, match="次元数"):
        retrieve_top1(
            query_vector=[1.0, 0.0],
            document_vectors=[[1.0, 0.0, 0.0]],
            documents=["文書"],
        )


def test_reject_zero_vector():
    with pytest.raises(ValueError, match="ゼロベクトル"):
        normalize([0.0, 0.0])


def test_reject_empty_documents():
    with pytest.raises(ValueError, match="件数"):
        retrieve_top1(
            query_vector=[1.0, 0.0],
            document_vectors=[],
            documents=[],
        )
