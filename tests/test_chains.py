
import pytest
from pydantic import ValidationError

from src.basic_chain import build_prompt as basic_prompt
from src.structured_chain import (
    TechnicalExplanation,
    build_prompt as structured_prompt,
)


def test_basic_prompt():
    result = basic_prompt().invoke(
        {"question": "Serviceとは何ですか？"}
    )

    assert len(result.messages) == 2
    assert "Serviceとは何ですか？" in result.messages[1].content


def test_structured_prompt():
    result = structured_prompt().invoke(
        {"question": "Podとは何ですか？"}
    )

    assert "Podとは何ですか？" in result.messages[1].content


def test_valid_schema():
    result = TechnicalExplanation(
        topic="Kubernetes Service",
        summary="Podに安定した接続先を提供する仕組み",
        key_points=[
            "PodのIP変更の影響を抑える",
            "接続先Podを選択する",
            "通信を振り分ける",
        ],
    )

    assert result.topic == "Kubernetes Service"
    assert len(result.key_points) == 3


def test_missing_field():
    with pytest.raises(ValidationError):
        TechnicalExplanation(
            topic="Service",
            summary="Podへの接続先",
        )


@pytest.mark.parametrize(
    "summary",
    [
        "",
        "あ" * 101,
    ],
)
def test_invalid_summary(summary):
    with pytest.raises(ValidationError):
        TechnicalExplanation(
            topic="Service",
            summary=summary,
            key_points=["A", "B", "C"],
        )


@pytest.mark.parametrize(
    "key_points",
    [
        [],
        ["A", "B"],
        ["A", "B", "C", "D"],
    ],
)
def test_invalid_key_points(key_points):
    with pytest.raises(ValidationError):
        TechnicalExplanation(
            topic="Service",
            summary="説明",
            key_points=key_points,
        )


def test_empty_topic():
    with pytest.raises(ValidationError):
        TechnicalExplanation(
            topic="",
            summary="説明",
            key_points=["A", "B", "C"],
        )
