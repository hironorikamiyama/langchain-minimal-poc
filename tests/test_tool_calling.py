
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.tool_calling import calculate_total, execute_tool_calls


def test_calculate_total():
    assert calculate_total.invoke(
        {"unit_price": 1200, "quantity": 3}
    ) == 3600


def test_calculate_total_zero():
    assert calculate_total.invoke(
        {"unit_price": 1200, "quantity": 0}
    ) == 0


@pytest.mark.parametrize(
    "unit_price,quantity",
    [
        (-1, 3),
        (1200, -1),
    ],
)
def test_calculate_total_negative(unit_price, quantity):
    with pytest.raises(ValueError):
        calculate_total.invoke(
            {"unit_price": unit_price, "quantity": quantity}
        )


def test_calculate_total_missing_argument():
    with pytest.raises(ValidationError):
        calculate_total.invoke({"unit_price": 1200})


def test_execute_tool_calls():
    message = SimpleNamespace(
        tool_calls=[
            {
                "name": "calculate_total",
                "args": {"unit_price": 1200, "quantity": 3},
                "id": "call_test_001",
            }
        ]
    )

    results = execute_tool_calls(message)

    assert len(results) == 1
    assert results[0].content == "3600"
    assert results[0].tool_call_id == "call_test_001"


def test_reject_unknown_tool():
    message = SimpleNamespace(
        tool_calls=[
            {
                "name": "delete_file",
                "args": {},
                "id": "call_test_002",
            }
        ]
    )

    with pytest.raises(ValueError, match="未許可"):
        execute_tool_calls(message)


def test_reject_multiple_tool_calls():
    message = SimpleNamespace(
        tool_calls=[
            {
                "name": "calculate_total",
                "args": {"unit_price": 1200, "quantity": 3},
                "id": "call_001",
            },
            {
                "name": "calculate_total",
                "args": {"unit_price": 100, "quantity": 2},
                "id": "call_002",
            },
        ]
    )

    with pytest.raises(ValueError, match="1件のみ"):
        execute_tool_calls(message)
