
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def calculate_total(unit_price: int, quantity: int) -> int:
    """商品の単価と数量から合計金額を計算する。"""
    if unit_price < 0:
        raise ValueError("単価は0以上で指定してください")
    if quantity < 0:
        raise ValueError("数量は0以上で指定してください")

    return unit_price * quantity


TOOLS = {calculate_total.name: calculate_total}


def execute_tool_calls(ai_message) -> list[ToolMessage]:
    """許可済みのツール呼び出しを1件だけ実行する。"""
    calls = ai_message.tool_calls

    if len(calls) != 1:
        raise ValueError(
            f"ツール呼び出しは1件のみ許可します: {len(calls)}件"
        )

    call = calls[0]
    tool_name = call["name"]

    if tool_name not in TOOLS:
        raise ValueError(f"未許可のツールです: {tool_name}")

    result = TOOLS[tool_name].invoke(call["args"])

    return [
        ToolMessage(
            content=str(result),
            tool_call_id=call["id"],
            name=tool_name,
        )
    ]


def main() -> None:
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEYが設定されていません")

    model = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    )

    model_with_tools = model.bind_tools(
        [calculate_total],
        tool_choice="required",
    )

    messages = [
        SystemMessage(
            content=(
                "あなたは計算アシスタントです。"
                "合計金額の計算にはcalculate_totalを使用してください。"
                "ツールが返した計算結果を変更しないでください。"
            )
        ),
        HumanMessage(
            content="単価1200円の商品を3個購入した場合の合計金額は？"
        ),
    ]

    # 1. LLMにツール呼び出しを要求させる
    ai_message = model_with_tools.invoke(messages)

    print("=== Tool Calls ===")
    print(ai_message.tool_calls)

    # 2. Python側で許可済みツールを実行
    tool_messages = execute_tool_calls(ai_message)

    print("\n=== Tool Result ===")
    for tool_message in tool_messages:
        print(tool_message.content)

    # 3. LLMに実行結果を渡して回答を生成
    # 2回目はツールを必須にせず、回答生成を依頼する
    final_messages = [
        *messages,
        ai_message,
        *tool_messages,
    ]

    final_response = model.invoke(final_messages)

    print("\n=== Final Answer ===")
    print(final_response.text)


if __name__ == "__main__":
    main()
