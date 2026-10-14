# LangChain Minimal PoC

PythonとLangChainを使用し、Gemini APIとの連携、プロンプトテンプレート、構造化出力、Pydanticによるバリデーション、およびTool Callingを検証する最小構成のPoCです。

LLMによるツール選択とPythonによる実処理を分離し、許可された処理のみを実行する仕組みを検証しています。

## Purpose

本PoCでは、以下を検証します。

- LangChain経由でGemini APIを呼び出す
- ChatPromptTemplateによるプロンプト管理を理解する
- Pydanticを用いてLLMの出力形式を検証する
- LangChainのTool Callingを利用し、LLMによるツール選択とPythonによる処理実行を検証する
- 外部APIに依存しない単体テストを実施する
- LLMとPythonの責任範囲を分離する

## Technology

- Python 3.14
- LangChain Core
- langchain-google-genai
- Gemini API
- Pydantic
- pytest
- python-dotenv

## Project Structure

```text
langchain-minimal-poc/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── basic_chain.py
│   ├── structured_chain.py
│   └── tool_calling.py
└── tests/
    ├── __init__.py
    ├── test_chains.py
    └── test_tool_calling.py
```

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure API Key

```bash
cp .env.example .env
```

`.env`に利用可能なGoogle APIキーを設定します。

```dotenv
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.8-flash
```

`.env`はGit管理対象から除外してください。

Google APIキーを利用するため、LLM呼び出し時にはネットワーク接続が必要です。

## Run

### Basic LLM Invocation

LangChainのChatPromptTemplateを利用して、Geminiに質問を送信します。

```bash
python -m src.basic_chain
```

処理の流れ：

```text
User Question
    |
    v
ChatPromptTemplate
    |
    v
ChatGoogleGenerativeAI
    |
    v
AIMessage
    |
    v
Text Output
```

基本的なLLM呼び出しには、LangChainのRunnableインターフェースを使用しています。

```python
chain = build_prompt() | model

result = chain.invoke(
    {
        "question": "KubernetesのServiceの役割を3行で説明してください。"
    }
)

print(result.text)
```

### Structured Output

Pydanticモデルを使用して、LLMから構造化データを取得します。

```bash
python -m src.structured_chain
```

使用するスキーマ：

```python
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
```

構造化出力には以下を使用します。

```python
structured_model = model.with_structured_output(
    TechnicalExplanation,
    method="json_schema",
)

chain = build_prompt() | structured_model
```

### Example Output

2026年10月13日に実行し、以下のJSONを取得しました。

```json
{
  "topic": "Kubernetes Service",
  "summary": "動的に変化するPod群に対して単一の固定エンドポイントを提供し、トラフィックの負荷分散やサービス検出を実現するKubernetesのネットワーク抽象化リソースです。",
  "key_points": [
    "PodのIPアドレス変更を隠蔽する固定の仮想IPとDNS名の提供",
    "ラベルセレクターに基づき複数Podへのトラフィックを負荷分散",
    "ClusterIPやLoadBalancerなど用途に応じた公開種別の選択が可能"
  ]
}
```

出力内容はモデルや実行条件によって異なる場合があります。

## Tool Calling

LangChainとGeminiを使用し、LLMからPython関数を呼び出すTool Callingの最小構成を検証しました。

本PoCでは、LLMがツール名と引数を生成し、Python側が許可されたツールのみを実行します。

### Implementation

`src/tool_calling.py`では、以下の処理を実装しています。

1. `@tool`による計算関数の定義
2. `bind_tools()`によるツールの登録
3. Geminiによるツール名・引数の生成
4. Python側での許可済みツールの実行
5. `ToolMessage`による実行結果の返却
6. Geminiによる最終回答の生成

### Tool Definition

計算処理には、Pythonで実装した`calculate_total()`を使用します。

```python
@tool
def calculate_total(unit_price: int, quantity: int) -> int:
    """商品の単価と数量から合計金額を計算する。"""
    if unit_price < 0:
        raise ValueError("単価は0以上で指定してください")
    if quantity < 0:
        raise ValueError("数量は0以上で指定してください")

    return unit_price * quantity
```

単価と数量を受け取り、Python側で合計金額を計算します。

LLMはツールの呼び出しを要求しますが、実際の計算処理はPythonが担当します。

### Tool Calling Architecture

```text
User Question
    |
    v
Gemini + LangChain
    |
    v
Tool Call Request
    |
    v
Allowed Tool Validation
    |
    v
Python calculate_total()
    |
    v
ToolMessage
    |
    v
Gemini Final Answer
```

### Run

```bash
python -m src.tool_calling
```

### Execution Result

2026年10月14日の検証では、以下を確認しました。

```text
Tool Name: calculate_total

Arguments:
  unit_price: 1200
  quantity: 3

Tool Result: 3600

Final Answer:
単価1,200円の商品を3個購入した場合の合計金額は、3,600円です。
```

この結果から、以下の一連の処理が正常に動作したことを確認しました。

- Geminiが`calculate_total`を選択した
- Geminiが単価と数量を引数として生成した
- Pythonが合計金額を計算した
- `ToolMessage`で計算結果をGeminiへ返却した
- Geminiが計算結果を基に最終回答を生成した

### Validation and Safety

以下の制御を実装しています。

- 許可されたツールのみ実行する
- 1回の処理で許可するツール呼び出しは1件のみ
- 単価・数量に負数を指定した場合はエラーとする
- 必須引数が不足する場合はPydanticで検出する
- ツールの実行結果を`ToolMessage`としてLLMへ返却する

ツール実行前に、Python側でツール名を確認しています。

```python
if tool_name not in TOOLS:
    raise ValueError(f"未許可のツールです: {tool_name}")
```

また、複数のツール呼び出しが要求された場合は、実行せずエラーとしています。

```python
if len(calls) != 1:
    raise ValueError(
        f"ツール呼び出しは1件のみ許可します: {len(calls)}件"
    )
```

これにより、本PoCでは未許可ツールや複数ツールの実行を防止しています。

ただし、本番運用に必要な実行権限制御、監査ログ、リソース制限などは未実装です。

## Testing

pytestを使用し、外部APIに依存しない単体テストを実装しています。

```bash
python -m pytest -v
```

### Basic and Structured Output Test Cases

基本機能・構造化出力に関するテスト10件を実装しています。

| No. | Test | Description |
|---|---|---|
| 1 | test_basic_prompt | 基本プロンプトの生成確認 |
| 2 | test_structured_prompt | 構造化出力用プロンプトの確認 |
| 3 | test_valid_schema | 正常なスキーマの検証 |
| 4 | test_missing_field | 必須項目の欠落を検出 |
| 5 | test_invalid_summary | summaryの空文字列を検出 |
| 6 | test_invalid_summary | summaryの文字数超過を検出 |
| 7 | test_invalid_key_points | key_pointsが0件の場合 |
| 8 | test_invalid_key_points | key_pointsが2件の場合 |
| 9 | test_invalid_key_points | key_pointsが4件の場合 |
| 10 | test_empty_topic | topicの空文字列を検出 |

### Tool Calling Test Cases

Tool Calling関連のテスト8件を実装しています。

| No. | Test | Description |
|---|---|---|
| 1 | test_calculate_total | 計算結果の確認 |
| 2 | test_calculate_total_zero | 数量0の処理確認 |
| 3 | test_calculate_total_negative | 負数入力の拒否（2ケース） |
| 4 | test_calculate_total_missing_argument | 必須引数欠落の検出 |
| 5 | test_execute_tool_calls | 許可済みツールの実行 |
| 6 | test_reject_unknown_tool | 未許可ツールの拒否 |
| 7 | test_reject_multiple_tool_calls | 複数ツール呼び出しの拒否 |

※`test_calculate_total_negative`は2パターンをパラメータ化しているため、Tool Calling関連のテスト実行数は8件です。

### Test Result

2026年10月14日の実行結果：

```text
18 passed, 1 warning in 0.39s
```

内訳：

- 基本機能・構造化出力：10件成功
- Tool Calling：8件成功
- 失敗：0件
- 警告：1件

警告内容：

```text
DeprecationWarning:
'_UnionGenericAlias' is deprecated and slated
for removal in Python 3.17
```

この警告は、Google GenAI SDK内部で使用している型に関するものです。

今回のテストには影響していません。

## Architecture

以下は構造化出力の処理構成です。

Tool Callingについては「Tool Calling > Tool Calling Architecture」を参照してください。

```text
+--------------------------+
| User Question            |
+--------------------------+
             |
             v
+--------------------------+
| ChatPromptTemplate       |
+--------------------------+
             |
             v
+--------------------------+
| ChatGoogleGenerativeAI   |
| Gemini API               |
+--------------------------+
             |
             v
+--------------------------+
| Structured Output        |
| with_structured_output() |
+--------------------------+
             |
             v
+--------------------------+
| Pydantic Validation      |
+--------------------------+
             |
             v
+--------------------------+
| JSON Output              |
+--------------------------+
```

本PoCでは、LLM呼び出しと出力形式の検証を分離しています。

### Responsibilities

LangChain：

- プロンプトテンプレートの管理
- LLM呼び出し
- Runnableによる処理の連結
- 構造化出力の取得
- Tool Callingのためのツール定義・呼び出し情報の受け渡し

Pydantic：

- 必須項目の確認
- データ型の検証
- 文字数の検証
- リスト件数の検証

Python：

- 許可済みツールの判定
- 計算処理の実行
- ツール呼び出し件数の制限
- 不正な値や未許可処理の拒否

pytest：

- プロンプト生成の単体テスト
- Pydanticスキーマの正常系・異常系テスト
- 計算処理の正常系・異常系テスト
- ツール呼び出し制御のテスト

## Validation Policy

TechnicalExplanationモデルには、以下の制約を設定しています。

| Field | Validation |
|---|---|
| topic | 1文字以上 |
| summary | 1～100文字 |
| key_points | ちょうど3要素 |

これらの条件に違反するデータは、Pydanticによって`ValidationError`として検出されます。

ただし、以下は検証対象外です。

- 回答内容の事実性
- 技術的な説明の正確性
- key_pointsの各要素が空文字列でないこと
- 外部情報との整合性
- 回答根拠の妥当性

構造化出力に成功しても、回答内容が必ず正しいとは限りません。

また、Tool CallingでPythonが計算を行っても、LLMの最終回答が実行結果を正しく反映しているとは限りません。

## Comparison with Existing Gemini RAG PoC

以前のGemini RAG PoCでは、以下の構成を検証しました。

```text
User Query
    |
    v
Gemini Embedding
    |
    v
Vector Similarity (NumPy)
    |
    v
Top-K Retrieval
    |
    v
Context Construction
    |
    v
Gemini LLM
    |
    v
Answer
```

一方、本PoCの構造化出力では、以下の構成を使用しています。

```text
User Question
    |
    v
LangChain Prompt
    |
    v
Gemini LLM
    |
    v
Pydantic Structured Output
    |
    v
JSON
```

現時点では、本PoCにRAG機能は実装していません。

また、両PoCで同一条件の性能比較や実装工数の比較は実施していません。

今後、LangChainを使用する場合と使用しない場合の違いを検証します。

## Limitations

本PoCには以下の制限があります。

- RAGは未実装
- Tool Callingは1回の処理につき単一ツールの実行に限定
- 複数ツールの連続実行は未対応
- AIエージェントの自律的な実行処理は未実装
- LangGraphは未使用
- ツールの自動リトライは未実装
- ツール実行時の例外をLLMへ返す処理は未実装
- 本番利用向けの監査ログや実行権限制御は未実装
- 外部APIへのリクエスト制御は未実装
- LLM出力内容の事実性は保証しない
- 実際のLLMを使用した自動評価テストは未実装
- APIキーと利用可能なGeminiモデルが必要

本PoCは、LangChainの基本機能とTool Callingの動作を理解するための技術検証を目的としています。

## Next Steps

今後、以下を検証する予定です。

1. LangChainと既存Gemini RAG PoCの構成比較
2. LangChainによるRAGの最小実装
3. Tool Callingの複数ツール対応
4. モデル呼び出し部分のモック化
5. ツール実行時の例外処理とログ管理
6. LLM出力の品質評価
7. LangGraphを用いた状態管理とワークフロー制御

## Conclusion

本PoCでは、以下の動作を確認しました。

- LangChain経由でGemini APIを呼び出せた
- ChatPromptTemplateによるプロンプト生成に成功した
- Pydanticモデルを指定した構造化出力を取得できた
- `@tool`を用いてPython関数をツールとして定義できた
- `bind_tools()`によるツール呼び出しに成功した
- `ToolMessage`による実行結果の返却に成功した
- 許可済みツールのみ実行する制御を実装できた
- スキーマ検証・Tool Calling関連の単体テスト18件が成功した
- 外部APIに依存しない単体テストを実装できた

2026年10月14日時点のテスト結果：

```text
18 passed, 1 warning in 0.39s
```

LangChainによるLLM連携、構造化出力、Tool Callingの基本構成について実装・検証できました。

特に、LLMによるツール選択とPythonによる処理実行を分離し、許可された処理のみ実行する仕組みを確認できました。

今後はRAGやLangGraphによるワークフロー制御を追加し、LangChain導入のメリットと制約を整理していきます。
