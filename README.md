# LangChain Minimal PoC

PythonとLangChainを使い、Gemini APIへの接続、
プロンプトテンプレート、構造化出力、
Pydanticによるバリデーションを検証する最小構成のPoCです。

## Purpose

本PoCでは、以下を検証します。

- LangChain経由でGeminiを呼び出す
- ChatPromptTemplateによるプロンプト管理を理解する
- Pydanticを用いてLLMの出力形式を検証する
- 外部APIに依存しない単体テストを実施する

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
│   └── structured_chain.py
└── tests/
    ├── __init__.py
    └── test_chains.py
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

Google APIキーを利用するため、実行時にはネットワーク接続が必要です。

## Run

### Basic LLM Invocation

LangChainのChatPromptTemplateを利用し、
Geminiに質問を送信します。

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

Pydanticモデルを使用して、
LLMから構造化データを取得します。

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

## Testing

pytestを使用し、外部APIに依存しない単体テストを実装しています。

```bash
python -m pytest -v
```

### Test Cases

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

### Test Result

2026年10月13日の実行結果：

```text
10 passed, 1 warning in 0.60s
```

警告内容：

```text
DeprecationWarning:
'_UnionGenericAlias' is deprecated and slated
for removal in Python 3.17
```

この警告は、Google GenAI SDK内部で使用している型に関するものです。

今回のテストには影響していません。

## Architecture

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

LangChain：
- プロンプトテンプレートの管理
- LLM呼び出し
- Runnableによる処理の連結
- 構造化出力の取得

Pydantic：
- 必須項目の確認
- データ型の検証
- 文字数の検証
- リスト件数の検証

pytest：
- プロンプト生成の単体テスト
- Pydanticスキーマの正常系テスト
- Pydanticスキーマの異常系テスト

## Validation Policy

TechnicalExplanationモデルには、以下の制約を設定しています。

| Field | Validation |
|---|---|
| topic | 1文字以上 |
| summary | 1～100文字 |
| key_points | ちょうど3要素 |

これらの条件に違反するデータは、
PydanticによってValidationErrorとして検出されます。

ただし、以下は検証対象外です。

- 回答内容の事実性
- 技術的な説明の正確性
- key_pointsの各要素が空文字列でないこと
- 外部情報との整合性
- 回答根拠の妥当性

構造化出力に成功しても、
回答内容が必ず正しいとは限りません。

## Comparison with Existing Gemini RAG PoC

以前のGemini RAG PoCでは、
以下の構成を検証しました。

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

一方、本PoCは以下の構成です。

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

現時点では、両PoCで同一条件の性能比較や
実装工数の比較は実施していません。

今後、LangChainを使用する場合と
使用しない場合の違いを検証します。

## Limitations

本PoCには以下の制限があります。

- RAGは未実装
- AIエージェントの自律的な実行処理は未実装
- Tool Callingは未実装
- LangGraphは未使用
- 外部APIへのリクエスト制御は未実装
- LLM出力内容の事実性は保証しない
- 実際のLLMを使用した自動評価テストは未実装
- APIキーと利用可能なGeminiモデルが必要

本PoCは、LangChainの基本機能を理解するための
技術検証を目的としています。

## Next Steps

今後、以下を検証する予定です。

1. LangChainと既存Gemini RAG PoCの構成比較
2. LangChainによるRAGの最小実装
3. Tool Callingの基本動作
4. モデル呼び出し部分のモック化
5. LLM出力の品質評価
6. LangGraphを用いた状態管理とワークフロー制御

## Conclusion

本PoCでは、以下の動作を確認しました。

- LangChain経由でGemini APIを呼び出せた
- ChatPromptTemplateによるプロンプト生成に成功した
- Pydanticモデルを指定した構造化出力を取得できた
- スキーマの正常系・異常系テスト10件が成功した
- 外部APIに依存しない単体テストを実装できた

LangChainによるLLM連携の基本構成を理解し、
出力形式をPython側で検証する仕組みを確認できました。

今後は、RAGやTool Callingなどの機能を追加し、
LangChainを導入することのメリットと制約を
実装・検証を通じて整理していきます。