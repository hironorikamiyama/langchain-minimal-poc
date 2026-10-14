# LangChain Minimal PoC

PythonとLangChainを使用し、Gemini APIとの連携、プロンプトテンプレート、Pydanticによる構造化出力、Tool Calling、およびRAGの最小構成を検証するPoCです。

本PoCでは、LLMによる判断とPythonによる処理実行・検証を分離する設計を重視しています。

また、既存のGemini API直接版RAGとの比較を実施し、Embeddingのタスク種別が類似度スコアや閾値判定に与える影響を調査しました。

## Purpose

本PoCでは、以下を検証します。

- LangChain経由でGemini APIを呼び出す
- ChatPromptTemplateによるプロンプト管理を理解する
- Pydanticを用いてLLMの出力形式を検証する
- Tool CallingによるLLMのツール選択とPythonでの処理実行を検証する
- EmbeddingとNumPyを用いた最小RAGを実装する
- Gemini API直接版とLangChain版のRAGを比較する
- Embeddingのタスク種別と類似度閾値の関係を検証する
- 外部APIに依存しない単体テストを実施する
- LLMとPythonの責任範囲を明確にする

## Technology

- Python 3.14
- LangChain Core
- langchain-google-genai
- google-genai
- Gemini API
- Pydantic
- NumPy
- pytest
- python-dotenv

## Project Structure

```text
langchain-minimal-poc/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   └── rag-comparison.md
├── src/
│   ├── __init__.py
│   ├── basic_chain.py
│   ├── structured_chain.py
│   ├── tool_calling.py
│   └── rag_comparison/
│       ├── __init__.py
│       ├── retrieval.py
│       ├── langchain_rag.py
│       └── compare_embeddings.py
└── tests/
    ├── __init__.py
    ├── test_chains.py
    ├── test_tool_calling.py
    └── test_rag_comparison.py
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

`requirements.txt`には、NumPyを含む本PoCに必要な依存ライブラリを記載します。

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

本PoCでは、`GOOGLE_API_KEY`を使用します。

`GOOGLE_API_KEY`と`GEMINI_API_KEY`が同時に設定されている場合、Google GenAI SDKから使用キーに関するメッセージが表示されることがあります。

APIを使用する機能の実行には、ネットワーク接続と利用可能なGeminiモデルが必要です。

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

#### Example Output

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

LangChainとGeminiを使用し、LLMがPython関数の呼び出しを要求するTool Callingの最小構成を検証しました。

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

## RAG Implementation

LangChainとGemini Embedding APIを利用し、ベクトルDBを使用しない最小RAGを実装しました。

本PoCでは、検索ロジックをPythonとNumPyで実装しています。

LangChainは主にEmbedding APIの呼び出し、プロンプト管理、LLM呼び出しに使用しています。

### RAG Configuration

| Item | Value |
|---|---|
| Embedding Model | gemini-embedding-001 |
| Embedding Dimensions | 768 |
| Task Type | SEMANTIC_SIMILARITY |
| Number of Documents | 4 |
| Similarity | L2-normalized dot product |
| Retrieval | Top-1 |
| Similarity Threshold | 0.65 |
| Vector Database | Not used |
| Retriever | Custom Python implementation |

### RAG Architecture

```text
User Query
    |
    v
GoogleGenerativeAIEmbeddings
    |
    v
Gemini Embedding API
    |
    v
L2 Normalization (NumPy)
    |
    v
Top-1 Similarity Search
    |
    v
Threshold Check (0.65)
    |
    +---- REJECT ----> Skip LLM
    |
    v
ChatPromptTemplate
    |
    v
ChatGoogleGenerativeAI
    |
    v
Answer
```

### Retrieval Implementation

`src/rag_comparison/retrieval.py`では、以下を実装しています。

- EmbeddingベクトルのL2正規化
- NumPyによる内積計算
- Top-1検索
- 類似度閾値によるACCEPT/REJECT判定
- ベクトル次元数不一致の検出
- ゼロベクトルの拒否
- 文書件数とベクトル件数の整合性確認

検索処理はLLM APIに依存しない関数として実装し、pytestによる単体テストが可能な構成にしています。

### Run

```bash
python -m src.rag_comparison.langchain_rag
```

### Execution Result

2026年10月14日のLangChain版RAGの実行結果：

| Query | Top-1 | Similarity | Decision |
|---|---|---:|---|
| Deployment | Doc 3 | 0.8687 | ACCEPT |
| Book Purchase | Doc 2 | 0.8610 | ACCEPT |
| Paid Leave | Doc 1 | 0.7371 | ACCEPT |

回答内容：

**Q1: 本番環境へのデプロイ**

```text
火曜日と木曜日の午前10時に実施可能です。
```

**Q2: 技術書購入**

```text
5,000円を超える場合は、事前に承認を得る必要があります。
```

**Q3: 有給休暇**

```text
参照情報からは回答できません。
```

Q3では類似度が閾値0.65を超えたため、検索層ではACCEPTと判定されました。

その後、LLMが参照情報に回答が存在しないと判断し、回答を拒否しました。

したがって、Q3は検索層による拒否ではなく、生成層による回答拒否です。

## Comparison with Existing Gemini RAG PoC

既存のGemini API直接版RAGとLangChain版RAGについて、Embeddingおよび検索結果の比較検証を実施しました。

詳細な検証結果は以下を参照してください。

[Gemini API Direct vs LangChain RAG Comparison](docs/rag-comparison.md)

### Implementations

| Item | Gemini API Direct | LangChain |
|---|---|---|
| Embedding | google-genai | langchain-google-genai |
| Embedding Model | gemini-embedding-001 | gemini-embedding-001 |
| Dimensions | 768 | 768 |
| Task Type | SEMANTIC_SIMILARITY | SEMANTIC_SIMILARITY |
| Similarity | NumPy dot product | NumPy dot product |
| Retrieval | Top-1 | Top-1 |
| Threshold | 0.65 | 0.65 |
| Vector Database | Not used | Not used |

### Execution

LangChain版RAG：

```bash
python -m src.rag_comparison.langchain_rag
```

Embedding比較：

```bash
python -m src.rag_comparison.compare_embeddings
```

### Initial Comparison

初回比較では、Embeddingのタスク種別が一致していませんでした。

| Item | Gemini API Direct | LangChain |
|---|---|---|
| Embedding Task Type | Not specified | SEMANTIC_SIMILARITY |

実測結果：

| Query | Gemini Direct | LangChain |
|---|---:|---:|
| Deployment | 0.8884 | 0.8687 |
| Book Purchase | 0.7718 | 0.8610 |
| Paid Leave | 0.5973 | 0.7371 |

この段階では、Embeddingの条件が異なるため、LangChainの有無による検索精度の差とは判断できません。

### Controlled Comparison

両実装へ`SEMANTIC_SIMILARITY`を明示指定し、条件を統一して再検証しました。

2026年10月14日の実測結果：

| Query | Gemini Direct | LangChain | Top-1 | Decision |
|---|---:|---:|---|---|
| Deployment | 0.868668 | 0.868668 | Doc 3 | ACCEPT |
| Book Purchase | 0.861001 | 0.861001 | Doc 2 | ACCEPT |
| Paid Leave | 0.737051 | 0.737051 | Doc 1 | ACCEPT |

確認結果：

- 4文書のEmbeddingベクトル最大絶対差：表示精度上0
- 3質問の類似度スコア差：表示精度上0
- Top-1検索結果：全件一致
- ACCEPT/REJECT判定：全件一致

この結果から、今回の検証範囲では、LangChainの利用そのものに起因する検索スコアの差は確認されませんでした。

一方、Embeddingのタスク種別によって検索スコアや閾値判定が変化することを確認しました。

### Findings

- Embeddingモデル名だけでなく、タスク種別と次元数も比較条件に含める必要がある
- Embedding条件が異なる場合、同一の閾値を使用しても検索判定が一致するとは限らない
- Top-1の文書IDが一致しても、類似度スコアが一致するとは限らない
- 類似度は回答可能性を直接表す指標ではない
- 検索層のREJECTと生成層の回答拒否を分けて評価する必要がある
- LangChain標準RetrieverやVector Storeの有効性は今回の比較対象外

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

### RAG Retrieval Test Cases

RAG検索処理に関するテスト6件を実装しています。

| No. | Test | Description |
|---|---|---|
| 1 | test_normalize | L2正規化の確認 |
| 2 | test_retrieve_top1_accepted | Top-1検索と採用判定 |
| 3 | test_retrieve_top1_rejected | 閾値未満の拒否判定 |
| 4 | test_reject_dimension_mismatch | ベクトル次元数不一致の検出 |
| 5 | test_reject_zero_vector | ゼロベクトルの拒否 |
| 6 | test_reject_empty_documents | 文書0件の拒否 |

### Test Result

2026年10月14日の実行結果：

```text
24 passed, 1 warning in 0.89s
```

内訳：

- 基本機能・構造化出力：10件成功
- Tool Calling：8件成功
- RAG検索処理：6件成功
- 合計：24件成功
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

なお、Embedding APIの直接版とLangChain版の比較は手動実行による検証です。

今回のpytestには実APIを呼び出す比較テストは含まれていません。

## Architecture

以下は構造化出力の処理構成です。

Tool Callingについては「Tool Calling > Tool Calling Architecture」を、RAGについては「RAG Implementation > RAG Architecture」を参照してください。

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

### Responsibilities

LangChain：

- プロンプトテンプレートの管理
- LLM呼び出し
- Runnableによる処理の連結
- 構造化出力の取得
- Tool Callingのためのツール定義・呼び出し情報の受け渡し
- Embedding APIの呼び出し

Pydantic：

- 必須項目の確認
- データ型の検証
- 文字数の検証
- リスト件数の検証

Python / NumPy：

- 許可済みツールの判定
- 計算処理の実行
- ツール呼び出し件数の制限
- EmbeddingのL2正規化
- 類似度の計算
- Top-1検索
- 閾値判定
- 不正な入力や未許可処理の拒否

pytest：

- プロンプト生成の単体テスト
- Pydanticスキーマの正常系・異常系テスト
- 計算処理の正常系・異常系テスト
- ツール呼び出し制御のテスト
- RAG検索処理の正常系・異常系テスト

## Validation Policy

### Structured Output

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

### Tool Calling

LLMが生成したツール呼び出しに対し、Python側で実行可否を判定します。

現在の制約：

- 許可済みツールのみ実行可能
- 1回につき1件のツール呼び出しのみ許可
- 計算ツールへの負数入力を拒否
- 必須引数の欠落を検出

ただし、LLMの最終回答がツールの実行結果を正しく反映しているかを自動評価する仕組みは未実装です。

### RAG Retrieval

RAG検索処理では、次の検証を行います。

- 文書とEmbeddingベクトルの件数確認
- Embeddingベクトルの次元数確認
- ゼロベクトルの拒否
- L2正規化
- 類似度の計算
- Top-1検索
- 類似度閾値によるACCEPT/REJECT判定

ただし、類似度閾値0.65は暫定値であり、検索対象の文書に回答が存在することを保証するものではありません。

閾値を超えた質問についても、参照情報に回答が存在しない場合があります。

そのため、検索層と生成層の両方で回答拒否を考慮する必要があります。

## Limitations

本PoCには以下の制限があります。

### LLM and Structured Output

- LLM出力内容の事実性は保証しない
- 構造化出力のスキーマ検証は内容の正確性を保証しない
- 実際のLLMを使用した自動評価テストは未実装
- APIキーと利用可能なGeminiモデルが必要

### Tool Calling

- Tool Callingは1回の処理につき単一ツールの実行に限定
- 複数ツールの連続実行は未対応
- ツールの自動リトライは未実装
- ツール実行時の例外をLLMへ返す処理は未実装
- 本番利用向けの監査ログや実行権限制御は未実装

### RAG

- RAGは最小構成で実装済み
- LangChain標準Retrieverは未使用
- Vector Storeは未使用
- 文書のチャンク分割は未実装
- Top-1検索のみ
- 類似度閾値0.65は暫定値
- Embeddingのキャッシュ・永続化は未実装
- 検索結果の品質評価は限定的
- 回答不能な質問に対する検索層の拒否精度は未評価
- API呼び出し回数・処理時間・料金の定量比較は未実施
- Embedding比較の自動テストは未実装

### Workflow

- AIエージェントの自律的な実行処理は未実装
- LangGraphは未使用
- 外部APIへのリクエスト制御は未実装

本PoCは、LangChainの基本機能とRAG・Tool Callingの処理構成を理解するための技術検証を目的としています。

## Next Steps

今後、以下を検証する予定です。

1. RAGの回答可能・回答不可能な質問セットの拡充
2. Embeddingタスク種別ごとの類似度閾値の再評価
3. RAG検索処理のTop-K対応
4. 検索精度の評価指標導入
5. LangChain標準Retrieverを用いたRAGの比較
6. モデル呼び出し部分のモック化
7. Embedding比較の自動テスト追加
8. Tool Callingの複数ツール対応
9. ツール実行時の例外処理とログ管理
10. LLM出力の品質評価
11. LangGraphを用いた状態管理とワークフロー制御

## Conclusion

本PoCでは、以下の動作を確認しました。

### Basic LLM and Structured Output

- LangChain経由でGemini APIを呼び出せた
- ChatPromptTemplateによるプロンプト生成に成功した
- Pydanticモデルを指定した構造化出力を取得できた
- スキーマの正常系・異常系テストに成功した

### Tool Calling

- `@tool`を用いてPython関数をツールとして定義できた
- `bind_tools()`によるツール呼び出しに成功した
- `ToolMessage`による実行結果の返却に成功した
- 許可済みツールのみ実行する制御を実装できた

### RAG

- Gemini Embedding APIをLangChain経由で利用できた
- NumPyによるL2正規化とTop-1検索を実装できた
- 類似度閾値0.65による判定処理を実装できた
- 既存Gemini API直接版RAGとの比較を実施できた
- Embeddingタスク種別による類似度スコアの変化を確認できた
- 同一Embedding条件でベクトルと検索スコアが一致することを確認できた
- 検索層と生成層の回答拒否を区別して評価できた

### Test Result

2026年10月14日時点の実行結果：

```text
24 passed, 1 warning in 0.89s
```

内訳：

- 基本機能・構造化出力：10件
- Tool Calling：8件
- RAG検索処理：6件

すべての単体テストが成功しました。

### Findings

今回の技術検証では、LangChainの基本的なLLM連携からTool Calling、RAGの最小実装までを確認しました。

特に、Gemini API直接版とLangChain版のRAG比較では、初回の類似度スコア差がEmbeddingタスク種別の違いによって説明できることを確認しました。

両実装の条件を統一した再検証では、4文書のEmbeddingベクトルと3質問の類似度スコアが表示精度上で一致しました。

一方、Embedding設定を変更した場合には、従来の類似度閾値が適切とは限らないことも明らかになりました。

本PoCでは、LLMとPythonの責任分界、比較条件の統一、実測値に基づく原因分析を重視しています。

今後は検索精度の評価、閾値の再検証、LangChain標準RetrieverやLangGraphの導入を進め、LangChainを採用する場合のメリットと制約をさらに整理していきます。