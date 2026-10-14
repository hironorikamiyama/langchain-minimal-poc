# Gemini API Direct vs LangChain RAG Comparison

## 1. Overview

本ドキュメントは、Gemini APIを直接使用したRAG PoCと、LangChainを使用したRAG PoCについて、実装方式および検索結果を比較した技術検証レポートです。

検証日：2026年10月14日

本検証では、当初、両実装の類似度スコアおよび回答拒否の判定に差が発生しました。

その原因を調査するため、Embeddingのタスク種別を統一して再検証しました。

結果として、同一条件で取得したEmbeddingベクトル、類似度スコア、Top-1検索結果および閾値判定が一致することを確認しました。

### Verification Objectives

- Gemini API直接版とLangChain版のRAG実装方式を比較する
- Embeddingのタスク種別が類似度スコアへ与える影響を確認する
- 同一条件におけるEmbeddingベクトルの一致を検証する
- Top-1検索結果および閾値判定を比較する
- 検索層と生成層の回答拒否を区別して評価する
- LangChainを導入する場合のメリットと制約を整理する

## 2. Comparison Targets

### 2.1 Gemini API Direct

既存リポジトリ：

`gemini-rag-poc`

対象ファイル：

`rag_poc.py`

主な使用技術：

- Python
- google-genai
- NumPy
- Gemini Embedding API
- Gemini LLM API

主な処理：

1. 参照文書をEmbeddingする
2. ユーザーの質問をEmbeddingする
3. NumPyによる内積でコサイン類似度を求める
4. Top-1の文書を選択する
5. 類似度が閾値未満の場合はLLMを呼び出さない
6. 閾値以上の場合のみ、参照文書を含めてLLMへ質問する

### 2.2 LangChain

対象リポジトリ：

`langchain-minimal-poc`

対象ファイル：

```text
src/rag_comparison/
├── __init__.py
├── retrieval.py
├── langchain_rag.py
└── compare_embeddings.py
```

主な使用技術：

- Python
- langchain-google-genai
- langchain-core
- google-genai（比較用）
- NumPy
- pytest

主な処理：

1. LangChain経由で参照文書をEmbeddingする
2. LangChain経由で質問をEmbeddingする
3. NumPyによる内積でコサイン類似度を求める
4. Top-1の文書を選択する
5. Python側で類似度の閾値判定を行う
6. 閾値以上の場合のみ、ChatPromptTemplate経由でGeminiへ質問する

本検証では、LangChain標準のVector StoreやRetrieverは使用していません。

検索ロジックをNumPyに統一し、Embedding APIの呼び出し方式による差を調査しています。

## 3. Verification Environment

| Item | Value |
|---|---|
| Verification Date | 2026-10-14 |
| OS | Linux（WSL2） |
| Python | 3.14.4（LangChain版のテスト実行環境） |
| Embedding Model | gemini-embedding-001 |
| Embedding Dimensions | 768 |
| Vector Normalization | L2 |
| Similarity Calculation | NumPy dot product |
| Retrieval | Top-1 |
| Similarity Threshold | 0.65 |
| Number of Documents | 4 |
| Number of Queries | 3 |
| Vector Database | Not used |

生成モデルは、既存PoCで以下を指定しています。

```python
LLM_MODEL = "gemini-3.6-flash"
```

今回のEmbedding条件統一テストでは、LLMによる回答生成を行っていません。

比較対象はEmbeddingベクトル、検索スコア、Top-1および閾値判定です。

## 4. Test Documents

既存PoCと同じ4件の文書を使用しました。

| Document ID | Content |
|---|---|
| Doc 1 | 開発チームのリモートワーク上限は週3日までとする。月曜と金曜は原則出社日とする。 |
| Doc 2 | 書籍購入費は月額5,000円まで事前申請なしで経費精算できる。5,000円を超える場合は事前の承認が必要。 |
| Doc 3 | 本番環境へのデプロイは火曜日と木曜日の午前10時にのみ実施可能。緊急デプロイにはテックリードの承認が必要。 |
| Doc 4 | 社内ツールのパスワードは90日ごとに更新が必要であり、16文字以上の英数記号混ざりで設定しなければならない。 |

今回のPoCでは、文書のチャンク分割やメタデータ検索は実装していません。

各文書を独立した1チャンクとして扱っています。

## 5. Test Queries

以下の3問を使用しました。

| Case | Query | Expected Behavior |
|---|---|---|
| Q1 | 本番環境へのデプロイは何曜日の何時にできますか？ | Doc 3を検索し、デプロイ日時を回答する |
| Q2 | 5000円を超える技術書を買いたい場合、先に何かする必要がありますか？ | Doc 2を検索し、事前承認が必要と回答する |
| Q3 | 有給休暇は年間何日取得できますか？ | 文書に存在しないため回答しない |

Q3は、参照文書にない情報に関する質問です。

検索層でLLM呼び出しを回避できるか、または生成層で回答を拒否できるかを確認するために使用しました。

## 6. Architecture Comparison

### 6.1 Gemini API Direct

```text
User Query
    |
    v
google-genai
    |
    v
Gemini Embedding API
    |
    v
NumPy L2 Normalization
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
Construct Prompt
    |
    v
Gemini generate_content()
    |
    v
Answer
```

### 6.2 LangChain

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
NumPy L2 Normalization
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

両実装とも、検索処理の最終判定はPython側で行います。

LLMに検索結果の採用・不採用を決定させていません。

## 7. Initial Comparison

### 7.1 Embedding Conditions

初回の比較では、以下の条件に違いがありました。

| Item | Gemini Direct | LangChain |
|---|---|---|
| Embedding Model | gemini-embedding-001 | gemini-embedding-001 |
| Dimensions | 768 | 768 |
| Task Type | Not specified | SEMANTIC_SIMILARITY |
| Normalization | L2 | L2 |
| Retrieval | Top-1 | Top-1 |
| Threshold | 0.65 | 0.65 |

既存Gemini API直接版では、`EmbedContentConfig`に`task_type`を指定していません。

一方、LangChain版では`SEMANTIC_SIMILARITY`を明示的に指定しました。

そのため、この段階のスコア比較は完全な同一条件ではありません。

### 7.2 Initial Retrieval Results

実測値：

| Query | Gemini Direct | LangChain | Difference |
|---|---:|---:|---:|
| Q1: Deployment | 0.8884 | 0.8687 | -0.0197 |
| Q2: Book Purchase | 0.7718 | 0.8610 | +0.0892 |
| Q3: Paid Leave | 0.5973 | 0.7371 | +0.1398 |

差分は「LangChain版のスコア - Gemini API直接版のスコア」で計算しています。

両実装とも、Top-1で選択された文書IDは同じでした。

- Q1：Doc 3
- Q2：Doc 2
- Q3：Doc 1

しかし、Q3では閾値判定が異なりました。

### 7.3 Difference in Rejection Behavior

| Item | Gemini Direct | LangChain |
|---|---|---|
| Q3 Top-1 | Doc 1 | Doc 1 |
| Q3 Similarity | 0.5973 | 0.7371 |
| Threshold | 0.65 | 0.65 |
| Retrieval Decision | REJECT | ACCEPT |
| LLM Invocation | Skipped | Executed |
| Final Behavior | Retrieval-layer rejection | Generation-layer rejection |

Gemini API直接版では、類似度0.5973が閾値0.65未満のため、LLMを呼び出していません。

一方、LangChain版では類似度0.7371が閾値を超えたため、LLMを呼び出しています。

LangChain版の回答は以下でした。

```text
参照情報からは回答できません。
```

したがって、LangChain版では最終的に回答拒否に成功していますが、検索層では質問を拒否できていません。

この違いは、閾値判定と生成モデルの回答拒否を区別して評価する必要があることを示しています。

## 8. API Error Investigation

LangChain版の実装時、Embeddingのタスク種別に関する400 INVALID_ARGUMENTが発生しました。

### 8.1 TASK_TYPE_UNSPECIFIED

当初、以下を指定しました。

```python
task_type="TASK_TYPE_UNSPECIFIED"
```

しかし、APIから次のエラーが返されました。

```text
400 INVALID_ARGUMENT

TASK_TYPE_UNSPECIFIED cannot be parsed
as a valid embedding task type.
```

### 8.2 DEFAULT

次に、以下を指定しました。

```python
task_type="DEFAULT"
```

こちらも、今回使用したAPIのエンドポイントでは受け付けられませんでした。

```text
400 INVALID_ARGUMENT

Invalid value at 'requests[0].task_type', "DEFAULT"
```

### 8.3 SEMANTIC_SIMILARITY

最終的に、文書と質問の両方へ以下を指定しました。

```python
task_type="SEMANTIC_SIMILARITY"
```

この設定では、4文書のEmbeddingおよび3問の検索が正常に動作しました。

なお、これらのエラーはタスク種別の指定に関するものであり、NumPyによる検索ロジックや閾値判定の不具合ではありません。

## 9. Controlled Comparison

初回比較で条件が揃っていなかったため、Gemini API直接版とLangChain版の両方で`SEMANTIC_SIMILARITY`を指定して再検証しました。

使用した比較スクリプト：

```text
src/rag_comparison/compare_embeddings.py
```

### 9.1 Unified Conditions

| Item | Gemini Direct | LangChain |
|---|---|---|
| API Key | Same | Same |
| Embedding Model | gemini-embedding-001 | gemini-embedding-001 |
| Dimensions | 768 | 768 |
| Task Type | SEMANTIC_SIMILARITY | SEMANTIC_SIMILARITY |
| Documents | Same 4 documents | Same 4 documents |
| Queries | Same 3 queries | Same 3 queries |
| Normalization | L2 | L2 |
| Similarity | NumPy dot product | NumPy dot product |
| Retrieval | Top-1 | Top-1 |
| Threshold | 0.65 | 0.65 |

生成モデルによる回答生成は、この比較では実行していません。

### 9.2 Document Embedding Comparison

実測結果：

```text
=== Embeddingベクトル差分 ===

Doc 1: 最大絶対差=0.00000000
Doc 2: 最大絶対差=0.00000000
Doc 3: 最大絶対差=0.00000000
Doc 4: 最大絶対差=0.00000000
```

4文書すべてについて、NumPyのfloat32配列で比較した最大絶対差が小数点以下8桁の表示精度で0となりました。

この結果から、今回の環境・設定では、両実装から取得した文書Embeddingベクトルが表示精度の範囲で一致していることを確認しました。

ただし、すべての環境・SDKバージョン・モデル更新後にビット単位で一致することを保証するものではありません。

### 9.3 Retrieval Score Comparison

実測結果：

| Query | Gemini Direct | LangChain | Absolute Difference |
|---|---:|---:|---:|
| Q1: Deployment | 0.868668 | 0.868668 | 0.00000000 |
| Q2: Book Purchase | 0.861001 | 0.861001 | 0.00000000 |
| Q3: Paid Leave | 0.737051 | 0.737051 | 0.00000000 |

すべての質問で、類似度スコアが表示精度の範囲で一致しました。

### 9.4 Top-1 and Decision Comparison

| Query | Top-1 Direct | Top-1 LangChain | Decision Direct | Decision LangChain |
|---|---|---|---|---|
| Q1 | Doc 3 | Doc 3 | ACCEPT | ACCEPT |
| Q2 | Doc 2 | Doc 2 | ACCEPT | ACCEPT |
| Q3 | Doc 1 | Doc 1 | ACCEPT | ACCEPT |

3問すべてで、Top-1検索結果と閾値判定が一致しました。

### 9.5 Interpretation

今回の結果では、同じEmbeddingタスク種別を指定した場合、Gemini API直接版とLangChain版の検索結果に表示精度上の差は確認されませんでした。

そのため、初回比較で発生した類似度スコアの差は、LangChainの使用そのものではなく、Embeddingタスク種別の違いに起因していたと考えられます。

ただし、これは今回の4文書・3質問・指定条件で得られた結果です。

LangChain全般の検索品質がGemini API直接版と常に等しいことを証明するものではありません。

## 10. Threshold Evaluation

本PoCの類似度閾値は0.65です。

しかし、Embeddingのタスク種別を変更したことで、Q3の判定が変化しました。

| Task Type | Q3 Similarity | Decision |
|---|---:|---|
| Not specified | 0.5973 | REJECT |
| SEMANTIC_SIMILARITY | 0.737051 | ACCEPT |

この結果から、Embedding設定を変更した際に既存の類似度閾値をそのまま流用することにはリスクがあると分かりました。

特に、参照文書に答えが存在しない質問であっても、意味的に近い文書がTop-1に選ばれ、閾値を超える場合があります。

また、類似度スコアは回答の正しさや、参照文書に答えが存在する確率そのものを表すものではありません。

### Implications

- Embeddingモデルやタスク種別を変更した場合は閾値を再評価する
- Top-1の類似度だけで回答可能性を断定しない
- 文書に存在しない質問のテストケースを増やす
- 検索層のREJECTと生成層の回答拒否を別々に記録する
- 閾値は少数のサンプルだけで最適化しない

今回の3質問だけでは、最適な類似度閾値を決定するには不十分です。

## 11. Implementation Comparison

| Item | Gemini API Direct | LangChain |
|---|---|---|
| API Integration | google-genai | langchain-google-genai |
| Embedding | Direct API | Wrapper API |
| Prompt Construction | f-string | ChatPromptTemplate |
| LLM Invocation | generate_content() | ChatGoogleGenerativeAI |
| Pipeline Composition | Python functions | Runnable / chain |
| Vector Calculation | NumPy | NumPy |
| Threshold Decision | Python | Python |
| Vector Database | Not used | Not used |
| Retriever | Custom | Custom |
| Automatic Evaluation | Not implemented | Not implemented |

### Gemini API Direct

Advantages:

- 依存する抽象化レイヤーが少ない
- APIへのリクエスト設定を直接確認しやすい
- 小規模なPoCでは処理構成が比較的単純
- SDK固有の機能へ直接アクセスできる

Considerations:

- プロンプトや処理の連結を個別に実装する必要がある
- 機能追加に伴い、処理の構成管理が複雑になる可能性がある

### LangChain

Advantages:

- ChatPromptTemplateによるプロンプト管理
- Runnableインターフェースによる処理の連結
- 構造化出力やTool Callingとの統合
- 他のLangChainコンポーネントへ拡張しやすい構成

Considerations:

- フレームワーク固有のAPIを理解する必要がある
- ラッパーの既定値やタスク種別の違いに注意が必要
- 依存ライブラリやバージョン管理の対象が増える
- エラー調査時にラッパー内部の処理確認が必要になる場合がある

上記は今回の実装から整理した設計上の特徴です。

処理速度、API料金、実装工数、保守工数について定量評価を行ったものではありません。

## 12. Unit Tests

LangChain版では、NumPy検索処理を外部APIから分離しました。

対象ファイル：

```text
src/rag_comparison/retrieval.py
tests/test_rag_comparison.py
```

追加したテスト：

| Test | Description |
|---|---|
| test_normalize | L2正規化の確認 |
| test_retrieve_top1_accepted | Top-1の採用判定 |
| test_retrieve_top1_rejected | Top-1の拒否判定 |
| test_reject_dimension_mismatch | 次元数不一致の検出 |
| test_reject_zero_vector | ゼロベクトルの拒否 |
| test_reject_empty_documents | 文書0件の拒否 |

### Test Result

2026年10月14日の実行結果：

```text
24 passed, 1 warning in 0.89s
```

内訳：

- 基本LLM・構造化出力：10件
- Tool Calling：8件
- RAG検索処理：6件
- 合計：24件

警告：

```text
DeprecationWarning:
'_UnionGenericAlias' is deprecated and slated
for removal in Python 3.17
```

Google GenAI SDK内部で使用している型に関する警告であり、今回のテストはすべて成功しています。

なお、Gemini API直接版とLangChain版のEmbedding比較は、現時点では手動実行による検証です。

この比較を自動化するpytestはまだ追加していません。

## 13. Limitations

今回の検証には以下の制限があります。

- テスト文書は4件のみ
- 検証質問は3件のみ
- Top-1検索のみ
- 類似度閾値は暫定値0.65
- LangChain標準Retrieverは未使用
- Vector Storeは未使用
- Embedding結果の永続化・キャッシュは未実装
- APIレスポンス時間は比較していない
- API料金は比較していない
- 実装工数・保守工数の定量評価は未実施
- LLMによる回答生成の同一条件比較は未実施
- Embedding比較スクリプトの自動テストは未実装
- モデルやSDKの将来の更新による挙動変化は未評価

また、今回のベクトル比較結果は、使用したモデル、API、SDK、入力文書、タスク種別などの条件に依存します。

## 14. Next Steps

今後は以下の検証を予定しています。

1. 回答可能・回答不可能な質問セットの拡充
2. Embeddingタスク種別ごとの閾値再評価
3. Top-K検索への拡張
4. 検索精度の評価指標導入
5. Gemini API直接版とLangChain版の処理時間比較
6. APIモックによる比較処理の自動テスト
7. LangChain標準RetrieverによるRAGの実装
8. LangGraphを用いた状態管理・ワークフロー制御

特に、閾値の妥当性については、回答可能な質問と回答不可能な質問を増やしたうえで評価する必要があります。

## 15. Conclusion

本検証では、Gemini API直接版とLangChain版のRAGを比較しました。

初回の比較では、Embeddingのタスク種別が異なっていたため、類似度スコアおよび回答拒否の判定に差が発生しました。

その後、両実装に`SEMANTIC_SIMILARITY`を指定し、Embeddingの条件を統一して再検証しました。

結果として、以下を確認しました。

- 4文書のEmbeddingベクトル差が表示精度上0となった
- 3質問の類似度スコア差が表示精度上0となった
- Top-1で選択された文書IDがすべて一致した
- 類似度閾値0.65によるACCEPT/REJECT判定がすべて一致した
- 初回の類似度差は、Embeddingタスク種別の違いによって説明できると分かった
- 既存の閾値を異なるEmbedding設定へそのまま流用する危険性を確認できた

今回の検証では、LangChainの利用そのものに起因する検索スコアの差は確認されませんでした。

一方、LangChainを導入する際には、ラッパーの既定値やAPIへのリクエスト条件を把握することが重要です。

また、RAGにおける回答拒否は、検索層と生成層の役割を区別して設計・評価する必要があります。

本PoCを通じて、フレームワークの採用有無だけでなく、比較条件を揃えること、判定処理をPython側で管理すること、実測値に基づいて原因を切り分けることの重要性を確認しました。