# RAGFlow 智能文档问答系统

RAGFlow 是一款基于检索增强生成（Retrieval-Augmented Generation, RAG）技术的智能文档问答系统，支持文档上传、多级分块、混合检索和智能问答。

---

## 核心技术亮点

### 1. 三级分层文档分块

不同于传统的单一粒度分块，本系统采用 **L1/L2/L3 三级分层策略**，保留文档的层级结构：

| 层级 | 粒度 | Token 数 | 用途 |
|------|------|----------|------|
| **L1** | 最大 | ~1200 | 保留完整主题单元，扩展上下文 |
| **L2** | 中等 | ~600 | 承接 L1 与 L3，补充段落级上下文 |
| **L3** | 最小 | ~300 | 向量入库的最小检索单元 |

**父子层级关系**：
```
L1 (祖父块)
  └── L2 (父块)
        └── L3 (叶子块)
```

**ID 格式示例**：
- L1: `sunzi_L1_0`
- L2: `sunzi_L1_0_L2_0`
- L3: `sunzi_L1_0_L2_0_L3_0`

**优势**：
- 检索时可回溯扩展到更大上下文
- 避免丢失重要语义连贯性
- L3 保证精确匹配，L1/L2 提供完整语义

### 2. 混合检索（Hybrid Search）

结合 **稠密向量 + 稀疏向量** 双重检索能力：

```
查询 → 稠密向量 (Dense Vector)  ─┬─→ RRF 融合 ─→ 重排序 ─→ 结果
        BM25 稀疏向量              ┘
```

- **稠密向量**：使用 Embedding 模型生成，捕捉语义相似性
- **稀疏向量（BM25）**：基于词频统计，精确关键词匹配
- **RRF (Reciprocal Rank Fusion)**：多结果融合排序，综合两种检索优势

### 3. AutoMerge 智能合并

针对 L3 检索结果碎片化问题，实现 **层级感知智能合并**：

**合并规则：**
- 同一个 L2 下有 **≥2 个 L3** → 合并成 L2
- 同一个 L1 下有 **≥2 个 L2** → 合并成 L1

```
输入：多个离散 L3 碎片
          ↓
1. 按 parent_id (L2) 分组 L3
          ↓
2. 判断是否满足合并条件
   - 同 L1 下有 ≥2 个 L2 组 → 合并所有 L2 及其 L3
   - 否则保持原样，扩展 L2/L1 上下文
          ↓
3. 查询 L2 内容（父块）
          ↓
4. 查询 L1 内容（祖父块）
          ↓
输出：合并后的完整段落 + L2/L1 扩展上下文
```

**效果**：不仅解决碎片化，还能根据层级关系智能聚合，提供更完整的语义单元。

### 4. 二次召回策略（Two-Stage Recall）

首次检索效果不佳时，自动触发 **二次召回**：

```
首次检索 → Grading 判断 → 通过? ─是→ 直接返回
                              │
                              └否→ 二次召回策略
                                     ├── step_back: 回溯问题（先抽象再精确）
                                     ├── hyde: 假设性答案（生成假答案反推检索）
                                     └── decompose: 子问题分解（化大为小）
                                   → 合并结果 → 返回
```

- **Grading Service**：LLM 判断上下文是否足够回答问题
- **Step-Back**：将问题抽象化 → 泛化检索 → 精确检索
- **HyDE (Hypothetical Document Embeddings)**：让 LLM 生成"理想答案"再检索
- **Decompose**：将复杂问题分解为多个简单子问题并行检索

### 5. 多 Provider 灵活切换

支持多种 Embedding 和 Reranker 提供者，灵活适配不同场景：

**Embedding 提供者**：

| 提供者 | 适用场景 | 配置 |
|--------|----------|------|
| `local` | 有 GPU，本地部署 | `EMBEDDING_MODEL` |
| `openai` | 使用 OpenAI API | `API_KEY`, `EMBEDDING_MODEL` |
| `cohere` | 使用 Cohere 服务 | `API_KEY`, `EMBEDDING_MODEL` |
| `zhipu` | 使用智谱 AI | `API_KEY`, `EMBEDDING_MODEL` |

**Reranker 提供者**：

| 提供者 | 适用场景 | 配置 |
|--------|----------|------|
| `local` | 本地 HuggingFace Rerank 模型 | `RERANK_MODEL_PATH` |
| `dashscope` | 阿里云 DashScope TextReRank API | `DASHSCOPE_API_KEY`, `RERANKER_MODEL` |

### 6. ReAct Agent 智能问答

基于 **Thought → Action → Observation** 循环的智能代理：

```
用户问题
   ↓
Thought: 判断是否需要检索知识库
   ↓
Action: rag_retrieve（调用 RAG 工具）
   ↓
Observation: 获取检索结果
   ↓
Response: 基于检索结果生成回答
```

**特点**：
- 自动判断是否需要检索（寒暄不需要）
- 工具注册机制，易于扩展
- 会话管理，支持多轮对话

### 7. 跨文档检索（CrossDocRAGRetrieveTool）

支持 **跨多个文档 Collection 联合检索**：

- 自动遍历所有已上传文档的 Collection
- 每个文档独立检索 + BM25 计算
- 全局 RRF 合并不同文档的结果
- 保留文档来源信息（doc_id）

**核心类**：`CrossDocRAGRetrieveTool` 继承自 `RAGRetrieveTool`，自动管理多文档检索逻辑。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         前端 (Web)                          │
│              HTML + CSS + JavaScript 轻量界面               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI 后端服务                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐  │
│  │ 上传接口 │  │ 问答接口 │  │ 文档管理  │  │  Agent     │  │
│  └──────────┘  └──────────┘  └───────────┘  └────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              TwoStageRecallService (二次召回)        │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐   │  │
│  │  │RewriteService│  │GradingService│  │HybridSearch │   │  │
│  │  │  step_back   │  │  判断是否足够  │  │Dense+Sparse│   │  │
│  │  │  hyde        │  │              │  │   RRF融合   │   │  │
│  │  │  decompose   │  │              │  │  AutoMerge │   │  │
│  │  └─────────────┘  └──────────────┘  └────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           │                                    │
           ▼                                    ▼
┌─────────────────────┐            ┌─────────────────────────┐
│      Milvus         │            │      PostgreSQL         │
│   (向量数据库)       │            │     (元数据存储)         │
│                     │            │                         │
│  - L3 块向量         │            │  - L1/L2 块内容          │
│  - 稠密向量          │            │  - 层级关系              │
│  - BM25 稀疏向量     │            │                         │
└─────────────────────┘            └─────────────────────────┘
```

---

## 完整工作流程

### 文档上传流程

```
1. 用户上传文档 (.txt/.pdf/.docx/.md)
          ↓
2. 文档加载
   - TXT: TextLoader 直接读取
   - PDF: UnstructuredPDFLoader (表格结构识别)
   - DOCX: UnstructuredWordDocumentLoader
   - MD: UnstructuredMarkdownLoader
          ↓
3. 三级分块 (L1→L2→L3)
   - L1: 1200 token, "\n\n\n" 分隔
   - L2: 600 token, "\n\n" 分隔
   - L3: 300 token, "\n" 分隔
          ↓
4. 向量生成
   - L3 生成稠密向量 (Embedding)
   - L3 计算 BM25 稀疏向量
          ↓
5. 存储
   ┌─────────────────┬─────────────────┐
   │     Milvus      │   PostgreSQL    │
   │                 │                 │
   │ L3块 + dense    │ L1块 + L2块     │
   │ L3块 + sparse   │ (内容存储)      │
   └─────────────────┴─────────────────┘
          ↓
6. 返回上传结果 (doc_id, chunk_count, ...)
```

### 智能问答流程

```
1. 用户提问
          ↓
2. ReAct Agent 解析
   - 寒暄/问候 → 直接回答
   - 知识库问题 → 调用 RAG 工具
          ↓
3. 查询向量化
   - 稠密向量: embed_dense(query)
   - 稀疏向量: compute_bm25(query)
          ↓
4. 混合检索 (Hybrid Search)
   ┌─────────────────────────────────┐
   │  AnnSearchRequest (dense)       │
   │      ↓ COSINE 相似度            │
   │  AnnSearchRequest (sparse)      │
   │      ↓ IP 内积                  │
   │  RRF 融合 (k=60)                │
   │  → 获取 Top-K L3 结果           │
   └─────────────────────────────────┘
          ↓
5. Rerank 重排序 (可选)
   - 本地模型 / DashScope / Cohere
          ↓
6. AutoMerge 合并 + 上下文扩展
   - L3 连续碎片合并
   - 扩展 L2 父块上下文
   - 扩展 L1 祖父块上下文
          ↓
7. Grading 判断
   - LLM 判断上下文是否足够
   - 足够 → 进入生成
   - 不足 → 触发二次召回
          ↓
8. 二次召回 (如需要)
   ┌──────────────────────────────────┐
   │  step_back:                     │
   │    生成抽象回溯问题 → 再检索      │
   │                                  │
   │  hyde:                          │
   │    生成假设性答案 → 再检索       │
   │                                  │
   │  decompose:                     │
   │    分解为子问题 → 并行检索       │
   └──────────────────────────────────┘
          ↓
9. 结果合并 + 去重 + 重排序
          ↓
10. LLM 生成回答
    - 组装完整上下文 (L3 + L2 + L1)
    - 提示词模板注入
    - 流式/非流式返回
          ↓
11. 返回最终答案
```

---

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 后端框架 | FastAPI + Uvicorn |
| 向量数据库 | Milvus 2.3+ |
| 关系数据库 | PostgreSQL 14+ |
| LLM 框架 | LangChain |
| Embedding | HuggingFace / OpenAI / Cohere / 智谱 AI |
| Reranker | BGE-Reranker / DashScope / Cohere |
| 文档解析 | Unstructured (PDF/DOCX/MD) |
| 中文分词 | Jieba |
| 前端 | 原生 HTML/CSS/JavaScript |

---

## 系统要求

- Python 3.10+
- Milvus 2.3+（向量数据库）
- PostgreSQL 14+（元数据存储）
- CUDA（可选，用于加速本地模型推理）

---

## 快速部署

### 1. 环境准备

```bash
git clone https://github.com/xzOOOOO/ragflow.git
cd ragflow
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# LLM 配置
API_KEY=your-api-key
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4

# Embedding 配置
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Reranker 配置
RERANK_PROVIDER=local
RERANK_MODEL_PATH=models/bge-reranker-base

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# PostgreSQL 配置
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=ragflow
PG_USER=postgres
PG_PASSWORD=postgres
```

### 3. 启动依赖服务

```bash
还没写好
```

### 4. 启动服务

```bash
# 后端 API
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 前端界面 (可选)
cd web && python -m http.server 3000
```

访问 `http://localhost:8000/docs` 查看 API 文档。

---

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/upload` | POST | 上传文档 |
| `/documents` | GET | 获取文档列表 |
| `/documents/{doc_id}` | DELETE | 删除文档 |
| `/chat` | POST | 文档问答 |

---

## 项目结构

```
ragflow/
├── backend/
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── db.py                # PostgreSQL 客户端
│   ├── embedding.py          # Embedding + BM25 服务
│   ├── rerank.py            # Reranker 服务
│   ├── rag.py               # RAG 生成服务
│   ├── llm.py               # LLM + Rewrite + Grading 服务
│   ├── milvus_client.py     # Milvus + Hybrid Search + AutoMerge
│   ├── document_manager.py  # 文档管理器
│   ├── document_process.py  # 三级分块服务
│   ├── schemas.py           # Pydantic 数据模型
│   ├── services.py          # 服务工厂
│   └── agent/
│       ├── agent.py         # ReAct Agent
│       ├── session.py       # 会话管理
│       ├── tools.py         # RAG 检索工具
│       └── toolservice.py   # 工具注册表
├── web/
│   ├── index.html           # 前端页面
│   ├── styles.css           # 样式文件
│   └── app.js               # 前端逻辑
├── .env                     # 环境变量配置
├── requirements.txt         # Python 依赖
└── README.md
```

---

## 常见问题

**Q: 支持哪些文档格式？**
A: 支持 TXT、PDF、DOCX、Markdown 四种常见格式。

**Q: 本地部署需要 GPU 吗？**
A: 使用本地 Embedding/Reranker 模型时建议配 GPU，可显著提升速度。没有 GPU 会回退到 CPU 模式。

**Q: 如何选择检索策略？**
A: `step_back`（默认）适合通用场景；`hyde` 适合需要生成假答案反推的场景；`decompose` 适合复杂多维度问题。

**Q: 为什么要三级分块？**
A: L3 保证精确检索，L2/L1 提供完整上下文。避免检索结果碎片化，让回答更连贯准确。

---

## License

MIT License
