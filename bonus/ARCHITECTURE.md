# Architecture — AI Memory Hybrid: Vector Store + Feature Store

**Author:** Đỗ Minh Khiêm — A20-K1-2A202600463
**Lab:** Day 19 Track 2 Bonus Challenge

---

## Sơ đồ kiến trúc

```
User Input (text / query)
        │
        ▼
┌───────────────────┐
│  HybridMemoryAgent│
│                   │
│  remember(text)   │──── chunk ──► embed (bge-small-en) ──► Qdrant (in-memory)
│                   │                                         [payload: user_id, text]
│  recall(query)    │
│    │              │
│    ├─① Feast ─────────────────► get_online_features()
│    │   online store (SQLite)     user_profile_features
│    │                             query_velocity_features
│    │
│    ├─② Qdrant semantic ─────► search(embed(query), filter=user_id) → top-K
│    │
│    ├─③ BM25 keyword ────────► BM25Okapi(user_docs).get_scores(query) → top-K
│    │
│    └─④ RRF fusion ─────────► 1/(k+rank_sem) + 1/(k+rank_kw) → top-3
│                               ┌──────────────────────────────┐
│                               │ Context string assembled:    │
│                               │ - User profile (topic, speed)│
│                               │ - Recent activity            │
│                               │ - Top-3 memory snippets      │
│                               └──────────────────────────────┘
└───────────────────┘
        │
        ▼
   LLM (optional, không implement trong POC)
```

**Data flow tóm tắt:**
- `remember()` → upsert vào Qdrant (vector) ngay lập tức
- `recall()` → đọc Feast (tabular profile) + hybrid search Qdrant (semantic + BM25 + RRF) → assemble context string

---

## 3 Quyết định kiến trúc với tradeoff explicit

### 1. Chunking strategy: Per-message (không phải per-conversation hay semantic break)

**Quyết định:** Mỗi lần gọi `remember(text)` → 1 vector entry trong Qdrant. Không chunk nhỏ hơn (per-sentence) cũng không gộp lớn hơn (per-conversation).

**Tradeoff:**

| Option | Retrieval quality | Storage cost | Context window |
|---|---|---|---|
| Per-sentence | Cao (granular) | Cao (5-10× nhiều vectors) | Tốn ít token/entry |
| **Per-message** | **Tốt (balanced)** | **Thấp (1 vector/memory)** | **Vừa** |
| Per-conversation | Thấp (coarse) | Rất thấp | Tốn nhiều token/entry |

**Lý do chọn per-message:** POC này nhắm đến conversational memory — mỗi memory là 1 atomic unit có ý nghĩa (1 tài liệu đã đọc, 1 ghi chú). Per-sentence tăng recall nhưng tốn storage và gây noise khi retrieve (nhiều fragment overlap). Per-conversation làm mất precision — query "Kubernetes scaling" sẽ pull cả conversation dài chứa nhiều topic.

**Limitation:** Nếu user nhập đoạn văn dài (>512 token), embedding sẽ truncate → cần sliding window chunking cho production.

---

### 2. Feature schema: Tabular features (không phải embedding features)

**Quyết định:** User profile lưu dưới dạng tabular (topic_affinity STRING, reading_speed_wpm INT, queries_last_hour INT) trong Feast, không lưu embedding của lịch sử query.

**Tradeoff:**

| Option | Expressiveness | Latency | Interpretability | Update cost |
|---|---|---|---|---|
| **Tabular (chọn)** | **Thấp-vừa** | **< 5ms** | **Cao** | **Thấp** |
| Embedding feature | Cao (latent prefs) | 10-50ms (need ANN) | Thấp | Cao (re-embed) |

**Lý do chọn tabular:** Feast được thiết kế tối ưu cho tabular → online lookup P99 < 10ms. Embedding features yêu cầu thêm 1 ANN index riêng (Qdrant) cho feature lookup — phức tạp hoá hệ thống. Với user VN, tabular features đủ để personalise (topic_affinity = "cloud" → boost cloud docs; reading_speed = 200wpm → prefer longer summaries).

**Khi nào nên dùng embedding features:** Nếu user có multi-topic interest phức tạp không thể encode bằng 1 string (ví dụ: thích cả "AI + pháp luật" đồng thời), cần embedding để capture latent preference. Trong POC này chưa cần.

---

### 3. Freshness strategy: 3-tier theo use case

**Quyết định:** Không áp dụng 1 TTL uniform, chia 3 tier:

| Layer | Component | Freshness | Mechanism |
|---|---|---|---|
| Episodic memory | Qdrant | **Sub-second** | `remember()` upsert ngay lập tức |
| Recent activity | query_velocity_features | **1 giờ** | Feast materialize-incremental mỗi giờ, TTL=1h |
| Stable profile | user_profile_features | **30 ngày** | Feast materialize daily, TTL=30d |

**Tradeoff theo use case:**
- **Conversational assistant** (user hỏi liên tiếp): cần episodic sub-second → Qdrant direct upsert đúng
- **Fraud detection / anomaly** (hỏi "user đang làm gì bất thường?"): cần query_velocity 1h → nếu TTL=30d sẽ miss tín hiệu real-time
- **Content recommendation** (hỏi "recommend gì phù hợp?"): cần stable profile → daily refresh đủ, không cần streaming

**Lý do không dùng streaming push (sub-second) cho tất cả:** Chi phí engineering cao (Kafka/Flink), complexity tăng mạnh. 30-day TTL cho profile là đủ vì reading speed và topic affinity không thay đổi hàng giờ.

---

## Lựa chọn bị loại bỏ — Lưu episodic memory trong Feature Store

**Phương án xem xét:** Dùng Feast embedding feature view để lưu episodic memory (vector của các cuộc hội thoại) thay vì Qdrant riêng.

**Lý do loại bỏ:**
1. **Re-index cycle khác nhau hoàn toàn:** Episodic memory cần upsert sub-second (user vừa nói → nhớ ngay). Feast materialize chạy theo batch (5 phút minimum trong production). Latency gap 5 phút không chấp nhận được cho conversational memory.
2. **Query pattern khác nhau:** Feature store tối ưu cho entity lookup `(user_id → features)`. Vector store tối ưu cho ANN search `(query_vector → top-K similar)`. Dùng Feast cho ANN sẽ cần scan toàn bộ online store — O(N) thay vì O(log N).
3. **Storage model không match:** Feast online store (SQLite/Redis) lưu key-value per entity. Episodic memory cần nhiều entries per user (1 user có hàng trăm memories) — không phải 1:1 mapping.

**Kết luận:** Tách riêng là đúng. Qdrant cho unstructured episodic (nhiều entries, ANN query). Feast cho structured profile (1 entry per user, O(1) lookup).

---

## Vietnamese-context Considerations

### 1. Code-switching (vi/en mix)
User VN thường viết mixed: "tôi muốn học về machine learning và cách apply vào production". Embedding model `bge-small-en` (English-only) sẽ encode phần tiếng Anh tốt nhưng miss ngữ nghĩa tiếng Việt. **Production fix:** Dùng `bge-m3` (multilingual, 1024-dim) hoặc `multilingual-e5-large` — hỗ trợ 100+ ngôn ngữ bao gồm tiếng Việt.

### 2. Tokenization cho BM25
BM25 hiện dùng whitespace split — đủ cho tiếng Anh nhưng suboptimal cho tiếng Việt (compound words: "học sinh" ≠ "học" + "sinh"). **Production fix:** Dùng `underthesea` (word segmentation) hoặc `pyvi` trước khi tokenize. Trade-off: tăng retrieval quality +10-15% nhưng thêm dependency ~50MB và latency +2-5ms.

### 3. Phonetic typo phổ biến trong VN
User VN hay typo dấu: "Kubernetes" → "kubernetis", "cloud" → "claud". **Mitigation:** Fuzzy BM25 hoặc query expansion bằng LLM (gợi ý spelling) trước khi search. Trong POC này chưa implement.

### 4. Privacy — Nghị định 13/2023/NĐ-CP
Dữ liệu cá nhân người dùng VN (lịch sử query, reading behavior) thuộc nhóm dữ liệu cá nhân theo NĐ-13. Production cần: consent rõ ràng khi collect, right-to-delete (xóa memories theo user_id), encryption at rest. POC này không implement — cần bổ sung trước khi production.

---

## Limitations của POC này

- **Không có privacy isolation thực sự:** Filter by `user_id` trong Qdrant là payload filter, không phải access control. User A có thể query memory của user B nếu bypass filter.
- **Không có CRUD trên memories:** Không implement update/delete memory. Trong production cần cho phép user "quên" — xóa memory entry theo ID.
- **In-memory Qdrant:** Restart mất toàn bộ episodic memory. Production cần Qdrant server với persistent storage.
- **BM25 rebuild mỗi lần remember():** O(N) rebuild. Production cần incremental update hoặc ElasticSearch.
- **Không có multi-device sync:** Nếu user login từ 2 thiết bị, Feast online store sẽ đồng bộ (server-side) nhưng Qdrant in-memory sẽ không.

---

## Vibe-coding workflow note

- **Prompt hiệu quả nhất:** "Given this Qdrant upsert pattern from app/search.py, write a `remember()` method that filters by user_id payload. Schema: {id: int, vector: list[float], payload: {user_id: str, text: str}}". Output đúng ngay lần đầu vì có concrete schema.
- **Prompt fail:** "Design the best chunking strategy for Vietnamese conversational AI". Quá open-ended → AI trả generic answer không có tradeoff. Phải tự nghĩ rồi hỏi AI validate.
