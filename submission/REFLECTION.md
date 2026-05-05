# Reflection — Lab 19

**Tên:** Đỗ Minh Khiêm
**Cohort:** A20-K1-2A202600463
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên 50 golden queries: với `exact` queries, BM25 và hybrid cùng đạt 96.7% vì các từ kỹ thuật xuất hiện verbatim trong corpus — BM25 match chính xác nên không cần vector. Với `mixed` queries, hybrid thắng tuyệt đối 100% (BM25: 97%, semantic: 98.5%) vì RRF kết hợp được điểm mạnh của cả hai retriever. Với `paraphrase` queries, cả ba mode đều thấp (BM25: 33.3%, hybrid: 32%) vì embedding model `bge-small-en` được train trên tiếng Anh, không hiểu paraphrase tiếng Việt tốt — đây là điểm yếu của multilingual mismatch.

Không nên dùng hybrid khi: (1) latency budget rất chặt và corpus chỉ có exact-match queries — BM25 P99 = 5.9ms vs hybrid P99 = 31ms, nhanh hơn 5×; (2) corpus nhỏ và queries luôn dùng từ kỹ thuật verbatim — BM25 đủ dùng, không cần chi phí embedding.

---

## Điều ngạc nhiên nhất khi làm lab này

Windows tạo TCP connection mới cho mỗi `httpx.get()` khiến wall-clock latency lên tới 2500ms/request — đổi sang `httpx.Client` persistent giảm xuống còn 25ms, nhanh hơn 100×. Đây là bài học thực tế về connection reuse mà benchmark lý thuyết không thể hiện.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
