"""5-query demo — HybridMemoryAgent combining Vector Store + Feature Store."""
from agent import HybridMemoryAgent

def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def show(label: str, result: str) -> None:
    print(f"\n[Query] {label}")
    print("-" * 40)
    print(result)

agent = HybridMemoryAgent()

# ── Seed episodic memories for u_001 ──────────────────────────
section("Seeding memories for u_001...")
memories = [
    "Kubernetes là hệ thống orchestration container giúp tự động deploy, scale và quản lý ứng dụng.",
    "Horizontal Pod Autoscaler (HPA) trong Kubernetes tự động scale số Pod dựa trên CPU/memory metrics.",
    "Helm là package manager cho Kubernetes, giúp quản lý manifest phức tạp qua templates.",
    "Cloud computing cho phép tự động mở rộng hạ tầng theo lưu lượng người dùng thực tế.",
    "Bảo mật cloud cần chú ý IAM, mã hoá at-rest, network policy và audit logging.",
    "Vector database như Qdrant hỗ trợ ANN search với HNSW index, phù hợp cho semantic search.",
    "Feature store giúp tránh training-serving skew qua Point-in-Time join và TTL-based freshness.",
]
for m in memories:
    agent.remember(m, user_id="u_001")
print(f"Stored {len(memories)} memories for u_001.")

# ── 5 Demo queries ─────────────────────────────────────────────
section("Demo Query 1 — Simple vector hit")
# Hỏi đơn giản: chỉ cần vector search để tìm episodic memory đúng chủ đề
print(show("Tôi đã đọc gì về Kubernetes?",
           agent.recall("Tôi đã đọc gì về Kubernetes?", user_id="u_001")))

section("Demo Query 2 — Cần profile context")
# Cần topic_affinity từ Feast để biết user quan tâm gì
print(show("Recommend đọc gì tiếp theo?",
           agent.recall("Recommend đọc gì tiếp theo?", user_id="u_001")))

section("Demo Query 3 — Cần recent activity")
# queries_last_hour cho biết user đang active hay không
print(show("Tôi đang quan tâm gì gần đây?",
           agent.recall("Tôi đang quan tâm gì gần đây?", user_id="u_001")))

section("Demo Query 4 — Paraphrase query (vector wins)")
# Query không dùng từ exact "Kubernetes" nhưng semantic tương tự
print(show("Tài liệu về tự động mở rộng hạ tầng container?",
           agent.recall("tài liệu về tự động mở rộng hạ tầng container", user_id="u_001")))

section("Demo Query 5 — Mixed hybrid + profile")
# Vừa cần episodic (cloud security) vừa cần profile (topic_affinity, language)
print(show("Cho tôi summary cloud security",
           agent.recall("cloud security bảo mật đám mây", user_id="u_001")))

section("Done")
print("All 5 queries completed successfully.")
