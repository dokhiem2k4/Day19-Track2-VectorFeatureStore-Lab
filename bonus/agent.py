"""HybridMemoryAgent — episodic memory (Qdrant) + stable profile (Feast)."""
from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastembed import TextEmbedding
from feast import FeatureStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

FEAST_DIR = REPO_ROOT / "app" / "feast_repo"
COLLECTION = "episodic_memory"
DIM = 384
RRF_K = 60

PROFILE_FEATURES = [
    "user_profile_features:topic_affinity",
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]


class HybridMemoryAgent:
    def __init__(self) -> None:
        self._embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        self._qdrant = QdrantClient(":memory:")
        self._qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )
        self._docs: list[dict] = []  # {id, user_id, text} for BM25
        self._id_counter = 0
        self._lock = Lock()
        self._fs = FeatureStore(repo_path=str(FEAST_DIR))

    def _embed(self, text: str) -> list[float]:
        return list(next(self._embed_model.embed([text])))

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        with self._lock:
            self._id_counter += 1
            point_id = self._id_counter
            vector = self._embed(text)
            self._qdrant.upsert(
                collection_name=COLLECTION,
                points=[PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"user_id": user_id, "text": text},
                )],
            )
            self._docs.append({"id": point_id, "user_id": user_id, "text": text})

    def _get_profile(self, user_id: str) -> dict:
        try:
            raw = self._fs.get_online_features(
                features=PROFILE_FEATURES,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {k: (v[0] if v else None) for k, v in raw.items()}
        except Exception:
            return {}

    def _hybrid_search(self, query: str, user_id: str, top_k: int) -> list[str]:
        user_docs = [d for d in self._docs if d["user_id"] == user_id]
        if not user_docs:
            return []

        depth = max(top_k * 5, 20)

        # Semantic search filtered by user_id
        sem_result = self._qdrant.query_points(
            collection_name=COLLECTION,
            query=self._embed(query),
            query_filter=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]),
            limit=depth,
        )
        sem_rank = {h.payload["text"]: i + 1 for i, h in enumerate(sem_result.points)}

        # BM25 on user-filtered docs only
        corpus = [d["text"].lower().split() for d in user_docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        sorted_docs = sorted(zip(scores, user_docs), key=lambda x: x[0], reverse=True)
        kw_rank = {d["text"]: i + 1 for i, (_, d) in enumerate(sorted_docs[:depth])}

        # RRF fusion
        all_texts = set(sem_rank) | set(kw_rank)
        rrf = {
            t: 1 / (RRF_K + sem_rank.get(t, depth + 1)) + 1 / (RRF_K + kw_rank.get(t, depth + 1))
            for t in all_texts
        }
        return sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k]

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories + user profile features → return assembled context."""
        # 1. Get user profile + recent activity from Feast online store
        profile = self._get_profile(user_id)
        topic     = profile.get("topic_affinity")
        speed     = profile.get("reading_speed_wpm")
        lang      = profile.get("preferred_language")
        q_hour    = profile.get("queries_last_hour")
        topics_24 = profile.get("distinct_topics_24h")

        # 2. Hybrid search Qdrant filtered by user_id
        top_memories = self._hybrid_search(query, user_id, top_k)

        # 3. Assemble context string
        if top_memories:
            mem_lines = "\n".join(f"  [{i+1}] {m}" for i, m in enumerate(top_memories))
        else:
            mem_lines = "  (no memories yet — call remember() first)"

        return (
            f"User profile ({user_id}):\n"
            f"  topic_affinity={topic}  reading_speed={speed}wpm  language={lang}\n"
            f"  queries_last_hour={q_hour}  distinct_topics_24h={topics_24}\n"
            f"\nTop-{top_k} memories for '{query}':\n"
            f"{mem_lines}"
        )
