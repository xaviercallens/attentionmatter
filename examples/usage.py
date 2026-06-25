#!/usr/bin/env python3
"""
Comprehensive usage examples for attn_scorer.

Run with: python examples/usage.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from attn_scorer import Scorer, ScorerConfig, Candidate
from attn_scorer.embeddings.base import EmbeddingBackend


# --- Mock embedding for examples (no network) ---
class MockEmbed(EmbeddingBackend):
    def __init__(self):
        self._dim = 64

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(hash(text) % 2**32)
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_batch(self, texts):
        return np.array([self.embed(t) for t in texts], np.float32)

    @property
    def dimension(self):
        return self._dim


def example_basic_scoring():
    """Example 1: Basic scoring and selection."""
    print("=" * 50)
    print("Example 1: Basic Scoring")
    print("=" * 50)

    scorer = Scorer(
        ScorerConfig(decay_factor=0.95, default_token_budget=100),
        embedding=MockEmbed(),
        token_counter=lambda t: len(t.split()),
    )

    messages = [
        Candidate(text="Your booking code is XYZ789.", ctype="fact", age=50, turn=5),
        Candidate(text="How's the weather?", ctype="chit_chat", age=30, turn=25),
        Candidate(text="Nice day for a walk.", ctype="chit_chat", age=29, turn=26),
        Candidate(text="What else can I help with?", ctype="history", age=1, turn=54),
    ]
    memories = [
        Candidate(text="Booking code XYZ789, flight AB123", ctype="memory", age=0),
    ]

    result = scorer.build_context(
        query="What is my booking code?",
        messages=messages,
        memories=memories,
        token_budget=30,
    )

    print(f"Selected {len(result.selected)} items ({result.token_count} tokens)")
    print(f"Reduction: {result.reduction_vs_full:.1f}%")
    for sc in result.selected:
        print(f"  [{sc.candidate.ctype}] score={sc.score:.4f}: {sc.candidate.text[:50]}")
    print()


def example_multimodal():
    """Example 2: Multi-modal scoring."""
    print("=" * 50)
    print("Example 2: Multi-Modal Scoring")
    print("=" * 50)

    from attn_scorer.multimodal import MultiModalScorer, MultiModalCandidate, ModalityType

    scorer = MultiModalScorer(
        config=ScorerConfig(default_token_budget=200),
        embedding=MockEmbed(),
        token_counter=lambda t: len(t.split()),
    )

    scorer.add_candidate(MultiModalCandidate(
        text_repr="Python function to calculate booking total",
        raw_content="def calc_total(items):\n    return sum(i.price for i in items)",
        modality=ModalityType.CODE, language="python", age=10, turn=5,
    ))
    scorer.add_candidate(MultiModalCandidate(
        text_repr="Flight prices table with destinations and costs",
        raw_content={"columns": ["dest", "price"], "rows": [["NYC", 450], ["LAX", 380]]},
        modality=ModalityType.TABLE, age=20, turn=3,
    ))
    scorer.add_candidate(MultiModalCandidate(
        text_repr="Chatting about weather",
        raw_content="It's sunny outside today",
        modality=ModalityType.TEXT, age=5, turn=8,
    ))

    result = scorer.score_and_select("Show me the flight prices")
    print(f"Selected {len(result.selected)} items ({result.token_count} tokens)")
    for sc in result.selected:
        mod = sc.candidate.metadata.get("modality", "text")
        print(f"  [{mod}] score={sc.score:.4f}: {sc.candidate.text[:60]}")
    print()


def example_cross_agent():
    """Example 3: Cross-agent memory sharing."""
    print("=" * 50)
    print("Example 3: Cross-Agent Memory Sharing")
    print("=" * 50)

    from attn_scorer.sharing import SharedMemoryBus, AccessPolicy
    from attn_scorer.sharing.access_control import AccessLevel

    emb = MockEmbed()
    bus = SharedMemoryBus(emb)

    # Set up two agents
    booking_policy = AccessPolicy(agent_id="booking_agent")
    booking_policy.grant_access("support_agent", AccessLevel.READ, max_entries=5)

    booking_store = bus.register_agent("booking_agent", booking_policy)
    bus.register_agent("support_agent")

    # Booking agent stores memories
    booking_store.add("Customer booked flight XYZ789 to NYC", topic="booking")
    booking_store.add("Customer prefers window seats", topic="preferences")
    booking_store.add("Internal pricing model v2.3", topic="internal")

    # Support agent queries across agents
    results = bus.query_across_agents(
        requester="support_agent",
        query="What flight did the customer book?",
        top_k=3,
    )
    print(f"Support agent found {len(results)} shared memories:")
    for c in results:
        print(f"  [{c.metadata.get('source_agent')}] {c.text[:60]}")
    print()


def example_streaming():
    """Example 4: Streaming context assembly."""
    print("=" * 50)
    print("Example 4: Streaming Context")
    print("=" * 50)

    from attn_scorer.streaming import StreamingContextManager

    stream = StreamingContextManager(
        config=ScorerConfig(decay_factor=0.95, default_token_budget=50),
        embedding=MockEmbed(),
        token_counter=lambda t: len(t.split()),
    )

    stream.set_query("What is my booking code?")

    # Messages arrive over time
    messages = [
        ("Your booking code is XYZ789.", "fact", 0),
        ("How's the weather today?", "chit_chat", 1),
        ("It's sunny and warm.", "chit_chat", 2),
        ("Anything else I can help with?", "history", 3),
    ]
    for text, ctype, turn in messages:
        stream.on_message(Candidate(text=text, ctype=ctype, age=0, turn=turn))

    result = stream.get_current_context()
    print(f"Buffer: {stream.buffer_size} messages")
    print(f"Selected: {len(result.selected)} ({result.token_count} tokens)")
    for sc in result.selected:
        print(f"  score={sc.score:.4f}: {sc.candidate.text[:50]}")
    print()


def example_feedback():
    """Example 5: Feedback learning."""
    print("=" * 50)
    print("Example 5: Feedback Learning")
    print("=" * 50)

    from attn_scorer.feedback import FeedbackLearner, FeedbackStore

    config = ScorerConfig(decay_factor=0.95)
    store = FeedbackStore()  # in-memory for example
    learner = FeedbackLearner(config, store)

    # Simulate feedback signals
    for i in range(15):
        learner.record_feedback(
            query="What is my booking?",
            candidate=Candidate(text="Booking code XYZ789", ctype="memory", age=50, turn=5),
            was_needed=True,
        )
        learner.record_feedback(
            query="What is my booking?",
            candidate=Candidate(text="Nice weather today", ctype="chit_chat", age=10, turn=40),
            was_needed=False,
        )

    # Learn from feedback
    result = learner.learn(min_signals=10)
    print(f"Learning status: {result['status']}")
    if result.get("adjustments"):
        for key, val in result["adjustments"].items():
            print(f"  Adjusted {key}: {val}")
    print(f"  Config decay: {config.decay_factor}")
    print(f"  Config type_weights: { {k: f'{v:.2f}' for k,v in config.type_weights.items()} }")
    print()


if __name__ == "__main__":
    example_basic_scoring()
    example_multimodal()
    example_cross_agent()
    example_streaming()
    example_feedback()
    print("All examples completed successfully!")
