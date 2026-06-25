"""Training data generation for the relevance classifier."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

FILLER_MESSAGES = [
    "How's the weather today?",
    "It's sunny and warm here, about 75 degrees.",
    "I watched a great movie last night.",
    "That sounds interesting! I love sci-fi movies.",
    "Do you follow any sports?",
    "I keep up with basketball during the season.",
    "Have you tried any new restaurants lately?",
    "There's a new Italian place downtown that's great.",
    "What are your plans for the weekend?",
    "Thinking about going hiking if the weather holds.",
    "Do you have any pets?",
    "I have a golden retriever named Max.",
    "Have you been reading anything good?",
    "I'm halfway through a mystery novel.",
    "Are you into any hobbies?",
    "I've been learning to play guitar.",
    "How's your work going?",
    "Pretty busy this week with deadlines.",
    "I love Italian food, especially fresh pasta.",
    "Jazz is perfect for relaxing in the evening.",
    "The playoffs have been exciting this year.",
    "I heard the mountain trail has great views.",
    "Dogs are wonderful companions.",
    "I enjoy mysteries too, especially the suspenseful ones.",
    "I listen to a mix of jazz and indie rock.",
    "Hope you get some rest after that.",
    "The forecast shows clear skies through Saturday.",
    "Both are great choices with lots of resources available.",
    "Competition is really driving innovation right now.",
    "Even 5 minutes outside makes a difference.",
]


@dataclass
class RelevanceSample:
    """A single training sample for the relevance classifier."""

    query: str
    candidate_text: str
    label: int  # 1 = relevant, 0 = not relevant
    age: int = 0
    ctype: str = "history"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "candidate_text": self.candidate_text,
            "label": self.label,
            "age": self.age,
            "ctype": self.ctype,
        }


@dataclass
class RelevanceTemplate:
    """Template for generating relevance samples."""

    query: str
    positive_texts: list[str]  # relevant candidates
    topic_keywords: list[str]  # keywords that signal relevance


# --- Templates covering diverse domains ---

TEMPLATES = [
    RelevanceTemplate(
        query="What is my booking code?",
        positive_texts=[
            "Your booking code is XYZ789. Flight AB123 to New York.",
            "Booking confirmation: reference number XYZ789 for your trip.",
            "I've generated your booking code: XYZ789. Keep it safe.",
        ],
        topic_keywords=["booking", "code", "reference", "confirmation", "XYZ789"],
    ),
    RelevanceTemplate(
        query="What drug allergies do I have?",
        positive_texts=[
            "You have a severe allergy to penicillin.",
            "Medical record shows allergy: penicillin, sulfa drugs.",
            "Important: patient is allergic to penicillin-based antibiotics.",
        ],
        topic_keywords=["allergy", "penicillin", "allergic", "drug", "medication"],
    ),
    RelevanceTemplate(
        query="What is my account number?",
        positive_texts=[
            "Your account number is ACC-9182736.",
            "Account details: number ACC-9182736, opened January 2024.",
            "I have your account on file: ACC-9182736.",
        ],
        topic_keywords=["account", "number", "ACC-9182736"],
    ),
    RelevanceTemplate(
        query="When is the project deadline?",
        positive_texts=[
            "The Project Atlas deadline is December 15, 2026.",
            "All deliverables for Project Atlas must be submitted by Dec 15.",
            "Deadline reminder: Project Atlas due December 15, 2026.",
        ],
        topic_keywords=["deadline", "December", "Atlas", "deliverable", "due"],
    ),
    RelevanceTemplate(
        query="What is my dietary preference?",
        positive_texts=[
            "You mentioned you're vegetarian.",
            "Dietary preference on file: vegetarian, no meat.",
            "Noted that you follow a vegetarian diet.",
        ],
        topic_keywords=["vegetarian", "diet", "dietary", "preference", "food"],
    ),
    RelevanceTemplate(
        query="What was the original problem?",
        positive_texts=[
            "The original issue was a blue screen error code 0x0000007E.",
            "You reported that your laptop shows error 0x0000007E in Excel.",
            "The problem started with blue screen crashes, error 0x0000007E.",
        ],
        topic_keywords=["error", "blue screen", "0x0000007E", "problem", "crash"],
    ),
    RelevanceTemplate(
        query="What is my name?",
        positive_texts=[
            "Your name is Alexander.",
            "User identified as Alexander in the onboarding session.",
            "Hello Alexander! I have your name on file.",
        ],
        topic_keywords=["name", "Alexander", "called"],
    ),
    RelevanceTemplate(
        query="What is my phone number?",
        positive_texts=[
            "Your phone number is 555-0142.",
            "Contact phone on file: 555-0142.",
            "I have 555-0142 as your phone number.",
        ],
        topic_keywords=["phone", "number", "555-0142", "contact", "call"],
    ),
    RelevanceTemplate(
        query="What is my address?",
        positive_texts=[
            "Your address is 742 Evergreen Terrace.",
            "Address on file: 742 Evergreen Terrace.",
            "You mentioned updating your address to 742 Evergreen Terrace.",
        ],
        topic_keywords=["address", "742", "Evergreen", "Terrace", "location"],
    ),
    RelevanceTemplate(
        query="What hotel did I book?",
        positive_texts=[
            "Hotel reservation confirmed: confirmation HTL-2847193.",
            "Your hotel booking is HTL-2847193, check-in March 15.",
            "Booked at the downtown hotel, confirmation: HTL-2847193.",
        ],
        topic_keywords=["hotel", "reservation", "HTL-2847193", "booking", "check-in"],
    ),
    RelevanceTemplate(
        query="What time is the meeting?",
        positive_texts=[
            "The meeting is scheduled for 3 PM tomorrow.",
            "Your team meeting is at 3:00 PM in Conference Room B.",
            "Reminder: meeting tomorrow at 3 PM with the design team.",
        ],
        topic_keywords=["meeting", "3 PM", "scheduled", "tomorrow", "conference"],
    ),
    RelevanceTemplate(
        query="What was the error message?",
        positive_texts=[
            "The error message was: 'Connection refused on port 5432'.",
            "You encountered: ConnectionRefusedError on port 5432.",
            "The database returned 'Connection refused' when connecting.",
        ],
        topic_keywords=["error", "connection", "refused", "port", "5432", "database"],
    ),
]


def generate_dataset(
    num_samples: int = 5000,
    negative_ratio: float = 3.0,
    max_age: int = 500,
    seed: int = 42,
) -> list[RelevanceSample]:
    """
    Generate a synthetic relevance dataset.

    For each template, generates positive samples from the template's texts
    and negative samples from unrelated filler messages.

    Args:
        num_samples: Approximate total number of samples.
        negative_ratio: Ratio of negatives to positives.
        max_age: Maximum age value for candidates.
        seed: Random seed.

    Returns:
        List of RelevanceSample objects.
    """
    rng = random.Random(seed)
    samples: list[RelevanceSample] = []

    positives_per_template = int(num_samples / (1 + negative_ratio) / len(TEMPLATES))
    negatives_per_template = int(positives_per_template * negative_ratio)

    for template in TEMPLATES:
        # Positive samples
        for _ in range(positives_per_template):
            text = rng.choice(template.positive_texts)
            age = rng.randint(0, max_age)
            samples.append(RelevanceSample(
                query=template.query,
                candidate_text=text,
                label=1,
                age=age,
                ctype=rng.choice(["history", "memory", "fact"]),
            ))

        # Hard negatives: messages that share some topic but aren't the answer
        for _ in range(negatives_per_template // 3):
            # Use another template's positive as a hard negative
            other = rng.choice([t for t in TEMPLATES if t.query != template.query])
            text = rng.choice(other.positive_texts)
            age = rng.randint(0, max_age)
            samples.append(RelevanceSample(
                query=template.query,
                candidate_text=text,
                label=0,
                age=age,
                ctype="history",
            ))

        # Easy negatives: generic filler
        for _ in range(negatives_per_template - negatives_per_template // 3):
            text = rng.choice(FILLER_MESSAGES)
            age = rng.randint(0, max_age)
            samples.append(RelevanceSample(
                query=template.query,
                candidate_text=text,
                label=0,
                age=age,
                ctype="chit_chat",
            ))

    rng.shuffle(samples)
    return samples


def save_dataset(samples: list[RelevanceSample], path: str | Path) -> None:
    """Save dataset to JSONL format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict()) + "\n")
    print(f"Saved {len(samples)} samples to {path}")


def load_dataset(path: str | Path) -> list[RelevanceSample]:
    """Load dataset from JSONL format."""
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            samples.append(RelevanceSample(**d))
    return samples
