"""Test scenarios for the Adaptive Attention Token Reduction PoC."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .memory import Message


@dataclass
class Scenario:
    """A single test scenario with conversation, memory seeds, query, and expected fact."""
    id: str
    description: str
    conversation: list[Message]
    seed_memories: list[tuple[str, dict]] = field(default_factory=list)
    query: str = ""
    key_fact: str = ""


# --- Helper generators for filler turns ---

CHIT_CHAT = [
    "How's the weather today?",
    "It's sunny and warm here, about 75 degrees.",
    "Nice! Do you think it will rain this weekend?",
    "The forecast shows clear skies through Saturday.",
    "I watched a great movie last night.",
    "Oh really? What was it about?",
    "It was a sci-fi thriller about time travel.",
    "That sounds interesting! I love sci-fi movies.",
    "Do you follow any sports?",
    "I keep up with basketball during the season.",
    "The playoffs have been exciting this year.",
    "Yeah, some really close games recently.",
    "Have you tried any new restaurants lately?",
    "There's a new Italian place downtown that's great.",
    "I love Italian food, especially fresh pasta.",
    "Their carbonara is fantastic.",
    "What are your plans for the weekend?",
    "Thinking about going hiking if the weather holds.",
    "There are some beautiful trails nearby.",
    "I heard the mountain trail has great views.",
    "Do you have any pets?",
    "I have a golden retriever named Max.",
    "Dogs are wonderful companions.",
    "Max loves going to the park every morning.",
    "Have you been reading anything good?",
    "I'm halfway through a mystery novel.",
    "I enjoy mysteries too, especially the suspenseful ones.",
    "This one has a great plot twist at the end.",
    "Are you into any hobbies?",
    "I've been learning to play guitar.",
    "That's great! How long have you been playing?",
    "About three months now, still a beginner.",
    "What kind of music do you like?",
    "I listen to a mix of jazz and indie rock.",
    "Jazz is perfect for relaxing in the evening.",
    "I agree, it helps me unwind after work.",
    "How's your work going?",
    "Pretty busy this week with deadlines.",
    "Hope you get some rest after that.",
    "Thanks, I'm planning a quiet weekend.",
]


def _filler_messages(start_turn: int, count: int, seed: int = 42) -> list[Message]:
    """Generate filler chit-chat messages."""
    rng = random.Random(seed)
    messages = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        text = rng.choice(CHIT_CHAT)
        messages.append(Message(text=text, role=role, turn=start_turn + i))
    return messages


# --- Scenario definitions ---


def _flight_booking_memory() -> Scenario:
    """Booking code given early, stored in LTM; user asks after many irrelevant turns."""
    conversation = [
        Message(text="I want to book a flight to New York next Monday.", role="user", turn=1),
        Message(text="Sure, I found multiple flights. Do you prefer morning or afternoon?", role="assistant", turn=2),
        Message(text="Morning, please.", role="user", turn=3),
        Message(text="Booked flight AB123 for you, leaving 8 AM. Here's your booking code: XYZ789.", role="assistant", turn=4, important=True),
        Message(text="Great, thanks!", role="user", turn=5),
    ]
    # Add 60 turns of chit-chat
    conversation.extend(_filler_messages(start_turn=6, count=60, seed=100))
    # Final query turn
    conversation.append(Message(
        text="By the way, can you remind me of my booking code for that New York flight?",
        role="user", turn=66,
    ))

    return Scenario(
        id="flight_booking_memory",
        description="Booking code XYZ789 given at turn 4; user asks after 60 irrelevant turns.",
        conversation=conversation,
        seed_memories=[
            ("Flight booking code for New York trip: XYZ789, flight AB123, departure 8 AM Monday",
             {"source_session": "booking_session", "importance": 1.0}),
        ],
        query="Can you remind me of my booking code for that New York flight?",
        key_fact="XYZ789",
    )


def _support_original_problem() -> Scenario:
    """Issue described at turn 2; user asks to recall it at turn 15."""
    conversation = [
        Message(text="Hi, I need help with my laptop.", role="user", turn=1),
        Message(text="My laptop keeps showing a blue screen error code 0x0000007E when I try to open Excel.", role="user", turn=2, important=True),
        Message(text="I'm sorry to hear that. Let me help you troubleshoot. Have you tried restarting?", role="assistant", turn=3),
        Message(text="Yes, I restarted twice.", role="user", turn=4),
        Message(text="Let's try running the system file checker. Open Command Prompt as admin.", role="assistant", turn=5),
        Message(text="Okay, I did that. It says it found some issues.", role="user", turn=6),
        Message(text="Good, let it repair those files. This might take a few minutes.", role="assistant", turn=7),
        Message(text="It's done. Should I restart again?", role="user", turn=8),
        Message(text="Yes, please restart and try opening Excel again.", role="assistant", turn=9),
        Message(text="Still getting the same error after restart.", role="user", turn=10),
        Message(text="Let's try repairing your Office installation. Go to Control Panel.", role="assistant", turn=11),
        Message(text="I'm in Control Panel, what next?", role="user", turn=12),
        Message(text="Find Microsoft Office, right-click, and select Repair.", role="assistant", turn=13),
        Message(text="The repair is running now.", role="user", turn=14),
    ]
    # Final question
    conversation.append(Message(
        text="Wait, remind me what the original problem was exactly?",
        role="user", turn=15,
    ))

    return Scenario(
        id="support_original_problem",
        description="Blue screen error 0x0000007E described at turn 2; user asks to recall it at turn 15.",
        conversation=conversation,
        seed_memories=[],
        query="Wait, remind me what the original problem was exactly?",
        key_fact="0x0000007E",
    )


def _preference_recall() -> Scenario:
    """User stated preference 50 turns ago; asks about it later."""
    conversation = [
        Message(text="Just so you know, I'm vegetarian.", role="user", turn=1, important=True),
        Message(text="Noted! I'll keep that in mind for any food-related recommendations.", role="assistant", turn=2),
    ]
    # 50 turns of filler
    conversation.extend(_filler_messages(start_turn=3, count=50, seed=200))
    # Query
    conversation.append(Message(
        text="What dietary preference do you have on file for me?",
        role="user", turn=53,
    ))

    return Scenario(
        id="preference_recall",
        description="User said 'I'm vegetarian' at turn 1; asks after 50 irrelevant turns.",
        conversation=conversation,
        seed_memories=[
            ("User dietary preference: vegetarian",
             {"source_session": "preferences", "importance": 1.0}),
        ],
        query="What dietary preference do you have on file for me?",
        key_fact="vegetarian",
    )


def _cross_session_name() -> Scenario:
    """User's name only in LTM from prior session; new session asks for it."""
    conversation = [
        Message(text="Hello! I'm back.", role="user", turn=1),
        Message(text="Welcome back! How can I help you today?", role="assistant", turn=2),
    ]
    # Some filler
    conversation.extend(_filler_messages(start_turn=3, count=10, seed=300))
    conversation.append(Message(
        text="Do you remember my name?",
        role="user", turn=13,
    ))

    return Scenario(
        id="cross_session_name",
        description="User's name 'Alexander' only in LTM; new session asks 'do you remember my name?'",
        conversation=conversation,
        seed_memories=[
            ("User's name is Alexander. Introduced themselves in session on January 15.",
             {"source_session": "onboarding_session", "importance": 1.0}),
        ],
        query="Do you remember my name?",
        key_fact="Alexander",
    )


def _irrelevant_heavy() -> Scenario:
    """100 turns of chit-chat with one important fact at turn 40; user asks at turn 101."""
    conversation = _filler_messages(start_turn=1, count=39, seed=400)
    conversation.append(Message(
        text="By the way, my account number is ACC-9182736. You might need it later.",
        role="user", turn=40, important=True,
    ))
    conversation.extend(_filler_messages(start_turn=41, count=60, seed=401))
    conversation.append(Message(
        text="What's my account number? I mentioned it earlier.",
        role="user", turn=101,
    ))

    return Scenario(
        id="irrelevant_heavy",
        description="Account number ACC-9182736 at turn 40 buried in 100 turns of chit-chat.",
        conversation=conversation,
        seed_memories=[
            ("User account number: ACC-9182736",
             {"source_session": "account_session", "importance": 1.0}),
        ],
        query="What's my account number? I mentioned it earlier.",
        key_fact="ACC-9182736",
    )


def _multi_fact() -> Scenario:
    """Two distinct facts at turns 5 and 30; final query needs both."""
    conversation = [
        Message(text="I'd like to set up my profile.", role="user", turn=1),
        Message(text="Sure! What's your email?", role="assistant", turn=2),
        Message(text="It's john.doe@example.com", role="user", turn=3, important=True),
        Message(text="Got it. And your phone number?", role="assistant", turn=4),
        Message(text="My phone is 555-0142.", role="user", turn=5, important=True),
    ]
    # Filler between the two facts
    conversation.extend(_filler_messages(start_turn=6, count=24, seed=500))
    conversation.append(Message(
        text="Actually, I want to update my address to 742 Evergreen Terrace.",
        role="user", turn=30, important=True,
    ))
    conversation.extend(_filler_messages(start_turn=31, count=10, seed=501))
    conversation.append(Message(
        text="Can you confirm my phone number and address on file?",
        role="user", turn=41,
    ))

    return Scenario(
        id="multi_fact",
        description="Phone 555-0142 at turn 5, address at turn 30; query asks for both.",
        conversation=conversation,
        seed_memories=[],
        query="Can you confirm my phone number and address on file?",
        key_fact="555-0142",  # primary check; address is secondary
    )


def _no_memory_needed() -> Scenario:
    """Answer is in the last 3 messages; all strategies should pass (sanity check)."""
    conversation = [
        Message(text="What time does the store close?", role="user", turn=1),
        Message(text="The store closes at 9 PM today.", role="assistant", turn=2),
        Message(text="And what about tomorrow?", role="user", turn=3),
        Message(text="Tomorrow it closes at 6 PM since it's Sunday.", role="assistant", turn=4),
        Message(text="What time did you say it closes tomorrow?", role="user", turn=5),
    ]

    return Scenario(
        id="no_memory_needed",
        description="Answer in last few messages; all strategies should pass.",
        conversation=conversation,
        seed_memories=[],
        query="What time did you say it closes tomorrow?",
        key_fact="6 PM",
    )


def load_scenarios() -> list[Scenario]:
    """Load all defined test scenarios."""
    return [
        _flight_booking_memory(),
        _support_original_problem(),
        _preference_recall(),
        _cross_session_name(),
        _irrelevant_heavy(),
        _multi_fact(),
        _no_memory_needed(),
    ]
