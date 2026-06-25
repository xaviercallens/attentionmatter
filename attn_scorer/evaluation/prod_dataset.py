"""Production-scale dataset generator with realistic message lengths."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..models import Candidate

# Realistic paragraph templates (50-200 words each)
DOMAIN_TEMPLATES = [
    "I've been looking into the issue you reported about {topic}. After "
    "reviewing the logs from the past 24 hours, it appears that the {component} "
    "service experienced intermittent failures between 2 AM and 4 AM UTC. The "
    "root cause seems to be related to connection pool exhaustion when traffic "
    "spikes occur. I've already applied a temporary fix by increasing the pool "
    "size from 50 to 200 connections. We should monitor this over the next few "
    "days to see if the issue recurs. If it does, we may need to implement "
    "circuit breaker patterns.",

    "Based on our conversation earlier, I want to confirm the details of your "
    "reservation. You have booking reference {code} for a {service} on {date}. "
    "The total cost is {amount} which includes all taxes and fees. Your check-in "
    "time is scheduled for 3 PM and checkout is at 11 AM. Please note that "
    "cancellation within 24 hours of arrival will incur a charge equal to one "
    "night's rate. Would you like me to add any special requests to this booking?",

    "Thank you for providing your medical history. I've noted that you have a "
    "documented allergy to {allergen} which was first identified in {year}. "
    "This is classified as a {severity} reaction. I've flagged this in your "
    "patient file and it will appear as an alert whenever any medication is "
    "prescribed. Please ensure you inform any new healthcare provider about this "
    "allergy. Is there anything else about your medical history I should record?",

    "I've processed your account update request. Your new details are as follows: "
    "account number {account_num}, primary contact email updated to {email}, and "
    "mailing address changed to {address}. These changes will take effect within "
    "24 business hours. You'll receive a confirmation email at both your old and "
    "new email addresses. If you did not authorize these changes, please contact "
    "our security team immediately at the number on the back of your card.",

    "The project timeline has been revised based on yesterday's stakeholder "
    "meeting. The new deadline for {project} is {deadline}. This represents a "
    "two-week extension from the original date. The key milestones are: design "
    "review by {milestone1}, development complete by {milestone2}, and QA sign-off "
    "by {milestone3}. All team members should update their sprint planning "
    "accordingly. Let me know if you need the resource allocation adjusted.",
]

FILLER_TEMPLATES = [
    "I understand your concern about that. Let me explain how the system works "
    "in more detail. When you submit a request, it goes through several stages "
    "of processing. First, it's validated against our business rules engine. "
    "Then it enters a queue where it's prioritized based on urgency and customer "
    "tier. Finally, it's assigned to the appropriate team for resolution. The "
    "entire process typically takes between 2 and 48 hours depending on complexity.",

    "That's a great question. Many of our customers ask about this particular "
    "aspect of the service. The short answer is that it depends on your specific "
    "use case and configuration. However, I can walk you through the most common "
    "scenarios and help you determine which option would work best for your "
    "situation. Would you like me to explain the differences between the standard "
    "and premium tiers?",

    "I appreciate your patience while we work through this. I know it can be "
    "frustrating when things don't work as expected. Let me check a few things "
    "on our end to see if there's a system issue or if it's something specific "
    "to your account configuration. In the meantime, could you tell me when you "
    "first noticed this problem and whether it happens consistently or only "
    "intermittently?",

    "Just to give you some context about why this process exists. We implemented "
    "this workflow last quarter after receiving feedback from several enterprise "
    "customers who wanted more control over their approval chains. The system "
    "now supports up to five levels of approval hierarchy, custom routing rules, "
    "and automated escalation paths. Most customers find that it reduces their "
    "mean time to resolution by about 30 percent.",

    "I've looked at the analytics for your account over the past month and here "
    "are some observations. Your usage has increased by approximately 15 percent "
    "compared to the previous period. The peak usage times are consistently "
    "between 9 AM and 11 AM on weekdays. There were three instances of rate "
    "limiting triggered during high-traffic events. I'd recommend reviewing your "
    "capacity planning to ensure you have sufficient headroom going forward.",

    "Let me share some best practices that other organizations in your industry "
    "have found helpful. First, establishing clear naming conventions early on "
    "prevents confusion as your team scales. Second, regular audits of access "
    "permissions help maintain security compliance. Third, setting up automated "
    "alerts for unusual activity patterns can catch issues before they impact "
    "your end users. Would any of these be relevant to your current setup?",
]

HARD_NEGATIVE_TEMPLATES = [
    "We also have another booking reference in our system — {fake_code} — which "
    "appears to be from a previous trip. This was a {fake_service} that was "
    "completed last month. I mention this in case there's any confusion between "
    "the two reservations.",

    "Regarding allergies in general, the most common ones we see are reactions to "
    "{fake_allergen}, followed by {fake_allergen2}. These typically present as "
    "mild symptoms. I want to make sure we're distinguishing between your specific "
    "documented allergy and general sensitivities.",

    "For reference, there was a system update on {fake_date} that affected some "
    "account numbers. If your account number has changed format recently, that's "
    "expected behavior. The old format was purely numeric and the new format "
    "includes a prefix.",
]


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    num_scenarios: int = 12
    turns_options: list[int] = field(default_factory=lambda: [500, 1000, 2000])
    info_density: float = 0.3  # 30% domain-relevant messages
    hard_negatives_per_scenario: int = 3
    multi_session_scenarios: int = 4  # how many span multiple sessions
    seed: int = 42


@dataclass
class ProdScenario:
    """A production-representative evaluation scenario."""
    id: str
    description: str
    candidates: list[Candidate]
    query: str
    key_facts: list[str]
    hard_negative_indices: list[int] = field(default_factory=list)
    total_tokens_estimate: int = 0
    num_turns: int = 0
    num_sessions: int = 1


class ProdDatasetGenerator:
    """Generates production-representative evaluation scenarios."""

    def __init__(self, config: DatasetConfig | None = None):
        self._config = config or DatasetConfig()
        self._rng = random.Random(self._config.seed)

    def generate(self) -> list[ProdScenario]:
        """Generate full evaluation dataset."""
        scenarios = []
        scenario_defs = self._define_scenarios()

        for i, sdef in enumerate(scenario_defs):
            scenario = self._build_scenario(sdef, i)
            scenarios.append(scenario)

        return scenarios

    def _define_scenarios(self) -> list[dict]:
        """Define scenario parameters."""
        defs = []
        turns_cycle = self._config.turns_options

        templates = [
            {
                "topic": "flight booking",
                "key_fact_text": "Your confirmed booking reference is BK-7291034. "
                    "This is for flight AA2847 departing London Heathrow Terminal 5 "
                    "at 14:30 on March 22nd. The total fare was 847.50 GBP including "
                    "all taxes and a checked bag allowance of 23kg.",
                "key_facts": ["BK-7291034"],
                "query": "What is my flight booking reference number?",
                "key_turn_position": 0.05,  # early in conversation
            },
            {
                "topic": "medical allergy",
                "key_fact_text": "After reviewing your test results, I can confirm "
                    "you have a severe anaphylactic allergy to amoxicillin. This was "
                    "verified through both skin prick testing and specific IgE blood "
                    "work conducted on January 14th. You should carry an EpiPen at "
                    "all times and wear a medical alert bracelet.",
                "key_facts": ["amoxicillin"],
                "query": "What antibiotic am I allergic to?",
                "key_turn_position": 0.08,
            },
            {
                "topic": "account details",
                "key_fact_text": "I've verified your identity and can confirm your "
                    "premium account number is ACC-4418273-PRO. This account was "
                    "opened on September 3rd, 2023 and is currently in good standing "
                    "with a credit limit of 15,000 USD.",
                "key_facts": ["ACC-4418273-PRO"],
                "query": "What is my account number?",
                "key_turn_position": 0.10,
            },
            {
                "topic": "project deadline",
                "key_fact_text": "The executive steering committee has confirmed that "
                    "Project Phoenix must deliver its final release by November 30th, "
                    "2026. This is a hard deadline tied to the regulatory filing "
                    "window. No extensions will be granted. All workstreams must "
                    "complete their deliverables two weeks prior for integration "
                    "testing.",
                "key_facts": ["November 30"],
                "query": "When is the Project Phoenix deadline?",
                "key_turn_position": 0.15,
            },
            {
                "topic": "dietary requirement",
                "key_fact_text": "I've recorded your dietary requirements: you follow "
                    "a strict vegan diet and also have a tree nut allergy (specifically "
                    "cashews and pistachios). All meal recommendations and restaurant "
                    "suggestions will be filtered accordingly. I'll also ensure any "
                    "catering orders exclude these items.",
                "key_facts": ["vegan"],
                "query": "What dietary restrictions do you have on file for me?",
                "key_turn_position": 0.03,
            },
            {
                "topic": "support ticket",
                "key_fact_text": "Your support ticket TKT-88412 has been escalated to "
                    "our Level 3 engineering team. The issue — intermittent 503 errors "
                    "on the /api/v2/payments endpoint — appears to be related to a "
                    "database connection leak under high concurrency. ETA for fix is "
                    "48 hours.",
                "key_facts": ["TKT-88412"],
                "query": "What is the ticket number for my API issue?",
                "key_turn_position": 0.20,
            },
        ]

        for i in range(self._config.num_scenarios):
            tmpl = templates[i % len(templates)]
            turns = turns_cycle[i % len(turns_cycle)]
            is_multi_session = i < self._config.multi_session_scenarios

            defs.append({
                **tmpl,
                "turns": turns,
                "multi_session": is_multi_session,
                "scenario_idx": i,
            })

        return defs

    def _build_scenario(self, sdef: dict, idx: int) -> ProdScenario:
        """Build a full scenario from definition."""
        turns = sdef["turns"]
        key_turn = max(5, int(turns * sdef["key_turn_position"]))

        candidates = []
        hard_neg_indices = []

        for t in range(turns):
            role = "user" if t % 2 == 0 else "assistant"
            age = turns - 1 - t

            if t == key_turn:
                # Key fact message
                candidates.append(Candidate(
                    text=sdef["key_fact_text"],
                    ctype="fact",
                    age=age,
                    turn=t,
                ))
            elif self._rng.random() < self._config.info_density:
                # Domain-relevant filler
                tmpl = self._rng.choice(DOMAIN_TEMPLATES)
                text = self._fill_template(tmpl)
                candidates.append(Candidate(
                    text=text, ctype="history", age=age, turn=t,
                ))
            else:
                # Generic filler
                tmpl = self._rng.choice(FILLER_TEMPLATES)
                candidates.append(Candidate(
                    text=tmpl, ctype="chit_chat", age=age, turn=t,
                ))

        # Insert hard negatives
        for hn_idx in range(self._config.hard_negatives_per_scenario):
            pos = self._rng.randint(key_turn + 5, turns - 1)
            tmpl = self._rng.choice(HARD_NEGATIVE_TEMPLATES)
            hn_text = self._fill_template(tmpl)
            candidates[pos] = Candidate(
                text=hn_text, ctype="history", age=turns - 1 - pos, turn=pos,
            )
            hard_neg_indices.append(pos)

        # Add LTM memory (if multi-session)
        if sdef["multi_session"]:
            candidates.append(Candidate(
                text=f"[Prior session memory] {sdef['key_fact_text'][:100]}",
                ctype="memory", age=0, turn=-1,
            ))

        # Estimate tokens (words × 1.3)
        total_words = sum(len(c.text.split()) for c in candidates)
        token_est = int(total_words * 1.3)

        return ProdScenario(
            id=f"prod_{idx:02d}_{sdef['topic'].replace(' ', '_')}_{turns}t",
            description=f"{sdef['topic']} ({turns} turns, key at turn {key_turn})",
            candidates=candidates,
            query=sdef["query"],
            key_facts=sdef["key_facts"],
            hard_negative_indices=hard_neg_indices,
            total_tokens_estimate=token_est,
            num_turns=turns,
            num_sessions=2 if sdef["multi_session"] else 1,
        )

    def _fill_template(self, tmpl: str) -> str:
        """Fill template placeholders with random values."""
        replacements = {
            "{topic}": self._rng.choice(["database connectivity", "API latency", "auth failures"]),
            "{component}": self._rng.choice(["payment", "auth", "notification", "search"]),
            "{code}": f"BK-{self._rng.randint(1000000, 9999999)}",
            "{service}": self._rng.choice(["hotel stay", "flight", "car rental"]),
            "{date}": f"{self._rng.choice(['January', 'March', 'June', 'September'])} {self._rng.randint(1, 28)}",
            "{amount}": f"{self._rng.randint(200, 2000)}.{self._rng.randint(0, 99):02d}",
            "{allergen}": self._rng.choice(["penicillin", "sulfa", "latex"]),
            "{year}": str(self._rng.randint(2018, 2024)),
            "{severity}": self._rng.choice(["moderate", "severe", "mild"]),
            "{account_num}": f"ACC-{self._rng.randint(1000000, 9999999)}",
            "{email}": f"user{self._rng.randint(100, 999)}@example.com",
            "{address}": f"{self._rng.randint(1, 999)} {self._rng.choice(['Oak', 'Elm', 'Pine'])} Street",
            "{project}": self._rng.choice(["Atlas", "Phoenix", "Horizon"]),
            "{deadline}": f"{self._rng.choice(['November', 'December'])} {self._rng.randint(1, 28)}, 2026",
            "{milestone1}": f"Week {self._rng.randint(1, 4)}",
            "{milestone2}": f"Week {self._rng.randint(5, 8)}",
            "{milestone3}": f"Week {self._rng.randint(9, 12)}",
            "{fake_code}": f"BK-{self._rng.randint(1000000, 9999999)}",
            "{fake_service}": self._rng.choice(["hotel stay", "car rental"]),
            "{fake_allergen}": self._rng.choice(["peanuts", "shellfish"]),
            "{fake_allergen2}": self._rng.choice(["dairy", "gluten"]),
            "{fake_date}": f"{self._rng.choice(['February', 'April'])} {self._rng.randint(1, 28)}",
        }
        result = tmpl
        for key, val in replacements.items():
            result = result.replace(key, val)
        return result
