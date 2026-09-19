from __future__ import annotations

import re

# A farewell must close the turn: at most three short words after it ("bye, thanks Clara").
_TAIL = r"(?:[\s,]+[\w'.]+){0,3}[\s,.!]*$"

# Matched against the end of each side's turn, lowercased. English and German.
_CALLER_FAREWELL = re.compile(
    r"\b(bye|goodbye|that'?s all|that is all|nothing else|no,? thanks?|no thank you|"
    r"tsch[üu]ss|auf wiederh[öo]ren|auf wiedersehen|das war'?s|das war alles|"
    r"nichts weiter|nein,? danke)\b" + _TAIL,
    re.IGNORECASE,
)
_AGENT_FAREWELL = re.compile(
    r"\b(goodbye|bye|have a (?:great|good|nice|wonderful|lovely) (?:day|evening|weekend)|"
    r"take care|auf wiederh[öo]ren|auf wiedersehen|tsch[üu]ss|"
    r"(?:einen )?sch[öo]nen tag(?: noch)?)\b" + _TAIL,
    re.IGNORECASE,
)
_HANDOFF = re.compile(r"\b(transferring you now|ich verbinde sie jetzt)\b", re.IGNORECASE)


def call_is_over(caller_text: str, agent_text: str) -> bool:
    """The agent should hang up after this exchange: both sides said goodbye, or the agent
    handed off. Needing both farewells keeps a mid-call "thanks, bye for now I mean" from
    ending anything unless the agent also closed. Decided in code, not by the LLM: adding
    a hang-up instruction to the prompt made the 9B model re-greet in German."""
    agent = agent_text.strip()
    if _HANDOFF.search(agent):
        return True
    return bool(_CALLER_FAREWELL.search(caller_text.strip()) and _AGENT_FAREWELL.search(agent))
