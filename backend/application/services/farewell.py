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
# Searched anywhere in the agent's closing sentence: a real reply opened its goodbye and
# kept going ("Goodbye, thank you for calling Hartley and Weber.").
_AGENT_FAREWELL = re.compile(
    r"\b(goodbye|bye|have a (?:great|good|nice|wonderful|lovely) (?:day|evening|weekend|afternoon)|"
    r"take care|thanks? (?:you )?for calling|auf wiederh[öo]ren|auf wiedersehen|tsch[üu]ss|"
    r"(?:einen )?sch[öo]nen tag(?: noch)?|dank für ihren anruf)\b",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"[^.!?]+[.!?]*")
_HANDOFF = re.compile(r"\b(transferring you now|ich verbinde sie jetzt)\b", re.IGNORECASE)


def call_is_over(caller_text: str, agent_text: str) -> bool:
    """The agent should hang up after this exchange: both sides said goodbye, or the agent
    handed off. Needing both farewells keeps a mid-call "thanks, bye for now I mean" from
    ending anything unless the agent also closed. Decided in code, not by the LLM: adding
    a hang-up instruction to the prompt made the 9B model re-greet in German."""
    agent = agent_text.strip()
    if _HANDOFF.search(agent):
        return True
    return bool(_CALLER_FAREWELL.search(caller_text.strip()) and _agent_is_closing(agent))


def _agent_is_closing(agent_text: str) -> bool:
    """The last sentence carries a farewell and asks nothing: "Goodbye? Before you go,
    what's your email?" keeps the line open."""
    sentences = [s.strip() for s in _SENTENCES.findall(agent_text) if s.strip()]
    if not sentences or sentences[-1].endswith("?"):
        return False
    return bool(_AGENT_FAREWELL.search(sentences[-1]))
