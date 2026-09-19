import pytest

from backend.application.services.farewell import call_is_over


@pytest.mark.parametrize(
    ("caller", "agent", "over"),
    [
        # Real replies from Qwen 3.5 9B to a caller's goodbye.
        ("No, that's all, thank you. Bye!", "Goodbye, Anna.", True),
        ("No, that's all, thank you. Bye!", "You're welcome, Anna. Have a great day.", True),
        (
            "Nein, das war alles, danke. Tschüss!",
            "Vielen Dank, Frau Weber. Ich wünsche Ihnen einen schönen Tag. Auf Wiederhören.",
            True,
        ),
        ("Nein danke.", "Gern geschehen. Einen schönen Tag noch!", True),
        ("Can I speak to a lawyer?", "Of course. Transferring you now.", True),
        # Not over: only one side is saying goodbye, or neither.
        ("Anna Weber, thanks.", "Thank you, Anna. Is this a tenancy matter?", False),
        (
            "No, that's all, thank you. Bye!",
            "Before you go, what's the best email for the confirmation?",
            False,
        ),
        ("Hi, I need help with my landlord.", "Goodbye.", False),
        ("Bye the way, my deadline is Friday.", "Thanks. Have a great day planned?", False),
        ("Thank you.", "You're welcome. Is there anything else?", False),
    ],
)
def test_call_is_over(caller: str, agent: str, over: bool) -> None:
    assert call_is_over(caller, agent) is over


@pytest.mark.parametrize(
    "caller",
    [
        "Bye the way, my deadline is Friday and I really need help.",
        "No, thanks for asking, it's about my landlord.",
    ],
)
def test_farewell_word_mid_sentence_is_not_a_goodbye(caller: str) -> None:
    assert not call_is_over(caller, "Goodbye.")
