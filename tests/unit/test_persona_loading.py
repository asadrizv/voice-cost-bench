from pathlib import Path

import pytest

from backend.infrastructure.config.personas import MissingAiDisclosure, YamlPersonaProvider


def _write_persona(directory: Path, language: str | None, greeting: str) -> None:
    language_line = "" if language is None else f"language: {language}\n"
    (directory / "clara.yaml").write_text(
        f'id: clara\n{language_line}greeting: "{greeting}"\nsystem_prompt: "Hi."\n'
    )


@pytest.mark.parametrize(
    ("language", "greeting"),
    [
        ("en", "Good morning, Hartley and Weber, this is Clara. How can I help you today?"),
        ("en", "Hello, Clara speaking, your assistant at the firm."),
        ("en", "Hi, this is Clara. Aileen will call you back."),
        ("de", "Guten Tag, Kanzlei Hartley und Weber, Sie sprechen mit Clara."),
        ("de", "Guten Tag, hier ist Clara, Ihre Assistentin. Wie kann ich helfen?"),
        # An English phrasing is not recognised as a disclosure to a German caller.
        ("de", "Guten Tag, this is Clara, an AI assistant."),
        ("fr", "Bonjour, je suis Clara, une assistante IA."),
    ],
)
def test_persona_whose_greeting_does_not_disclose_ai_fails_naming_the_file(
    tmp_path: Path, language: str, greeting: str
) -> None:
    _write_persona(tmp_path, language, greeting)
    with pytest.raises(MissingAiDisclosure, match="clara.yaml"):
        YamlPersonaProvider(tmp_path).get("clara")


@pytest.mark.parametrize(
    ("language", "greeting"),
    [
        ("en", "Good morning, this is Clara, the firm's AI assistant. How can I help?"),
        ("en", "Hello, you're speaking with an AI receptionist."),
        ("en", "Hi, I'm Clara, an artificial intelligence answering for the firm."),
        ("de", "Guten Tag, hier ist Clara, Ihre KI-Assistentin. Wie kann ich helfen?"),
        ("de", "Guten Tag, Sie sprechen mit einer KI."),
        ("de", "Hallo, ich bin Clara, eine künstliche Intelligenz der Kanzlei."),
    ],
)
def test_persona_whose_greeting_discloses_ai_in_its_language_loads(
    tmp_path: Path, language: str, greeting: str
) -> None:
    _write_persona(tmp_path, language, greeting)
    persona = YamlPersonaProvider(tmp_path).get("clara")
    assert (persona.language, persona.greeting) == (language, greeting)


def test_persona_without_a_language_is_held_to_the_english_phrasings(tmp_path: Path) -> None:
    _write_persona(tmp_path, None, "Hello, this is Clara, an AI assistant.")
    assert YamlPersonaProvider(tmp_path).get("clara").language == "en"


SHIPPED = YamlPersonaProvider(Path(__file__).resolve().parents[2] / "config" / "personas")


def test_both_languages_ship_a_persona() -> None:
    assert SHIPPED.available() == ["law_firm", "law_firm_de"]


@pytest.mark.parametrize("persona_id", SHIPPED.available())
def test_every_shipped_persona_loads_with_an_ai_disclosure(persona_id: str) -> None:
    assert SHIPPED.get(persona_id).id == persona_id
