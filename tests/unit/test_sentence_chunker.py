from backend.application.services.sentence_chunker import SentenceChunker


def feed(chunker: SentenceChunker, text: str, step: int = 3) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), step):
        out.extend(chunker.push(text[i : i + step]))
    return out + chunker.flush()


def test_first_sentence_is_released_as_soon_as_it_ends() -> None:
    c = SentenceChunker()
    assert c.push("Sure. ") == ["Sure."]


def test_splits_stream_into_sentences() -> None:
    text = "Thank you, Ms. Weber. I can offer Tuesday at ten. Would that suit you?"
    chunks = feed(SentenceChunker(first_chunk_min_chars=24, min_chars=10), text)
    assert " ".join(chunks) == text
    assert chunks[-1] == "Would that suit you?"


def test_waits_for_whitespace_so_decimals_are_not_split() -> None:
    c = SentenceChunker()
    assert c.push("The fee is 3.") == []
    assert c.push("5 percent of the claim value and it is due on signing. ") == [
        "The fee is 3.5 percent of the claim value and it is due on signing."
    ]


def test_first_chunk_may_break_at_a_clause_to_start_speaking_sooner() -> None:
    c = SentenceChunker(first_chunk_min_chars=20)
    out = c.push("Thank you for calling about your tenancy dispute, ")
    assert out == ["Thank you for calling about your tenancy dispute,"]
    # later chunks only break on sentence ends
    assert c.push("which sounds stressful, ") == []


def test_short_later_sentences_are_merged() -> None:
    c = SentenceChunker(first_chunk_min_chars=1, min_chars=30)
    assert c.push("Okay. ") == ["Okay."]
    assert c.push("Yes. ") == []
    assert c.push("And we can do Thursday afternoon too. ") == [
        "Yes. And we can do Thursday afternoon too."
    ]


def test_flush_returns_the_tail() -> None:
    c = SentenceChunker()
    c.push("no punctuation at all")
    assert c.flush() == ["no punctuation at all"]
    assert c.flush() == []


def test_german_sentences() -> None:
    text = "Guten Tag, Kanzlei Hartley. Worum geht es bei Ihrem Anliegen? "
    chunks = feed(SentenceChunker(), text)
    assert chunks[-1] == "Worum geht es bei Ihrem Anliegen?"
