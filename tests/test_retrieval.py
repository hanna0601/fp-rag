from app.services.retrieval import detect_intent, rewrite_query


def test_detect_intent_for_greeting() -> None:
    assert detect_intent("hello") == "chitchat"


def test_detect_intent_for_identity_chitchat() -> None:
    assert detect_intent("hello, who are you") == "chitchat"


def test_detect_intent_for_knowledge_question() -> None:
    assert detect_intent("What does the report say about churn?") == "knowledge"


def test_query_rewrite_preserves_focus_terms() -> None:
    rewritten = rewrite_query("What does the annual report say about retention?")
    assert "retention" in rewritten.lower()


def test_query_rewrite_drops_obvious_filler_terms() -> None:
    rewritten = rewrite_query("What problem does the Transformer paper claim to solve?")
    assert "keywords:" in rewritten
    assert "does" not in rewritten.split("keywords:", maxsplit=1)[1]
    assert "transformer" in rewritten.lower()
