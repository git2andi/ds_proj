from builders import _clean_reason, _option_hint
from models import OptionCard


def test_persona_reasons_are_normalized_without_word_truncation():
    reason = (
        "Managed backups reduce the risk of losing important research files "
        "during collaboration across several departments and locations."
    )

    cleaned = _clean_reason(reason)

    assert cleaned == reason
    assert len(cleaned.split()) > 11


def test_option_hints_preserve_complete_public_fact():
    option = OptionCard(
        id="A",
        name="University Cloud",
        short_name="Cloud",
        attrs={"storage": "managed"},
        upside=(
            "Managed backups reduce the risk of losing important research files "
            "during collaboration across several departments and locations"
        ),
        concern="Recurring fees continue for the full duration of the research project",
    )

    assert _option_hint(option, True) == option.upside
    assert _option_hint(option, False) == option.concern
