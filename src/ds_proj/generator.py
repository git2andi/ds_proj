def generate_utterance(profile, act, history, topic="the task"):
    """
    Generate a simple utterance from a dialogue act.
    history: list of dicts with at least {'speaker': ..., 'act': ...}
    """
    last_speaker = history[-1]["speaker"] if history else None

    if act == "propose":
        return f"I think we should focus on {topic} first."

    if act == "agree":
        if last_speaker:
            return f"I agree with {last_speaker}. That makes sense to me."
        return "I agree with that."

    if act == "disagree":
        if last_speaker:
            return f"I am not sure I agree with {last_speaker}. We may need another view."
        return "I do not fully agree with that."

    if act == "ask":
        if last_speaker:
            return f"{last_speaker}, could you explain your point a bit more?"
        return f"Could someone clarify what we should do about {topic}?"

    return "I have nothing to add."