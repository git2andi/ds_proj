"""Labelled semantic acceptance corpus (todo_validation.md item 1).

Each fixture is one visible participant utterance plus the evidence a correct
interpretation pipeline must extract from it. The corpus is the regression
contract for the staged parsing/validation migration: later stubbed-validator
tests assert that the new evidence pipeline maps these utterances to exactly
this labelled semantics, without endpoint-specific regex growth.

The fixtures are pure data — no LLM calls, no src imports. Option ids and
participant ids refer to the standard test scenario from tests.fixtures
(make_scenario / make_state):

    A = Museum and Cafe Day ("Museum")      cost 24 euros, duration 4 hours
    B = Lake Bike Ride      ("Bike Ride")   cost 12 euros, duration 6 hours
    C = Escape Room         ("Escape Room") cost 32 euros, duration 2 hours
    shared context: only Saturday available; budget 60 euros per person
    participants: p1 Mira, p2 Jonas, p3 Lea

Design rules encoded here:
- public commitment (vote/accept) is distinct from ordinary preference;
- one utterance may carry several evidence types at once (multi-label);
- evidence binds to specific options, never globally to every mention;
- ambiguous references stay ambiguous instead of being guessed;
- grounding claims are classified, not keyword-matched.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- controlled vocabularies (shared with the future evidence contract) ---

SUPPORT_WEAK = "weak"
SUPPORT_CONDITIONAL = "conditional"
SUPPORT_FIRM = "firm"

CONCERN_ORDINARY = "ordinary"
CONCERN_HARD = "hard"  # hard rejection / veto strength

Q_DIRECT = "direct"
Q_GROUP = "group"
Q_RHETORICAL = "rhetorical"

KIND_FACTUAL = "factual"
KIND_COMPARATIVE = "comparative"
KIND_PREFERENCE = "preference"
KIND_PROPOSAL = "proposal"

ANSWER_FULL = "full"
ANSWER_PARTIAL = "partial"
ANSWER_EVASIVE = "evasive"
ANSWER_UNRELATED = "unrelated"

COMMIT_VOTE = "vote"      # explicit vote wording
COMMIT_ACCEPT = "accept"  # acceptance / can-live-with wording

CLAIM_LISTED_FACT = "listed_fact"
CLAIM_ARITHMETIC = "arithmetic"           # simple reproducible arithmetic over listed values
CLAIM_OPINION = "opinion"
CLAIM_INFERENCE = "inference"             # qualified logical inference from listed facts
CLAIM_UNCERTAINTY = "uncertainty"
CLAIM_INVENTED = "invented_detail"        # concrete detail not in the scenario
CLAIM_CROSS_OPTION = "cross_option_transfer"  # another option's value applied to this one


@dataclass(frozen=True)
class Question:
    scope: str                     # Q_DIRECT / Q_GROUP / Q_RHETORICAL
    kind: str = KIND_FACTUAL
    addressee: str | None = None   # participant id for direct questions
    options: tuple[str, ...] = ()  # option scope when visible


@dataclass(frozen=True)
class Comparison:
    options: tuple[str, ...]
    favored: str | None = None     # only when the text visibly favors one side
    dimension: str | None = None   # only when visible in the text


@dataclass(frozen=True)
class Answer:
    completeness: str              # ANSWER_FULL / PARTIAL / EVASIVE / UNRELATED
    addresses_target: bool


@dataclass(frozen=True)
class Switch:
    source: str | None             # None when the old pick is not visible in text
    target: str
    has_visible_reason: bool


@dataclass(frozen=True)
class Claim:
    span: str                      # exact substring of the utterance
    kind: str                      # CLAIM_* vocabulary
    option: str | None = None      # subject option when applicable


@dataclass(frozen=True)
class SemanticFixture:
    fixture_id: str
    category: str
    text: str
    speaker_id: str = "p1"
    # Optional immediately-preceding public turn, for answer/pronoun fixtures.
    context_speaker_id: str | None = None
    context_text: str | None = None
    # --- expected visible evidence (multi-label target semantics) ---
    support: tuple[tuple[str, str], ...] = ()   # (option_id, SUPPORT_* strength)
    concerns: tuple[tuple[str, str], ...] = ()  # (option_id, CONCERN_* severity)
    comparisons: tuple[Comparison, ...] = ()
    questions: tuple[Question, ...] = ()
    answer: Answer | None = None
    concession: bool = False                    # visibly moving off an earlier stance
    softens_toward: str | None = None
    proposes: str | None = None                 # option proposed as common ground
    commitment: tuple[str, str] | None = None   # (COMMIT_VOTE|COMMIT_ACCEPT, option_id)
    switch: Switch | None = None
    blocker_raised: str | None = None
    blocker_resolved: str | None = None
    claims: tuple[Claim, ...] = ()
    ambiguous_reference: bool = False           # a reference must stay unresolved
    context_resolved_options: tuple[str, ...] = ()  # options resolvable only via context
    notes: str = ""

    def evidence_kinds(self) -> set[str]:
        """Which evidence families this fixture expects (for coverage checks)."""
        kinds: set[str] = set()
        if self.support:
            kinds.add("support")
        if self.concerns:
            kinds.add("concern")
        if self.comparisons:
            kinds.add("comparison")
        if self.questions:
            kinds.add("question")
        if self.answer:
            kinds.add("answer")
        if self.concession:
            kinds.add("concession")
        if self.softens_toward:
            kinds.add("softening")
        if self.proposes:
            kinds.add("proposal")
        if self.commitment:
            kinds.add("commitment")
        if self.switch:
            kinds.add("switch")
        if self.blocker_raised:
            kinds.add("blocker")
        if self.blocker_resolved:
            kinds.add("blocker_resolution")
        return kinds


FIXTURES: tuple[SemanticFixture, ...] = (
    # ------------------------------------------------------------------
    # Support: direct, indirect, weak, conditional, firm
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="support_direct",
        category="support_direct",
        text="I really like the Museum — it keeps the whole day easy to plan.",
        support=(("A", SUPPORT_FIRM),),
        claims=(Claim("it keeps the whole day easy to plan", CLAIM_OPINION, "A"),),
    ),
    SemanticFixture(
        fixture_id="support_indirect",
        category="support_indirect",
        text="The Museum feels like the easiest day for everyone.",
        support=(("A", SUPPORT_WEAK),),
        claims=(Claim("feels like the easiest day for everyone", CLAIM_OPINION, "A"),),
        notes="Known current failure form from the todo: no support verb, no commitment phrase.",
    ),
    SemanticFixture(
        fixture_id="support_weak",
        category="support_weak",
        text="The Bike Ride could be nice, I suppose.",
        support=(("B", SUPPORT_WEAK),),
        claims=(Claim("could be nice", CLAIM_OPINION, "B"),),
    ),
    SemanticFixture(
        fixture_id="support_conditional",
        category="support_conditional",
        text="I can support the Escape Room as long as we book early.",
        support=(("C", SUPPORT_CONDITIONAL),),
        notes="Conditional support is not a public commitment.",
    ),
    SemanticFixture(
        fixture_id="support_firm",
        category="support_firm",
        text="The Bike Ride is clearly the best fit for the group.",
        support=(("B", SUPPORT_FIRM),),
        claims=(Claim("clearly the best fit for the group", CLAIM_OPINION, "B"),),
        notes="Firm support without commitment wording — still not a vote.",
    ),
    # ------------------------------------------------------------------
    # Concern: keyword, natural wording, hard rejection, blocker resolution
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="concern_keyword",
        category="concern_keyword",
        text="My main concern with the Escape Room is the 32 euro price.",
        concerns=(("C", CONCERN_ORDINARY),),
        claims=(Claim("the 32 euro price", CLAIM_LISTED_FACT, "C"),),
    ),
    SemanticFixture(
        fixture_id="concern_natural_hesitant",
        category="concern_natural",
        text="I'm hesitant about the Escape Room cost.",
        concerns=(("C", CONCERN_ORDINARY),),
        notes="Known current failure form from the todo: no concern keyword.",
    ),
    SemanticFixture(
        fixture_id="concern_natural_pause",
        category="concern_natural",
        text="The Museum's price gives me pause.",
        concerns=(("A", CONCERN_ORDINARY),),
        notes="Known current failure form from the todo: no concern keyword.",
    ),
    SemanticFixture(
        fixture_id="reject_hard_veto",
        category="hard_rejection",
        text="The Escape Room just doesn't work for me — count me out.",
        concerns=(("C", CONCERN_HARD),),
        blocker_raised="C",
    ),
    SemanticFixture(
        fixture_id="reject_hard_explicit",
        category="hard_rejection",
        text="I can't support the Bike Ride; six hours of riding is a hard no for me.",
        concerns=(("B", CONCERN_HARD),),
        blocker_raised="B",
        claims=(Claim("six hours of riding", CLAIM_LISTED_FACT, "B"),),
    ),
    SemanticFixture(
        fixture_id="blocker_resolution_accept",
        category="blocker_resolution",
        text="That fixes my concern; I can live with the Escape Room.",
        blocker_resolved="C",
        commitment=(COMMIT_ACCEPT, "C"),
        notes="Same-line resolution followed by acceptance must be possible (item 6).",
    ),
    # ------------------------------------------------------------------
    # Comparisons: balanced, directional, key-difference, comparative question
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="compare_balanced",
        category="compare_balanced",
        text="The Museum is calmer; the Bike Ride is cheaper.",
        comparisons=(Comparison(options=("A", "B")),),
        claims=(
            Claim("The Museum is calmer", CLAIM_OPINION, "A"),
            Claim("the Bike Ride is cheaper", CLAIM_ARITHMETIC, "B"),
        ),
        notes="Known current failure form from the todo: no comparative keyword.",
    ),
    SemanticFixture(
        fixture_id="compare_directional",
        category="compare_directional",
        text="The Bike Ride beats the Escape Room on cost.",
        comparisons=(Comparison(options=("B", "C"), favored="B", dimension="cost"),),
        claims=(Claim("beats the Escape Room on cost", CLAIM_ARITHMETIC, "B"),),
    ),
    SemanticFixture(
        fixture_id="compare_key_difference",
        category="compare_balanced",
        text="Between the Museum and the Bike Ride, flexibility is the key difference.",
        comparisons=(Comparison(options=("A", "B"), dimension="flexibility"),),
        notes="Known current failure form from the todo.",
    ),
    SemanticFixture(
        fixture_id="compare_question",
        category="comparative_question",
        text="Which is easier to plan, the Museum or the Escape Room?",
        comparisons=(Comparison(options=("A", "C"), dimension="planning"),),
        questions=(Question(scope=Q_GROUP, kind=KIND_COMPARATIVE, options=("A", "C")),),
        notes="Realizes both ASK and COMPARE; primary label ASK must not erase the comparison.",
    ),
    # ------------------------------------------------------------------
    # Questions: direct, group, rhetorical
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="ask_direct",
        category="question_direct",
        text="Jonas, would the six-hour ride be too much for you?",
        questions=(Question(scope=Q_DIRECT, kind=KIND_PREFERENCE, addressee="p2", options=("B",)),),
        claims=(Claim("the six-hour ride", CLAIM_LISTED_FACT, "B"),),
    ),
    SemanticFixture(
        fixture_id="ask_group",
        category="question_group",
        text="Could we all manage the 32 euros for the Escape Room?",
        questions=(Question(scope=Q_GROUP, kind=KIND_FACTUAL, options=("C",)),),
        claims=(Claim("the 32 euros", CLAIM_LISTED_FACT, "C"),),
    ),
    SemanticFixture(
        fixture_id="ask_rhetorical",
        category="question_rhetorical",
        text="The Escape Room is quite pricey, right?",
        questions=(Question(scope=Q_RHETORICAL, kind=KIND_PREFERENCE, options=("C",)),),
        concerns=(("C", CONCERN_ORDINARY),),
        notes="Rhetorical tail: no genuine question thread, but the concern is real evidence.",
    ),
    # ------------------------------------------------------------------
    # Answers: full, partial, evasive, unrelated (context = the question turn)
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="answer_full",
        category="answer_full",
        text="It's a six-hour loop, so most of the afternoon.",
        speaker_id="p1",
        context_speaker_id="p2",
        context_text="How long does the Bike Ride take?",
        answer=Answer(completeness=ANSWER_FULL, addresses_target=True),
        context_resolved_options=("B",),
        claims=(
            Claim("a six-hour loop", CLAIM_LISTED_FACT, "B"),
            Claim("so most of the afternoon", CLAIM_INFERENCE, "B"),
        ),
    ),
    SemanticFixture(
        fixture_id="answer_partial",
        category="answer_partial",
        text="It's 32 euros; I'm not sure about rescheduling.",
        speaker_id="p1",
        context_speaker_id="p3",
        context_text="What does the Escape Room cost, and can we reschedule it?",
        answer=Answer(completeness=ANSWER_PARTIAL, addresses_target=True),
        context_resolved_options=("C",),
        claims=(
            Claim("It's 32 euros", CLAIM_LISTED_FACT, "C"),
            Claim("I'm not sure about rescheduling", CLAIM_UNCERTAINTY, "C"),
        ),
    ),
    SemanticFixture(
        fixture_id="answer_evasive",
        category="answer_evasive",
        text="Hard to say — depends how the morning goes.",
        speaker_id="p1",
        context_speaker_id="p2",
        context_text="Mira, does the Museum work with your schedule?",
        answer=Answer(completeness=ANSWER_EVASIVE, addresses_target=False),
        claims=(Claim("depends how the morning goes", CLAIM_UNCERTAINTY, None),),
    ),
    SemanticFixture(
        fixture_id="answer_unrelated",
        category="answer_unrelated",
        text="The Museum has a lovely cafe, by the way.",
        speaker_id="p1",
        context_speaker_id="p3",
        context_text="What does the Escape Room cost?",
        answer=Answer(completeness=ANSWER_UNRELATED, addresses_target=False),
        support=(("A", SUPPORT_WEAK),),
        claims=(Claim("a lovely cafe", CLAIM_OPINION, "A"),),
        notes="Cafe is grounded via the option name 'Museum and Cafe Day'; 'lovely' is opinion.",
    ),
    # ------------------------------------------------------------------
    # Concessions and softening
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="soften_phrase",
        category="softening",
        text="The Bike Ride is starting to make more sense to me.",
        softens_toward="B",
        concession=True,
    ),
    SemanticFixture(
        fixture_id="soften_natural",
        category="softening",
        text="Honestly, you're winning me over on the Museum.",
        softens_toward="A",
        concession=True,
        notes="Natural softening wording outside the current regex vocabulary.",
    ),
    SemanticFixture(
        fixture_id="concession_priority",
        category="concession",
        text="Fair enough — the price matters more than I expected; the Museum has real advantages.",
        concession=True,
        support=(("A", SUPPORT_WEAK),),
        claims=(Claim("the Museum has real advantages", CLAIM_OPINION, "A"),),
    ),
    # ------------------------------------------------------------------
    # Proposals and compromises
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="compromise_question",
        category="proposal",
        text="Could we all live with the Museum as the middle ground?",
        proposes="A",
        questions=(Question(scope=Q_GROUP, kind=KIND_PROPOSAL, options=("A",)),),
        notes="Multi-function: proposal + group question; neither erases the other.",
    ),
    SemanticFixture(
        fixture_id="compromise_statement",
        category="proposal",
        text="Maybe we meet in the middle with the Museum — everyone gets something.",
        proposes="A",
    ),
    # ------------------------------------------------------------------
    # Explicit votes in varied, menu-less natural language
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="vote_direct",
        category="vote_direct",
        text="I vote for the Bike Ride.",
        commitment=(COMMIT_VOTE, "B"),
    ),
    SemanticFixture(
        fixture_id="vote_menuless_has_my_vote",
        category="vote_menuless",
        text="B has my vote.",
        commitment=(COMMIT_VOTE, "B"),
        notes="Known current failure form from the todo: bare label + unlisted phrase family.",
    ),
    SemanticFixture(
        fixture_id="vote_menuless_backing",
        category="vote_menuless",
        text="I'm backing B.",
        commitment=(COMMIT_VOTE, "B"),
        notes="Known current failure form from the todo.",
    ),
    SemanticFixture(
        fixture_id="vote_menuless_put_me_down",
        category="vote_menuless",
        text="Put me down for the Escape Room.",
        commitment=(COMMIT_VOTE, "C"),
    ),
    # ------------------------------------------------------------------
    # Preferences that are NOT votes
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="preference_worth_considering",
        category="preference_not_vote",
        text="B is worth considering.",
        support=(("B", SUPPORT_WEAK),),
        notes="Known current failure form from the todo: must never parse as a commitment.",
    ),
    SemanticFixture(
        fixture_id="preference_lean",
        category="preference_not_vote",
        text="I lean toward the Museum for now.",
        support=(("A", SUPPORT_WEAK),),
    ),
    SemanticFixture(
        fixture_id="conditional_not_vote",
        category="preference_not_vote",
        text="I'd go with the Escape Room only if we can move the booking.",
        support=(("C", SUPPORT_CONDITIONAL),),
        notes="Unresolved prerequisite: commitment wording plus 'only if' is not a vote.",
    ),
    # ------------------------------------------------------------------
    # Vote switches with and without a visible reason
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="switch_with_reason_multi",
        category="switch_with_reason",
        text=(
            "I still dislike the Museum's price, but I'm switching to the Bike Ride "
            "because it's cheaper and more flexible. Would that work for everyone?"
        ),
        concerns=(("A", CONCERN_ORDINARY),),
        commitment=(COMMIT_VOTE, "B"),
        switch=Switch(source="A", target="B", has_visible_reason=True),
        questions=(Question(scope=Q_GROUP, kind=KIND_PROPOSAL, options=("B",)),),
        claims=(
            Claim("it's cheaper", CLAIM_ARITHMETIC, "B"),
            Claim("more flexible", CLAIM_OPINION, "B"),
        ),
        notes=(
            "Canonical multi-function example from the todo: concern + commitment + "
            "switch + reason + group question in one line. The trailing group question "
            "must not void the visible commitment."
        ),
    ),
    SemanticFixture(
        fixture_id="switch_no_reason",
        category="switch_no_reason",
        text="Actually, I'm switching to the Escape Room.",
        commitment=(COMMIT_VOTE, "C"),
        switch=Switch(source=None, target="C", has_visible_reason=False),
        notes="The switch is visible but carries no visible reason — validation decides policy.",
    ),
    # ------------------------------------------------------------------
    # Pronouns: unambiguous public referent vs ambiguous
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="pronoun_clear",
        category="pronoun_unambiguous",
        text="It is the most expensive one, that's true.",
        speaker_id="p1",
        context_speaker_id="p3",
        context_text="The Escape Room worries me a little.",
        concerns=(("C", CONCERN_ORDINARY),),
        context_resolved_options=("C",),
        claims=(Claim("the most expensive one", CLAIM_ARITHMETIC, "C"),),
        notes="Exactly one public referent (previous turn); resolution must be marked context-resolved.",
    ),
    SemanticFixture(
        fixture_id="pronoun_ambiguous",
        category="pronoun_ambiguous",
        text="I think it's the smarter pick.",
        speaker_id="p1",
        context_speaker_id="p2",
        context_text="The Museum and the Bike Ride both stay under budget.",
        ambiguous_reference=True,
        notes="Two plausible public referents: the reference must stay unresolved, no evidence bound.",
    ),
    # ------------------------------------------------------------------
    # Multiple options receiving different evidence in one utterance
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="multi_option_split",
        category="multi_option_evidence",
        text="The Museum keeps the day simple, but the Escape Room's price worries me.",
        support=(("A", SUPPORT_WEAK),),
        concerns=(("C", CONCERN_ORDINARY),),
        claims=(
            Claim("keeps the day simple", CLAIM_OPINION, "A"),
            Claim("the Escape Room's price worries me", CLAIM_OPINION, "C"),
        ),
        notes="Support binds to A only, concern binds to C only — no global spillover.",
    ),
    # ------------------------------------------------------------------
    # Grounding: listed facts, invented details, cross-option transfer,
    # arithmetic, qualified inference, uncertainty, opinion
    # ------------------------------------------------------------------
    SemanticFixture(
        fixture_id="fact_listed",
        category="grounding_listed_fact",
        text="The Museum costs 24 euros and takes about four hours.",
        claims=(
            Claim("costs 24 euros", CLAIM_LISTED_FACT, "A"),
            Claim("takes about four hours", CLAIM_LISTED_FACT, "A"),
        ),
    ),
    SemanticFixture(
        fixture_id="fact_invented",
        category="grounding_invented",
        text="The Museum has free entry on Saturdays, so cost isn't an issue.",
        claims=(Claim("free entry on Saturdays", CLAIM_INVENTED, "A"),),
        notes="Concrete detail not in the scenario: must fail grounding.",
    ),
    SemanticFixture(
        fixture_id="fact_cross_option",
        category="grounding_cross_option",
        text="The Bike Ride costs 32 euros, which is a lot.",
        concerns=(("B", CONCERN_ORDINARY),),
        claims=(Claim("The Bike Ride costs 32 euros", CLAIM_CROSS_OPTION, "B"),),
        notes="32 euros is the Escape Room's price; word-membership grounding would wrongly pass this.",
    ),
    SemanticFixture(
        fixture_id="fact_arithmetic",
        category="grounding_arithmetic",
        text="The Bike Ride is 20 euros cheaper than the Escape Room.",
        comparisons=(Comparison(options=("B", "C"), favored="B", dimension="cost"),),
        claims=(Claim("20 euros cheaper", CLAIM_ARITHMETIC, "B"),),
        notes="32 - 12 = 20: simple reproducible arithmetic must pass grounding.",
    ),
    SemanticFixture(
        fixture_id="inference_qualified",
        category="grounding_inference",
        text="With only 2 hours, the Escape Room probably leaves the evening free.",
        claims=(
            Claim("only 2 hours", CLAIM_LISTED_FACT, "C"),
            Claim("probably leaves the evening free", CLAIM_INFERENCE, "C"),
        ),
    ),
    SemanticFixture(
        fixture_id="uncertainty_budget",
        category="grounding_uncertainty",
        text="I'm not sure the 60 euro budget stretches to dinner too.",
        claims=(
            Claim("the 60 euro budget", CLAIM_LISTED_FACT, None),
            Claim("not sure the 60 euro budget stretches to dinner", CLAIM_UNCERTAINTY, None),
        ),
    ),
    SemanticFixture(
        fixture_id="opinion_fun",
        category="grounding_opinion",
        text="The Escape Room sounds the most fun to me.",
        support=(("C", SUPPORT_WEAK),),
        claims=(Claim("sounds the most fun to me", CLAIM_OPINION, "C"),),
    ),
)


REQUIRED_CATEGORIES: tuple[str, ...] = (
    "support_direct",
    "support_indirect",
    "support_weak",
    "support_conditional",
    "support_firm",
    "concern_keyword",
    "concern_natural",
    "hard_rejection",
    "blocker_resolution",
    "compare_balanced",
    "compare_directional",
    "comparative_question",
    "question_direct",
    "question_group",
    "question_rhetorical",
    "answer_full",
    "answer_partial",
    "answer_evasive",
    "answer_unrelated",
    "softening",
    "concession",
    "proposal",
    "vote_direct",
    "vote_menuless",
    "preference_not_vote",
    "switch_with_reason",
    "switch_no_reason",
    "pronoun_unambiguous",
    "pronoun_ambiguous",
    "multi_option_evidence",
    "grounding_listed_fact",
    "grounding_invented",
    "grounding_cross_option",
    "grounding_arithmetic",
    "grounding_inference",
    "grounding_uncertainty",
    "grounding_opinion",
)


def by_id(fixture_id: str) -> SemanticFixture:
    for fixture in FIXTURES:
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(fixture_id)
