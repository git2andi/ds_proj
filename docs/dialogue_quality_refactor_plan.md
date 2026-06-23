# Dialogue Quality Refactor Plan

This plan keeps the explicit option board. The options are useful control: they reduce hallucinated alternatives, make state tracking possible, and make evaluation easier. The refactor should target over-control in prompting, repair, routing, and closure behavior.

## Core diagnosis

The simulator currently produces valid decision transcripts, but not sufficiently interactive dialogue. Most runs follow the same visible sequence:

```text
option board -> opening preferences -> compare/support/object -> compromise/vote -> confirmation -> closure
```

This makes topics feel interchangeable. The code has many useful safeguards, but several of them now make the dialogue sound more generic:

- too many style repairs
- too much static persona/state dumped into every prompt
- speaker selection driven by phase and fairness instead of local interaction
- option-card language dominating human reply language
- stance shifts without visible persuasion
- moderator nudges that sound like a fixed meeting script

The target is not more research machinery. The target is actual interaction: people answer, challenge, concede, invite, resist, and change position for visible reasons.

## Design rule: keep full persona, prompt compact persona

Do not remove traits, goals, backstory, or concerns from the data model. They give the agents depth. But do not include the full persona profile in every single turn prompt.

Use two persona representations:

### 1. Full persona profile

Stored in state. Used for consistency and long-range behavior.

Contains:

- name and role
- goal
- preferred option
- acceptable options
- concerns
- backstory
- speaking style
- relevant personality tendencies
- hard blockers or reconsider conditions

### 2. Runtime persona card

Rendered into the prompt for one turn. Short and local.

Contains only:

- current lean
- one active concern
- one speaking habit
- one relevant prior claim
- one relevant relationship/addressee cue
- optional concession state

Example:

```text
Speaker: Kai. Current lean: B.
Active concern: avoids extra setup work.
Speaking habit: asks practical feasibility questions before agreeing.
Already said: he rejected C because it looked too risky.
Now responding to: Ava argued C is safer because venue staff handle setup.
```

This preserves depth while avoiding prompt bloat.

## Implementation steps

### Step 1: Change repair policy

Goal: reduce extra LLM calls and stop repair-generated generic phrasing.

Change:

- Keep repairs for structural errors only.
- Convert style checks to diagnostics.
- Log style failures but do not LLM-repair them.

Structural repair examples:

- missing machine trailer
- invalid option
- malformed vote
- invented option
- hard-blocker violation
- multi-speaker output

Diagnostic-only examples:

- robotic phrase
- repeated opener
- self-narration
- possessive option opener
- awkward closing
- mild semantic repetition

Validation should become less responsible for making text natural. Routing and prompting should create naturalness earlier.

### Step 2: Build `runtime_speaker_card()`

Goal: keep persona depth but reduce per-turn prompt size.

Add a function in `src/prompts.py` or a small helper module:

```python
def runtime_speaker_card(persona, state, intent, recent_turns) -> str:
    ...
```

It should summarize:

- current lean
- active concern relevant to intent.focus_option
- speaking habit
- one previous claim by this speaker
- addressee relationship or local target
- concession bridge if needed

Then replace the full `speaker_card()` in normal turn generation. Keep the full card only for debugging or setup validation.

### Step 3: Add local interaction obligations

Goal: make turns depend on previous turns.

Add state for unresolved interaction obligations:

```python
@dataclass
class InteractionObligation:
    kind: str  # question, challenge, disagreement, confirmation_request
    source_turn_id: str
    source_speaker_id: str
    target_speaker_id: str | None
    option_id: str | None
    claim: str
    resolved: bool = False
```

Populate this when a turn asks a direct question, challenges a claim, or requests confirmation from a holdout.

### Step 4: Change router priority

Goal: route like conversation, not like a checklist.

New priority:

1. Answer pending direct question.
2. Respond to a challenge.
3. Let a relevant participant self-select.
4. Invite a quiet participant only if they have relevant information.
5. Use phase/coverage logic only as fallback.

Add `source_turn_id` and `routing_reason` to `MoveIntent`.

Example:

```python
MoveIntent(
    speaker_id="kai",
    act="answer",
    addressee_id="ava",
    source_turn_id="t12",
    focus_option="C",
    routing_reason="Ava asked Kai whether setup support would remove his blocker."
)
```

### Step 5: Change prompt from board-response to turn-response

Goal: make generated text answer a specific local move.

The prompt should say:

```text
Respond to Ava's previous point: "..."
Your job: answer her setup question and either keep or soften your objection.
Do not restate the full option card.
```

Only provide full option facts when the act requires it, for example compare/vote. For support/object/ask/react, provide only the relevant option facts and prior claim.

### Step 6: Add concession bridge logic

Goal: make stance changes believable.

Before changing `current_preference`, check whether there is a visible reason:

- another participant gave a relevant argument
- a blocker was resolved
- a tradeoff changed
- group priority shifted

If no reason exists, route another discussion turn before allowing acceptance.

Prompt structure for concession:

```text
You previously preferred A because <old reason>.
Ava just argued for C because <new reason>.
If you move toward C, explicitly show what changed.
```

### Step 7: Make closure outcome-specific

Goal: avoid fake consensus language.

Closure rules:

- `consensus`: everyone accepted; brief agreement is fine.
- `fallback`: majority decision; preserve dissent.
- `unresolved`: state deadlock directly.

Example fallback closure:

```text
We'll go with C as the majority pick, but Kai's setup concern is still unresolved, so that needs checking before it is final.
```

### Step 8: Add interaction quality metrics

Goal: measure actual dialogue quality.

Add to eval:

- anchored turn rate
- direct addressee rate
- question-answer rate
- stance-change evidence rate
- semantic repetition rate
- moderator dependency rate
- closure honesty

Use these next to existing structural metrics.

## Suggested order of work

1. Disable LLM repair for style-level issues.
2. Implement compact runtime persona card.
3. Add `source_turn_id` and `routing_reason` to `MoveIntent`.
4. Add interaction obligations to state.
5. Route pending questions/challenges before phase logic.
6. Add concession bridge checks before preference changes.
7. Make closure outcome-specific.
8. Add the new eval metrics.
9. Run the same scenarios and compare against the old logs.

## What not to do yet

Do not remove the option board.
Do not remove persona depth.
Do not add many more research-paper abstractions.
Do not add more repair checks before fixing routing/prompting.
Do not make the moderator responsible for solving every deadlock.

## Link to failures file

The detailed failure tracking lives in:

```text
docs/known_failures.md
```

The most relevant open problems for this refactor are:

- O1: repair layer ineffective on persistent patterns
- O2: repetition and content recycling
- O3: slogan-like utterances / no interactional grounding
- O4: under-motivated stance changes
- O5: consensus vs fallback closure confusion
- O6: mechanical moderator
- O7: weak addressee handling
- O8: awkward closings

The new implementation should close these by changing routing and prompt design, not by adding a larger list of banned phrases.
