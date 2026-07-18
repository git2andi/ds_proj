# Choose a dataset for the computer vision course project

The group needs to select a dataset suitable for a computer vision course project involving image classification. The dataset must be publicly accessible, contain labeled images, and be manageable in size for three participants.

## Options

- A) CIFAR-10 Image Dataset — number of images: 60000; image resolution: 32x32; number of classes: 10; license: open (+ Widely used for image classification with balanced class distribution; − Low image resolution may limit complex feature extraction)
- B) Fashion-MNIST Dataset — number of images: 70000; image resolution: 28x28; number of classes: 10; license: open (+ Contains fashion item images that offer a different domain than typical object datasets; − Very low resolution images might restrict model complexity)
- C) Caltech 101 Dataset — number of images: 9144; image resolution: varied; number of classes: 101; license: academic use (+ Diverse object categories with moderate dataset size; − Varied image sizes require preprocessing; smaller dataset size)
- D) Cats vs Dogs Dataset — number of images: 25000; image resolution: varied; number of classes: 2; license: open (+ Binary classification with natural images and moderate dataset size; − Limited to two classes, which may reduce project scope)

## Participants

- **Maya**: preferred B; engagement 3, verbosity 2, directness 3, stubbornness 2
- **Liam**: preferred A; engagement 5, verbosity 3, directness 3, stubbornness 1
- **Zoe**: preferred B; engagement 3, verbosity 1, directness 3, stubbornness 1

## Dialogue

**Moderator:** Today we need to decide: Choose a dataset for the computer vision course project.
**Zoe:** Simple, labeled, and manageable—Fashion-MNIST offers a fresh domain.
**Maya:** Fashion-MNIST works well—fresh domain and not too big.
**Liam:** CIFAR- Image is widely used with balanced classes and open license, ideal for benchmarking.
**Maya:** Low resolution means missing out on key textures I need.
**Liam:** True, but could low resolution actually simplify training by limiting model size?
**Zoe:** Low res can limit model complexity, so it’s tricky.
**Liam:** Both have open licenses, so that doesn’t affect dataset choice for me.
**Liam:** Fashion-MNIST’s domain variety and open license make its size more practical for training.
**Zoe:** Yeah, 10 classes keep it straightforward enough for quick experiments.
**Moderator:** Let’s confirm the final choices.
**Maya:** I choose Fashion-MNIST.
**Liam:** I choose Fashion-MNIST.
**Zoe:** Fashion-MNIST is my final choice.
**Moderator:** Everyone agreed on Fashion-MNIST.

## Outcome

- Status: successful
- Final option: B
- Votes: {'p1': 'B', 'p2': 'B', 'p3': 'B'}
- Reason: All participants voted for the same option

## Core metrics

- participant_count: 3
- participant_turns: 12
- voluntary_turns: 6
- moderator_turns: 3
- moderator_ratio: 0.2
- avg_words_per_participant_turn: 8.92
- visible_preference_changes: 1
- repair_turns: 0
- dropped_turns: 1
- fallback_turns: 0
- response_failures: 0
- protocol_errors: 0
- vote_outcome_consistent: True
- input_tokens: 6872
- output_tokens: 1611
- llm_calls: 13
- voluntary_turns_by_persona: {'p1': 1, 'p2': 3, 'p3': 2}