# interpreter

The robot is changing from a companion into an interpreter between a Turi Māori / NZSL signer and a non-signer, in both directions. Pepper speaks and shows emotion through voice and eye colour only. Language is NZ English. Sign recognition and sign generation are other people's modules; this branch holds the decision layer that uses the KG, culture layer and PAD to decide how Pepper speaks and what the sign generator is told.

## Directions

- **Task 1:** signer to non-signer
- **Task 2:** non-signer to signer

## Rule

Interpret, never reply; PAD carries the speaker's emotion, not the robot's persona.

Status: Step 1 (mocks + baseline pipeline) added; awaiting sign-off. Tasks 1 and 2 are pass-through baselines.
