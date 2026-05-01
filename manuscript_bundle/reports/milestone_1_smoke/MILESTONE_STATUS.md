# Milestone status: Milestone 1 — Experiment registry and result schema lock

| Field | Value |
|-------|-------|
| Status | PASS |
| Experiment id | milestone.smoke_lock |
| Manifest schema | 1 |
| Output convention | esl_canonical_v1 |

## Tests executed

pytest -q (126 passed, full suite)

## Artifacts

Copied canonical files under:

`/Users/jainendra/Documents/Cursor/ESL/manuscript_bundle/reports/milestone_1_smoke/artifacts`

## Open issues / caveats

- `runs/milestone1_smoke` is gitignored; regenerate with `esl.experiments.registry_smoke.run_registry_esl('milestone.smoke_lock', run_dir, ...)`.
- Direct `run_esl(...)` calls (outside `run_named_preset` / `run_registry_esl`) still omit `manifest.json`; Milestone 2+ should route paper runs through the registry helpers or extend the trainer once.

## Paper-faithfulness check

Canonical manifest + metrics identifiers follow PRD §10–11; trainer trajectory keys match PRD §11.

## Readiness for next milestone

Stop after M1; await explicit confirmation before Milestone 2.
