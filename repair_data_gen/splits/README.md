# PMB split workspaces

These directories isolate split-specific configuration, samples, run records,
audits and derived datasets. Pipeline code is shared from
`../shared_pipeline/`; prompts remain immutable under
`../rsr_generation/prompts/` during the transition.

- `train/` indexes the existing frozen train lineage. It does not duplicate or
  move train data.
- `dev/` is the active workspace for development-data construction.
- `test/` is deliberately non-executable until its protocol is frozen.

Keeping outputs separate does not imply that train, dev and test use identical
sampling policies. Their shared transformations and checks must be the same;
caps, marker design, auditing and pair exclusions are split-specific config.
