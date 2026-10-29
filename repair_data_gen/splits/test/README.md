# Test split: not built

Status: deliberately disabled. Do not generate test candidates or inspect test
outputs while development-data rules and checkpoint-selection inputs are still
being decided.

Before this directory receives an executable config, freeze all of the
following:

- per-type candidate and accepted-item policy;
- matched marked/unmarked construction;
- separation of seen markers from held-out `sorry`;
- replacement-pair exclusion against both train and dev;
- full audit protocol;
- final schema, seeds, manifests and checksums.

Test is not a 30%-marked copy of train/dev: every selected repair site is
intended to receive matched unmarked and marked surface forms. Add its config
only after that protocol is adopted.
