# Training instruction versions

Used versions are immutable. SHA-256 instruction hashes cover UTF-8 text without a trailing newline.

| Version | Protocol | Instruction SHA-256 | Protocol SHA-256 |
| --- | --- | --- | --- |
| legacy_canary | [protocol.json](../protocol.json) | `36e7fd0cf17bc43c239087d600c9675787129378d6d282dd7b1a3d48839b7ddc` | `3c0fd7c4d9da6b7e1a9cf11b99b2d5a43a9958d9e90128f415c8d383d2ece9e6` |
| sbn_v2 | [protocol.json](sbn_v2/protocol.json) | `c0eb559f1bc8c356afc3e73d59075cb21999453431af28fb97f3ee3ca5f91384` | `06014c69afc6833cbe9773e49ddc2f7959b0d102eade767b6fbb9d853f61c139` |

The legacy protocol was used by Hábrók canary job 33942287. SBN v2 names the requested output format; data, target mapping and candidate hyperparameters are unchanged. Full-development submission selects SBN v2 explicitly; the default training CLI retains the legacy protocol.
