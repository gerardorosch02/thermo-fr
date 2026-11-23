# Test of W, exception to stop condition 4

In force from the owner's word of 2026-11-23, 11:30 London. Asked "Do you accept the exception wording, or want it
changed?", he answered "1. yes". The wording is the one put to him in `spread_w_stop_condition_4_exception_draft.md`
(sha256 in `docs/experiments.md`), unchanged, and that draft holds the reasoning and what the evidence does and does not
show.

Stop condition 4 has not been run on the test window and is not met yet. This document says how it will be judged in
January 2027. No forecast of the window has been scored and no error against any price has been computed.

## The wording in force

> **Exception to stop condition 4 of the test of W.** The stored forecasts identified in the registration by sha256
> ab885a4f... and a0a5973c... and the workspace in which they were made are no longer to hand. In their place the
> condition is checked against the reconstructions of 2026-11-23 (file hashes above), made from the hashed static files
> with the frozen design commands in a workspace that passes the logger's self-test against the reference machine. The
> reconstructions reproduce every registered design figure exactly and one registered file to the byte. They do not carry
> the registered hashes of the other two files, and this is not explained. In January 2027 stop condition 4 is met if the
> rebuild reproduces the reconstructions on 2024-05-01 to 2026-10-10 to 1e-9 for H, G, C and W, in a workspace that passes
> the same self-test on the day. HW, GK and GM are checked the same way and reported. The result is reported as "met under
> the exception of 2026-11-23", never as met without qualification, and the read states this beside the decisions.

The file hashes the wording refers to are in the table below.

## The original wording, kept

> 4. With the new data joined on, the stored forecasts of H, G, C and W on 2024-05-01 to 2026-10-10 reproduce to 1e-9.

## Both sets of hashes

| File | Registered sha256 | Regenerated file sha256 | Regenerated, uncompressed csv sha256 | Relation |
|---|---|---|---|---|
| `test_predictions.csv.gz` (H, G, C on the opened window) | c197bab143d014432fa9c4ed245cf511bb7ee6149d005f1a5d54450e4969c280 | 2fb1e07b2482ad3b63c997222069b0c229d1f8f18d3f5a2af3d23a6b1ae62432 | 6c30810f07870cdda2741d9da17f769a69388d5363b2d6813dd05a287d407394 | equal byte for byte once the gzip timestamp is set to 2026-11-19 16:41:02 UTC |
| `w_predictions.csv.gz` (H, G, C, W, 21,430 hours) | ab885a4f3e5fb882a5de62ff4053d58dcbea03faead17d13d64b6b17a6b367cb | f82df0ad123319e2e0d55c7e684f8dcec9057f215927d0c1f01bda10fb966efb | abf754e467c12e74eea0bca39ae7f4a9e0d28109766ec6be1089a450ad77c2ef | registered hash not reproduced; cause not known |
| `comparator_predictions.csv.gz` (all seven, 21,430 hours) | a0a5973ccb1bc836daafe95e64c74f45a266cf840437c1df372d4792dd4b8b5f | 6d6a4de11cbb4442b0c4a9569f897144eef5d1a6131faf8658acf5d60b80ffdf | 373232965338ae41abd24db4bb34e56337f89da25b81fc7dfed698f0258694ea | registered hash not reproduced; cause not known |
| `dev_predictions.csv.gz` (H, G, C on the backtest) | none found in the registration | 503e8244f5991e50bdc27f9f007d948fa9b444a7c7133763d4f3eeba130a0b44 | e179d0cc401c0f85f2b335f5bc8f685da3b1ba6dabc7f60e9713bd4fb088c4c6 | nothing to compare with |

The regenerated files are on the laptop in `thermo-fr-spread\kept_forecasts_2026-11-23` with `HASHES.json`. For the two
files whose registered hash was not reproduced, a timestamp alone was tried for every second from 2026-11-18 12:00 to
2026-11-22 00:00 UTC. The files may differ in layout, or have been written outside those hours. This is not resolved.


## How the code applies it

`src/thermo_fr/rebuild/spread_w_rebuild.py`, function `reproduce_check`. The condition passes only if all of these hold.

- The file of stored forecasts given to it is the reconstruction of 2026-11-23, checked by file hash.
- The workspace has passed the logger's self-test, and the output of that run is given to it.
- H, G, C and W agree with the reconstruction on every stored hour to 1e-9.

HW, GK and GM are compared the same way and reported. The record carries the words "met under the exception of
2026-11-23", and the read must state them beside the decisions.

## Also decided on 2026-11-23, 11:30 London

Asked whether he wanted a second set of figures using a realised German price from the second source, reported beside
the result and deciding nothing, the owner answered "2. no second set of figures". So the point left open at the end of
amendment 6 is closed. No figure in this test uses a realised price from a second source.
