# Test of W, stop condition 4: a proposed exception, DRAFT for the owner

**Status: a draft. Stop condition 4 is not declared met.** Written on 2026-11-23 after the owner's words of 11:22 London.

> I accept the regenerated forecasts as documented reconstructions. Please retain both sets of hashes, quote the original
> wording of stop condition 4, and identify the reference used for the 21,430-hour comparison. If the condition requires
> the registered hashes, record the proposed substitution as an explicit exception and bring that wording back to me
> before declaring the condition met.

## The original wording

From `docs/prereg/spread_w_prereg.md`, under "Stop conditions, before any error of the new window is computed".

> 4. With the new data joined on, the stored forecasts of H, G, C and W on 2024-05-01 to 2026-10-10 reproduce to 1e-9.

The same document identifies the stored forecasts.

> The frozen module reproduces the explored variant to 5.7e-14 on all 21,430 hours (stored forecasts sha256
> ab885a4f3e5fb882a5de62ff4053d58dcbea03faead17d13d64b6b17a6b367cb).

Amendment 3, section 4, point 3, says where the condition holds.

> **Stop condition 4 holds on the reference machine only.** "The stored forecasts reproduce to 1e-9" is a statement about
> the workspace in which they were made. The January rebuild runs there.

## Does the condition require the registered hashes?

In effect, yes. The sentence of the condition names no hash. But "the stored forecasts" are the files the registration
identifies by hash, and amendment 3 ties the condition to the workspace in which they were made. Neither is to hand.

- The stored files were not found in the logger's static folder or in `thermo-fr-handoff` on the laptop.
- The workspace in which they were made is not the workspace of today.

So meeting the condition with regenerated files in another workspace is a substitution, and it is recorded here as an
exception and not as a pass.

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

## The reference used for the 21,430-hour comparison

The regenerated `comparator_predictions.csv.gz` (sha256 6d6a4de11cbb4442...). It holds the
forecasts of H, HW, G, GK, GM, W and C on the 21,430 hours of 2024-05-01 to 2026-10-10, 893 days.

It was compared with the forecasts the draft rebuild code produced on 2026-11-23, in the same workspace, from the same
static files, with the same frozen model functions. The largest difference was 9e-14.

**That comparison is not independent.** Both sides were made on the same day by the same frozen code from the same files.
It shows that the rebuild's join and plumbing do not disturb the forecasts. It does not show that either side equals what
was stored on 2026-11-19.

## What does tie the reconstruction to the originals

1. **The registered design figures.** All 337 numbers in `spread_w_design.json`, `spread_w_comparator_design.json` and
   `spread_eval_stop_checks.json` come out of the regenerated files with a largest difference of 0. They include the
   hourly errors of every model on the backtest and on the opened window, by month and by slice.
2. **One file reproduced to the byte.** `test_predictions.csv.gz`, 2,448 hours of H, G and C, equals the registered file
   apart from the timestamp in its header.
3. **The frozen forecasts of 2026-10-10.** The six forecasts in `static/reference.json`, made on the reference machine,
   are reproduced by today's workspace to 7e-15 by the logger's self-test and to 4e-15 by the rebuild.

What this does not give is an hour-by-hour comparison of W, HW, GK and GM with the stored files on the 21,406 hours that
are not 2026-10-10. For those the evidence is the aggregate figures in point 1.

## Proposed wording of the exception

> **Exception to stop condition 4 of the test of W.** The stored forecasts identified in the registration by sha256
> ab885a4f... and a0a5973c... and the workspace in which they were made are no longer to hand. In their place the
> condition is checked against the reconstructions of 2026-11-23 (file hashes above), made from the hashed static files
> with the frozen design commands in a workspace that passes the logger's self-test against the reference machine. The
> reconstructions reproduce every registered design figure exactly and one registered file to the byte. They do not carry
> the registered hashes of the other two files, and this is not explained. In January 2027 stop condition 4 is met if the
> rebuild reproduces the reconstructions on 2024-05-01 to 2026-10-10 to 1e-9 for H, G, C and W, in a workspace that passes
> the same self-test on the day. HW, GK and GM are checked the same way and reported. The result is reported as "met under
> the exception of 2026-11-23", never as met without qualification, and the read states this beside the decisions.

Nothing is declared met until the owner accepts this wording or changes it.
