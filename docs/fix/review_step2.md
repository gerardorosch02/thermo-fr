# Review of step 2 (radiation label fix F) and freeze

2026-10-18, reviewer. I recomputed everything from the saved prediction files in scratchpad/fix/base and fix/F, using my own code (step1rev/step2_check.py). That includes my own implementation of the S2 weights. No price backtest was run.

**Verdict: SIGN.** The bar was applied exactly as pre-registered, and both decisions reproduce.

## Checks
- **Bases.** honest_v2, LEAR and S2 equal the logged sprint predictions to 6e-14.
- **Rows.** The index, the strict flags (including LEAR's), the targets and the 20,752 strict hours on 865 days are identical across base and F. No F member has a missing value.
- **Slice day sets.** They come from the logged basis. My slice MAEs for the base reproduce the logged ones (15.01, 18.88, 17.66, 15.11, 22.31).

**honest_v2-F against honest_v2: FAILS.**
- MAE 15.7040 to 15.7308.
- Paired mean +0.0266, two-sided 95% interval [-0.0960, +0.1423].
- One-sided 95% upper bound (7-day blocks, 5,000 resamples, seed 0) +0.1229, above +0.10. With seeds 1 to 4 it is 0.121 to 0.124, so the result does not depend on the seed.
- Slices: weekends -0.01, holidays -0.22, windiest -0.22, sunniest +0.08, top 5% +0.32, 15 to 19 UTC +0.08.

**S2-F against S2: PASSES all conditions.**
- MAE 14.3192 to 14.3086.
- Paired mean -0.0107, two-sided [-0.0640, +0.0398], upper bound +0.0315 (0.030 to 0.033 across seeds).
- Slices: weekends -0.02, holidays -0.16, windiest -0.01, sunniest -0.10, top 5% +0.12, 15 to 19 UTC +0.05.

**Note for the owner's adoption decision.** S2-F's gradient boosting member is honest_v2-F, which failed alone. Adopting S2-F while the public honest_v2 stays on the old labels would mean two model files, and two input bases in the morning job. The release step must plan for that, or for S2-F replacing both.

## Freeze (fix_prereg.md section 4 item 5), recorded 2026-10-18 before any shadow score is read
| Item | sha256 or hash |
|---|---|
| fix_prereg.md | 6b9f13952ac7d17d60e72f20db29c446415ae10510ec56bca823cf25af3d61a3 |
| Code commit (solar-align) | 3b92475e76cdbf3f9ccd43508b6ead5d2e79c029 |
| src tree at that commit; `git archive 3b92475 src` sha256 | 47670a8747ad04e7b652f8e1cb05b88d1e3768c1; 2f854a31654c79ccec3bc14cc3d578cedd5fb036c59cfd4fff92d32a62187d15 |
| S2-F model-file rebuild rule, fix/replay_rebuild_rule.md | a40cf7b0c35d9816b961b1f430767afb62e30095aae9e5762f6cd73ef6613cec |
| Stage scripts: fix_stage.py, lear_stage_aligned.py, common.py | e4212086…48e36, fe8f6d2c…5c22c, 88e34dd4…4f020 |
| inputs_aligned.csv | b7ba09b22b71c4cab8a7c7c35586992674f453dc258b91c211433a0ce6e5fb58 |

- **Rebuild rule.** fix/replay_rebuild_rule.md writes out the rule that the pre-registration states in one sentence. It names the code path and settings: corrected history plus converted windows; honest_v2-F rebuilt to each published file's training window with 300 trees and 31 leaves; LEAR-F refitted daily; arm A's weight-state window; and the comparison rule. The designer should confirm it. Any change before day 40 voids the freeze and is logged as a deviation.
- **Shadow data.** I read no `shadow_*` table and no shadow-settle output, and I read no shadow scores. The only shadow code I read was shadow.py's source, for the weight-state constants.
- **Worktree.** docs/experiments.md is modified in the worktree. That is outside src and not part of the frozen code.

## Freeze reset, 2026-10-18
The designer amended fix/replay_rebuild_rule.md (fallback-day scoring, weight-state settings, release question out of scope) before any shadow_* table, score or settle output was read by anyone. The previous rule hash a40cf7b0... is void. New rule hash: 1fe2660e655cf9d4b6d02c44636a04327e172878cc777159d955f840a768d434. fix_prereg.md (6b9f1395...) and commit 3b92475 are unchanged.
