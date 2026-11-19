# Amendment 1 to the July to October development evaluation: stop condition 1

Written on 2026-11-19 at about 12:05 London. The two tree settings have been fitted. Nothing of the LEAR, the ensemble or Q
has been fitted or scored on the period, T300's error on the period has not been read, and no comparison has been computed.

## What happened

Stop condition 1 said: T800 on the filled history gives the logged 25.81 on the period within 0.10. It gave **23.56**. The
run stopped there, as the plan requires, and the reason was looked for before anything else.

## The reason

The logged figure was computed on the inputs file as it is, which holds no wind or solar proxy from 2026-08-20 to
2026-10-10. Over those 52 days the tree model was scored without three of its inputs (the two proxies and the residual load
built from them). This plan fills the proxies by the project's own rule before any model reads the file, so its tree model
has them.

| T800, 800 trees of 63 leaves | Inputs as they are | Proxies filled |
|---|---|---|
| Whole period, 2,448 hours | **25.8101** (logged 25.81) | 23.56 |
| 1 July to 19 August, 1,200 hours | 21.25 | 21.25 (forecasts equal to 5.7e-14) |
| 20 August to 9 October, 1,224 hours | 29.46 | 25.35 |
| 1 to 10 October, 240 hours | 36.00 (logged 36.00) | 27.80 |

So the code reproduces both logged figures to the printed digits on the inputs they were computed on, and before the hole
the two versions are the same forecasts. The difference is the hole and nothing else.

The other two tree checks passed: T800 and T300 reproduce the stored June 2026 forecasts to 8.5e-14 and exactly, and the
period has 2,448 strict hours on 102 days.

## What changes

- **Stop condition 1 is replaced** by the check it was meant to be: T800 on the inputs as they are reproduces the logged
  25.81 within 0.10. It does (25.8101). The condition passes in that form.
- **Nothing else changes.** The models, the one filled history every model reads, the three comparisons, the definition
  of distance, the bands and the recurrence rule are as frozen.
- **A finding to report with the results, whatever they are:** the project's logged holdout figures for the tree model
  (25.81, and 36.00 for the crash) were measured with the model missing inputs for half the period. With its inputs the
  same model scores 23.56 and 27.80. Every earlier decision that compared a candidate with 25.81 used that handicapped
  figure. Which of those decisions it could change is not examined here.

Figures of T800 on the period were always going to be read first, by design of the stop condition. The sub-period figures
above were read to find the reason and are part of this amendment.
