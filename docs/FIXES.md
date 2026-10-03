# Phase 8 defects and pending verification

Updated 2026-10-03. User decisions are registered; no further scientific
approval is pending. Registration, one-shot scoring, protected files and no
push remain mandatory. Existing results are preserved.

## Launch blockers to finish before arming

| Item | Required correction | State |
|---|---|---|
| Queue cached benchmark records | Require launch/source identity for new runs and the registered historical-log evidence; revalidate through the shared completion validator | Claude draft, review found gaps |
| Queue end identity | Unreadable or changed end identity invalidates the fresh stage, instead of silently using the start identity | Claude draft, review found gap |
| Queue cached command PASS | Require real non-null log/output hashes and complete evidence before any dependency can pass | Claude draft, review found gaps |
| Queue/G0 tests | Add synthetic lease/completion/cached-evidence/AC/launcher fixtures and fresh-output path checks | Claude session limit; unfinished |
| A4c frozen evidence | Match fit/profile/fallback/current scientific source, input and pre-frozen binary hashes; require associated CSVs/progress logs before reservation | Worktree draft, unfinished |
| A4c singleton profile | Directly evaluate each fixed grid point when there are no nuisance parameters; retain exact penalty guard | Worktree draft, unfinished |
| Input convergence identity | Freeze before compute, reject end drift; coordinate by main records, with an explicitly serialized validation-stage exception | Worktree draft, unfinished |
| Input-only flow domain | Register and implement opt-in public-input numerical flow bound; default keeps old bounds; separate build option and integration assertion | Registered C1 9e4db00; implementation pending, no solve |
| OAT failed-design label | Check the legacy design dictionary's converged field, not dictionary truthiness | Worktree draft, unfinished |
| Architecture provenance | Cite verified primary sources or explicitly label unverified diagnostic architecture proxies | Registered C1 9e4db00; code/document integration pending |

Both executor sessions ended at Claude's shared session limit, resetting at
10:10 pm America/New_York. The unfinished queue implementation is isolated in
`phase8-queue-recovery-20261003`; the main wrapper is preserved. Neither
the replacement chain nor new scientific commands are armed. Numerical checks run on AC after review.

## Non-blocking defects and limits

- Old benchmark records lack clean power evidence in two timings and have
  registered overlap flags. Preserve them and report flags; the two existing
  rerun jobs are the only supplementary benchmark jobs registered.
- A4c coefficient/efficiency/cooling envelopes are declared engineering
  ranges informed by cited evidence, not measured Trent confidence intervals.
- Error-budget shares are calibration-engine responses normalized by known
  aggregates; they are not an additive allocation of held-out error.
- The all-family check is representative design convergence, with public
  input records and declared architecture proxies; it is not geared
  off-design validation or empirical engine validation.
- Track 4 Ma Eq. 25 thermal-unit ambiguity remains flagged. Diagnostic
  scores, loss balancing and future empirical/Sajben work retain their
  previously registered scope and order.
