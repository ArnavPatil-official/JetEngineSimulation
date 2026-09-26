# Phase 5 scratch — superseded, kept for the record

`identifiability_profile_pooled_2026-09-25.py` is the first, never-committed draft of
`scripts/validation/identifiability_profile.py`. It profiled the v4 fuel-flow objective by
re-running the full Cantera cycle inside bounded Powell searches, with `ct.Solution`
globally monkeypatched to a pool of mutable preloaded objects (validated at one point only).
The reviewer stopped its run on 2026-09-25 after several minutes without completing even the
two-parameter pilot; it produced no outputs. Superseded by the closed-form profile of
`docs/plan_phase5_review.md` Phase 2 item 3. Not used by any result.
