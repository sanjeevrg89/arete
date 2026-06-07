# Validation log

Track which skills have been validated, against what, and what was found. See
[`VALIDATION.md`](VALIDATION.md) for the procedure and the "when is a skill validated?" checklist.

A skill is **validated** only after Layer 5 (real repo/task, correct + lift over baseline) + a
content-accuracy spot-check — not just green CI. "Validated" has a shelf life for fast-moving skills.

| Skill | Date (verify) | Repo / task used | Layers 1–4 | Layer 5 verdict | Issues found | Action / status |
|-------|---------------|------------------|------------|-----------------|--------------|-----------------|
| _example_ `go-best-practices` | 2026-06-07 | acme/api-svc — "review pkg/order" | pass | correct, caught 3 real issues, lift vs baseline | none | ✅ validated |
| _example_ `serving-frameworks` | 2026-06-07 | internal demo — engine pick | pass | mostly right; one flag outdated | a config flag drifted | guide fixed + functional check added; re-validate |
| | | | | | | |

Legend for status: ✅ validated · 🟠 issues (being fixed) · 🔁 needs re-validation (stale) · ⬜ not yet validated.
