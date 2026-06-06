---
name: go-best-practices
description: World-class Go guidelines aligned to the Google Go Style Guide (Style Guide, Style Decisions, Best Practices) plus community consensus, for writing, reviewing, and refactoring Go. Use whenever working in a Go codebase (.go files, go.mod) — authoring packages/APIs, handling errors, concurrency/goroutines/context, interfaces, testing, performance, or reviewing a Go diff/PR. Covers Google's clarity>simplicity>concision>maintainability>consistency hierarchy, naming/receiver/error decisions, %w wrapping, goroutine lifecycle, table-driven tests with go-cmp, and a review checklist.
---

# Go Best Practices

Apply distinguished-engineer Go judgment, **held to the Google Go Style Guide bar**, by default on any
Go work. Google's five principles in priority order: **Clarity > Simplicity > Concision >
Maintainability > Consistency** — when they conflict, the higher one wins.

## How to use this skill

1. **Read `go-guidelines.md`** in this skill directory — it is the full ruleset (the single source
   of truth). Apply it to whatever Go task is at hand. For concrete before/after patterns to imitate
   (error wrapping, receiver consistency, table tests with `go-cmp`, options, goroutine lifetime),
   read **`examples.md`**.
2. Match the surrounding codebase's existing conventions where they conflict with a stylistic
   preference here; apply the correctness/concurrency/error rules regardless.
3. Before declaring Go work complete, ensure these pass and report honestly if any don't:
   `gofmt`/`goimports` · `go vet ./...` · `go build ./...` · `go test -race ./...` · `golangci-lint run`.

## The essentials (full rationale in `go-guidelines.md`)

- **Clarity over cleverness; simplest design that works.** Add abstraction only on real, proven need.
- **Errors are values.** Handle every one; wrap with `fmt.Errorf("...: %w", err)` (lowercase, no
  "failed to"); inspect with `errors.Is`/`errors.As`. Libraries return errors — they don't log or panic.
- **Every goroutine needs an owner and a guaranteed exit.** Plumb `context.Context` (first arg) for
  I/O/blocking work; bound concurrency; `go test -race` is mandatory.
- **Accept interfaces, return concrete types.** Define small interfaces at the consumer, not next to
  the implementation. No interfaces "just in case."
- **Idiomatic naming:** MixedCaps, initialisms keep case (`userID`, `apiURL`), no stutter, getters
  omit `Get`. `gofmt` is non-negotiable.
- **Make the zero value useful;** mind nil-map writes and slice aliasing; set timeouts on all network
  calls; `defer` Close/cancel and check writer `Close` errors.
- **Tests are table-driven**, behavior-focused, `t.Run`/`t.Parallel`, race-clean, no time/rand flakiness.
- **Security:** validate external input, `crypto/rand` for secrets, parameterize SQL, never log secrets.

When reviewing a Go diff, use the review checklist at the end of `go-guidelines.md`.
