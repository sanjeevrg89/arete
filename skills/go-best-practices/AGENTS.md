# AGENTS.md — Go Engineering Standards

> Cross-tool agent instructions (Codex, Cursor, Jules, Amp, and any tool that reads `AGENTS.md`).
> The full, authoritative ruleset lives in **`go-guidelines.md`** next to this file — read it before
> writing or reviewing Go, and apply it. Concrete before/after patterns to imitate are in
> **`examples.md`**. This file is the always-on summary.
>
> **Standard: the Google Go Style Guide.** Five principles, priority order — **Clarity > Simplicity >
> Concision > Maintainability > Consistency.** When they conflict, the higher one wins. Where Google's
> published guide and this summary differ, Google wins.

## When working in Go (`.go` files, `go.mod`), apply these by default:

- **Clarity over cleverness; ship the simplest design that works.** Abstraction must be earned by a
  real, proven need (rule of three) — never speculative.
- **Naming (Google decisions):** initialisms one case (`userID`, `apiURL`); name length ∝ scope;
  receiver = short consistent abbrev of the type; constants describe meaning not value (`MaxSize`, not
  `MAX_SIZE`); no stutter (`widget.New`); named results only for caller clarity, not naked returns.
- **Receivers:** pointer if it mutates / holds a no-copy field / is large; all methods pointer *or*
  all value, not mixed. Context is always the first arg `ctx`; never custom context types.
- **Errors:** `%w` for inspectable chains (place at end), `%v`/canonical translation at system
  boundaries; no empty "it failed" annotations; avoid in-band sentinels (return `value, ok/err`);
  `log.Fatal` only in `main`/startup; panics never cross a package boundary.
- **Tests:** table-driven with field names + `t.Run`; helpers return `error` rather than taking
  `*testing.T`; compare with `google/go-cmp`; never `t.Fatal` from a goroutine.
- **Errors are values.** Handle every error. Wrap with `fmt.Errorf("context: %w", err)` — lowercase,
  no "failed to", no trailing punctuation. Inspect with `errors.Is` / `errors.As`, never string match.
  Sentinels via `errors.New`; structured cases via error types. **Libraries return errors; they do not
  log and do not panic.** `panic` is for programmer bugs only.
- **Every goroutine has an owner and a guaranteed exit.** Plumb `context.Context` as the first arg to
  anything that blocks or does I/O; respect `ctx.Done()`; `defer cancel()`. Bound concurrency
  (worker pool / semaphore / `errgroup.SetLimit`) — never unbounded `go f()` per request.
  Sender closes channels. `go test -race` is mandatory.
- **Accept interfaces, return concrete types.** Define small (1–3 method) interfaces at the consumer,
  not beside their implementation. No interface for a single impl "just in case." Watch the typed-nil-
  in-interface trap.
- **Idiomatic style:** `gofmt` always; MixedCaps; initialisms keep case (`userID`, `apiURL`, `ServeHTTP`);
  no stutter (`user.Service`, not `user.UserService`); getters omit `Get`. Return early, happy path
  left-aligned.
- **Make the zero value useful.** Don't write to a nil map. Mind slice aliasing/`append`; use
  `make([]T, 0, n)` when size is known; full-slice expr `s[a:b:c]` to cap capacity.
- **Resources:** `defer` Close/Unlock/cancel at acquisition; **check writer `Close()` errors**; always
  close `http.Response.Body`; set timeouts on every network call (zero-value `http.Client` never times out).
- **Tests:** table-driven with `t.Run(tc.name, ...)`; `t.Parallel()` for independent tests; behavior
  through public API; `t.Helper()` in helpers; `t.Cleanup()` for teardown; fuzz untrusted parsers;
  inject clock/seed to avoid flakiness; run with `-race`.
- **Security:** validate/bound external input; `crypto/rand` for secrets (never `math/rand`);
  `subtle.ConstantTimeCompare`; parameterized SQL; `html/template` for HTML; never log secrets/PII;
  run `govulncheck`.
- **Generics sparingly:** only for real type-safe reuse across multiple concrete types. Prefer std
  `slices`/`maps`/`cmp`. Don't reinvent the standard library.

## Definition of done for Go changes
All of the following must pass; report honestly if any fail:
`gofmt`/`goimports` · `go vet ./...` · `go build ./...` · `go test -race ./...` · `golangci-lint run`
(and `govulncheck` where configured).

## Reviewing a Go diff
Use the review checklist at the end of `go-guidelines.md`.
