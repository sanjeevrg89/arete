# Go Engineering Guidelines

Production-grade Go guidance aligned to the **Google Go Style Guide** (Style Guide · Style Decisions ·
Best Practices) and the wider Go community consensus, written from the perspective of a distinguished
engineer. This is the single source of truth; the tool-specific entry files (`SKILL.md`, `AGENTS.md`,
`GEMINI.md`) all defer to this document.

> Apply these by default when writing, reviewing, or refactoring Go. They are defaults, not dogma —
> when you deviate, say why in a comment or the PR description. Where this document and the Google Go
> Style Guide differ, **Google's published guide wins** for code intended to meet that bar.

---

## 0. Core principles (Google's hierarchy)

Google defines five principles **in priority order**. When two conflict, the higher one wins —
clarity beats concision, simplicity beats consistency, and so on.

1. **Clarity** — the code's purpose and rationale are clear *to the reader*, not just the author.
   Judge clarity through the reader's eyes. Cover both *what* the code does and *why*.
2. **Simplicity** — accomplish the goal in the simplest way, in behavior and performance. Prefer the
   boring solution. Add abstraction only after a real second use case proves it (rule of three).
3. **Concision** — high signal-to-noise ratio; the relevant details are easy to discern. Remove
   noise, but never at the expense of clarity.
4. **Maintainability** — written so it can be easily changed correctly: good names, tests, no hidden
   coupling, no surprising action-at-a-distance.
5. **Consistency** — consistent with the surrounding code and the broader codebase. Match local
   convention over personal preference; this is the lowest-priority principle, so it yields to the
   four above.

Supporting maxims (community + std lib):
- **The standard library is the reference implementation** — match its style and patterns first.
- **Errors are values** — handle them like data, not exceptions.
- **A little copying is better than a little dependency.**
- **Make the zero value useful.**
- **Concurrency is not free** — every goroutine needs an owner and a guaranteed exit.

Non-negotiable gates before you call code "done": `gofmt`/`goimports`, `go vet`, `go build ./...`,
`go test -race ./...`, and a linter (`golangci-lint`). If any fail, the work isn't finished.
`gofmt` is not a matter of opinion — all source must match its output.

---

## 1. Project & package layout

- Keep `main` packages thin: parse flags/config, wire dependencies, call into a library package, exit.
  Business logic lives in importable packages so it is testable without spawning a process.
- Package name = its purpose, lowercase, no underscores, no plurals, no `util`/`common`/`helpers`
  grab-bags. The name is part of every call site: `chi.NewRouter()`, not `chiutils.New()`.
- Avoid stutter: in package `user`, call it `user.Service`, not `user.UserService`.
- Use `internal/` to make packages importable only within your module. Default new packages to
  internal unless you intend to publish them as API.
- Don't over-adopt `pkg/` and elaborate "standard layouts" for small services — flat is fine.
  Add structure when the package count, not the line count, demands it.
- One package per directory; the directory name should match the package name.
- Organize by domain/feature, not by technical layer (`order/`, not `models/ controllers/ services/`).
- **Import grouping** (goimports enforces): (1) standard library, (2) other project/vendored packages,
  (3) protobuf imports, (4) blank/side-effect imports. **Blank imports (`import _`) are allowed only in
  `main` or test packages, never in a library package.** Dot imports (`import .`) only in tests.

---

## 2. API & package design

- **Design the call site first.** Write the code you wish you could call, then implement it.
- Export the minimum. Unexported by default; you can always promote later, you can't easily demote.
- **Accept interfaces, return concrete types.** Callers get flexibility; you keep the freedom to add
  methods without breaking them.
- Keep exported surfaces stable. Once `v1`, treat exported names as a contract (semver).
- Prefer many small, focused functions over one with a boolean/`mode` parameter that changes behavior.
- **Keep argument lists short.** When a signature grows, first try splitting the function into simpler
  ones. If options are genuinely needed, choose by caller profile:
  - **Option struct** (last parameter) when *most callers set one or more options* — self-documenting
    fields, defaults omittable.
  - **Variadic functional options** when *most callers need none* and there are many options:
    ```go
    func New(addr string, opts ...Option) *Server
    ```
    Options should accept a parameter rather than using mere presence to signal a value.
  Reserve options for genuinely optional knobs; required deps go as explicit positional args.
- **Constructors:** use `NewX`. If the zero value is already useful, prefer declaring `var x T` over a
  zero-filled composite literal (`Point{X: 0, Y: 0}`); for pointers use `new(T)` or `&T{}`. Use
  `MustX` constructors only for startup/package-var init or tests — never on user input.
- Return early; keep the happy path at minimum indentation (left-aligned). Handle errors and edge
  cases first and `return`.

---

## 3. Naming & style

- MixedCaps / mixedCaps, never snake_case (the only exception is generated-code package names).
  Exported = leading uppercase, unexported = leading lowercase.
- **Initialisms keep one consistent case:** `userID`, `ServeHTTP`, `apiURL`, `parseXMLAPIResponse` —
  not `userId`, `apiUrl`, `parseXmlApiResponse`. Each initialism is all-upper or all-lower as a unit.
  Mixed-case initialisms (`gRPC`, `iOS`) follow prose, adjusting only the first letter for export.
- **Name length is proportional to scope and inversely proportional to use count.** `i`, `r`, `buf`
  in tight scopes; descriptive names for package-level identifiers. A heavily-used var earns a short name.
- **Omit type/type-like words from names:** `users` not `userSlice`, `userCount` not `numUsers`,
  `u` not `userVar`.
- **Receiver names** are short (1–2 letters) abbreviations of the type, applied *consistently* across
  every method of that type: `func (t Tray)`, not `func (tray Tray)` or mixing `t`/`this`/`self`.
- Getters omit `Get`: `u.Name()`, `pkg.Counts()` — not `GetName`/`GetCounts`. Use `Fetch`/`Compute`/
  `Load` to signal expensive work. Setters keep `Set`.
- **Constant names describe what the value *denotes*, not the value itself.** `MaxPacketSize`, not
  `MAX_PACKET_SIZE` or `K64`. MixedCaps like everything else — no SCREAMING_SNAKE, no `k` prefix.
- **Avoid repetition (no stutter):** `widget.New`, not `widget.NewWidget`; `user.Service`, not
  `user.UserService`. Don't repeat the package name, receiver type, or argument types in a name —
  prefer `Parse` over `ParseYAMLConfig` when context makes it obvious.
- Interface naming: single-method interfaces take the method name + `-er` (`Reader`, `Closer`).
- **Named result parameters** only when they add clarity about caller actions (e.g. documenting which
  of two same-typed results is which), *not* just to enable naked returns. Avoid naked returns in
  anything but the shortest functions.
- Comments are full sentences. Every exported identifier gets a doc comment **beginning with the
  identifier name** and an article (`// Server handles incoming requests.`). Package comment sits
  immediately above `package` with no blank line, one per package. Explain *why*, not *what*.
- **Line length:** there is no fixed limit. If a line feels too long, refactor rather than mechanically
  splitting. Don't line-break an `if` condition or a function signature just to fit a column.

### Receiver type: pointer vs value (Google's rule)
- Use a **pointer receiver** when: the method mutates the receiver; the struct holds a field that must
  not be copied (`sync.Mutex`, etc.); the struct is large; or it holds a pointer to mutable state.
- Use a **value receiver** for small, naturally-value types, and for maps/funcs/channels/slices that
  aren't resliced/reallocated.
- **Be consistent:** make a type's methods *all* pointer or *all* value, not a mix. When in doubt, use
  a pointer receiver.

---

## 4. Error handling

This is where most Go code lives or dies. Be disciplined.

- **Handle every error explicitly.** Never `_ =` an error unless you write a comment justifying it.
- Add context as you go up the stack with `%w`, lowercase, no trailing punctuation, no "failed to":
  ```go
  if err != nil {
      return fmt.Errorf("load config %q: %w", path, err)
  }
  ```
  The wrapped chain reads top-down like a breadcrumb trail. Don't repeat context the caller already adds.
- **`%w` vs `%v` is a deliberate choice.** Use `%w` when you want callers to programmatically inspect
  or unwrap the cause (`errors.Is`/`errors.As`); use `%v` for simple annotation, or at a **system
  boundary** (RPC/IPC) where you intentionally sever the chain and translate to a canonical error
  space (e.g. gRPC status codes) rather than leaking internal error types.
- **`%w` placement:** put it at the **end** of the message (`"load config %q: %w"`). The exception is
  sentinel-category errors where leading with `%w` identifies the class up front.
- **Don't add empty context.** Skip an annotation whose only purpose is to say "this failed" without
  adding new information the underlying error doesn't already carry — it just makes messages redundant.
- Inspect with `errors.Is` (sentinel) and `errors.As` (typed), never string matching.
- **Give errors structure for callers who need it:** sentinels comparable with `errors.Is`
  (`var ErrNotFound = errors.New("not found")`), or error *types* (implement `Error()`) when callers
  need structured fields. Don't over-engineer errors no caller inspects.
- **Libraries return errors; they do not log and do not panic.** Logging is the caller's decision;
  doing both double-reports. Use `log.Fatal` only for program-initialization failures or violated
  invariants where internal state is unrecoverable — and only in a binary's `main`/startup, never in a
  reusable library, especially not for transient errors.
- `panic` is for API misuse analogous to a language-level bug (out-of-bounds), or as a package-internal
  control-flow detail that **always has a matching `recover` in the same package** — panics must
  **never escape across a package boundary**. Recover only at well-defined boundaries (request handler,
  worker loop) to convert an unexpected panic into a 500 + log; re-panic if it isn't yours.
- **Avoid in-band error signaling** (returning -1, "", or nil to mean "error"). Return an additional
  `bool`/`error` (`value, ok` / `value, err`) so the validity of the result is explicit.
- For multi-error aggregation use `errors.Join`.

---

## 5. Concurrency

The hardest part of Go to get right. Default to *not* using goroutines until you need them.

- **Every goroutine must have a clear owner and a guaranteed way to stop.** If you can't point to
  where it exits, you have a leak.
- **Don't start a goroutine without knowing how it ends.** Tie lifetime to a `context.Context`,
  a `sync.WaitGroup`, or `errgroup.Group`.
- Pass `context.Context` as the first parameter (`ctx context.Context`) to anything that blocks,
  does I/O, or spawns work. Respect cancellation: select on `ctx.Done()`.
- **Channels orchestrate; mutexes protect.** Use channels to pass ownership/signal; use `sync.Mutex`
  to guard shared in-memory state. Don't build elaborate channel choreography where a mutex is clearer.
- The sender closes a channel, never the receiver. Closing signals "no more values."
- Prefer `golang.org/x/sync/errgroup` for fan-out where any error should cancel siblings and you
  need to wait for all. Use `errgroup.SetLimit` to bound concurrency rather than unbounded goroutines.
- Never share a `sync.Mutex`, `WaitGroup`, or any struct containing them by value — pass pointers.
  `go vet` catches copied locks.
- Bound concurrency. Unbounded `go f()` per request is a classic outage (memory, FD, downstream load).
  Use a worker pool or a semaphore (`golang.org/x/sync/semaphore` or a buffered channel).
- **`-race` is mandatory in CI.** A data race is always a bug, even if "it works."
- Avoid the loop-variable capture bug. (Fixed in Go 1.22+ where each iteration gets a fresh variable;
  on older toolchains, shadow with `i := i`.) Know which toolchain you target.
- `time.After` in a `select` loop leaks until it fires — use a `time.Timer` you can `Stop`, or a
  `context` deadline, in hot loops.

---

## 6. Context

- Context carries deadlines, cancellation, and request-scoped values across API boundaries — nothing else.
- **`context.Context` is always the first parameter**, named `ctx`. The one exception is HTTP handlers,
  which get it from `req.Context()`.
- **Never store a `Context` in a struct;** pass it explicitly through the call chain.
- **Never create custom context types or wrap `context.Context` in your own interface** — this is
  absolute in Google's guide. Use `context.Context` as-is.
- `context.Background()` belongs only in entrypoints (`main`, `init`, tests). Use `context.TODO()`
  when you don't yet have one to thread; never pass `nil`.
- Providing a context implies cancellation should interrupt the function and return `ctx.Err()`.
  Document only when behavior deviates.
- `context.WithValue` is for request-scoped data that crosses API boundaries (request ID, auth
  principal, trace span) — never for optional function parameters or to avoid plumbing. Use a private
  key type to avoid collisions.
- Always `defer cancel()` for `WithCancel`/`WithTimeout`/`WithDeadline`, even when the work finishes
  early — not cancelling leaks the timer/context.

---

## 7. Interfaces & abstraction

- **Define interfaces where they're consumed, not where the implementation lives.** The consumer
  knows what it needs; declaring it there keeps interfaces small and avoids premature coupling.
- Keep interfaces small — one to three methods. Big interfaces are hard to implement and mock.
- Don't create an interface for a single implementation "just in case." Add it when you have a second
  implementation or a real test-double need. Premature interfaces are abstraction debt.
- Don't return interfaces from constructors unless the abstraction is the point; return the struct.
- Accept the narrowest interface that does the job (`io.Reader`, not `*os.File`).
- A `nil` interface is not a `nil` pointer wrapped in an interface — returning a typed nil pointer as
  an `error`/interface makes `err != nil` true. Return a literal `nil`.

---

## 8. Data, slices, maps, memory

- The zero value should be usable where possible: a `bytes.Buffer{}` is ready to write; a
  `sync.Mutex{}` is unlocked. Design your structs the same way.
- Reading a `nil` map is fine; writing panics. Initialize before writing (`make` or literal).
- **Slices share backing arrays.** `append` may or may not allocate; a sub-slice aliases its parent.
  Copy when you need independence; use full-slice expressions `s[a:b:c]` to cap capacity and prevent
  surprise aliasing when handing a slice to untrusted code.
- Preallocate with `make([]T, 0, n)` when the size is known — avoids repeated growth/copy.
- Don't hold a small sub-slice of a huge array; it pins the whole backing array in memory. Copy out.
- Pass large structs by pointer to avoid copies, but prefer values for small, immutable data — values
  are simpler, escape less, and are safe to share.
- Strings are immutable UTF-8 bytes; ranging yields runes, indexing yields bytes. Use
  `utf8.RuneCountInString` for character counts, not `len`.
- Prefer `strings.Builder` for incremental string construction; never `+=` in a loop.

---

## 9. Resource management

- `defer` Close/Unlock/cancel immediately after acquiring the resource — colocate acquisition and release.
- **Check the error from `Close()` on writers** (a failed flush loses data). `defer f.Close()` silently
  drops it; for files you write, close explicitly and check, or use a named return + deferred closure.
- `defer` runs at function return, not block end — don't `defer` inside a loop expecting per-iteration
  release; extract a function or call explicitly.
- Always `resp.Body.Close()` on `http.Response`, even on non-2xx, or you leak connections.
- Set timeouts on everything that talks to the network: `http.Client.Timeout`, server
  `ReadHeaderTimeout`/`WriteTimeout`, DB connection/query deadlines. The zero-value `http.Client`
  has *no* timeout and will hang forever.

---

## 10. Performance

- **Measure before optimizing.** Write a benchmark (`func BenchmarkX(b *testing.B)`), profile with
  `pprof`, then act on data. Intuition about Go performance is frequently wrong.
- Use `go test -bench=. -benchmem` and watch allocations — allocation count is often the real lever.
- `b.ResetTimer()` after setup; `b.ReportAllocs()`; mark parallel benchmarks with `b.RunParallel`.
- Reduce allocations in hot paths: reuse buffers (`sync.Pool` for short-lived large objects),
  preallocate slices, avoid `interface{}` boxing and unnecessary `[]byte`↔`string` conversions.
- Don't micro-optimize cold code. 99% of code should optimize for readability; profile to find the 1%.
- Prefer streaming (`io.Reader`/`io.Writer`) over loading whole payloads into memory.
- Know escape analysis basics (`go build -gcflags=-m`) but don't contort code to fight it without a profile.

---

## 11. Testing

- **Table-driven tests are the default.** A slice of named cases with a single assertion loop.
  Name the field `name` and use `t.Run(tc.name, ...)` for sub-tests and targeted runs.
- Use `t.Parallel()` for independent tests; capture the loop variable correctly on older toolchains.
- Prefer the standard library + minimal helpers. `testify/require` is acceptable for terse assertions
  on teams that already use it; don't introduce heavy BDD frameworks. `require` stops on failure,
  `assert` continues — use `require` for preconditions.
- Test behavior through the public API, not internals. Avoid asserting on private fields.
- Use `t.Helper()` in setup/teardown helpers so failures point at the calling test line. A helper
  may call `t.Fatal` only for setup the test can't run without.
- **Don't write assertion helpers that take `*testing.T`.** Per Google, a shared validation function
  should *return a value* (typically `error`) and let the test decide how to report — keeps failure
  reporting orthogonal. `testify/require` is acceptable on teams already standardized on it.
- **Compare with `github.com/google/go-cmp/cmp`** (`cmp.Diff`/`cmp.Equal`) for structs/slices/maps —
  it returns a readable diff and supports options; don't hand-roll `reflect.DeepEqual` assertions.
- **Never call `t.Fatal`/`t.FailNow`/`t.Helper` from a spawned goroutine** — use `t.Error` + `return`.
  `FailNow` only works on the test's own goroutine.
- Golden files for large outputs, gated behind a `-update` flag.
- `t.Cleanup()` over `defer` for test teardown — it composes with helpers and parallel subtests.
- **Fuzz** parsers and anything taking untrusted bytes (`func FuzzX(f *testing.F)`).
- Run `go test -race ./...` in CI. Race-free is part of "passing."
- Mock at the boundary (an interface you own), not deep internals. Prefer fakes/in-memory
  implementations over generated mock spaghetti where practical.
- Avoid time and randomness flakiness: inject a clock and a seed/`rand.Rand` rather than calling
  `time.Now()`/global rand directly in logic under test.

---

## 12. Modules & dependencies

- One module per repo for most services; multi-module only when you genuinely publish independent
  units with separate version cadences.
- Keep `go.mod` `go` directive accurate; commit `go.sum`. Run `go mod tidy` before pushing.
- Vet every dependency: maintenance, transitive footprint, license, security history. Each dep is a
  long-term liability you'll patch and audit.
- Pin tool versions (linters, codegen) via a `tools.go` + module, or a versioned CI image — don't rely
  on "whatever's installed."
- Prefer std-lib and `golang.org/x/...` before third-party. Avoid frameworks that take over `main`.

---

## 13. Observability

- Use **`log/slog`** (std lib, structured) for new code. Structured key/value logs, not `fmt.Printf`.
- Log at boundaries with context; don't log-and-return the same error (pick one owner — usually the
  top of the request handler).
- Pass a logger via dependency injection or `slog.Default()`; thread request-scoped fields via context
  or a child logger. Never log secrets, tokens, PII, or full request bodies.
- Instrument with metrics (RED/USE) and tracing (OpenTelemetry) at service boundaries. Make the
  failure modes observable before they happen in prod.
- Make logs actionable: include the identifiers needed to find the affected entity, not noise.

---

## 14. Security

- Validate and bound all external input (size, length, encoding) before use. Treat anything from the
  network, files, env, or users as hostile.
- Use `crypto/rand` for tokens/keys/secrets, never `math/rand`. Use `subtle.ConstantTimeCompare` for
  secret comparison.
- Parameterize SQL (`db.QueryContext(ctx, "... WHERE id = $1", id)`); never string-concatenate queries.
- Use `html/template` (auto-escaping) for HTML; `text/template` does not escape.
- Don't put secrets in code or logs. Load from env/secret manager; scrub from error messages.
- Set server timeouts and body-size limits (`http.MaxBytesReader`) to resist slow-loris/DoS.
- Keep dependencies patched; run `govulncheck` in CI.

---

## 15. Tooling (wire these into CI)

- `gofmt`/`goimports` — formatting + import grouping. Enforce, don't debate.
- `go vet` — catches real bugs (printf mismatches, copied locks, struct tags).
- `staticcheck` / `golangci-lint` — the standard meta-linter. Start with a sane default set; add rules
  deliberately, don't enable everything and drown in noise.
- `govulncheck` — known-vulnerability scanning against your actual call graph.
- `go test -race -cover ./...` — correctness + race + coverage signal.
- Make these a single `make check` / CI step so "green" means all of them passed.

---

## 16. Common antipatterns to reject in review

- Naked `panic`/`os.Exit` outside `main`/init or a deliberate, documented invariant.
- Swallowed errors (`_ =`, empty `if err != nil {}`, or logging without handling).
- `interface{}`/`any` where a concrete type or generic would do; type-switch sprawl.
- Goroutines with no exit path; `go f()` per request without bounds.
- Mutating a shared slice/map without synchronization.
- Interfaces defined next to their single implementation, returned from constructors.
- `init()` doing real work (I/O, network, ordering-dependent setup). Keep `init` trivial or absent.
- Premature generics. Don't parameterize until at least two concrete types share real logic.
- Stringly-typed APIs and magic constants instead of named types/enums (`type State int` + `iota`).
- Context stored in structs, or `WithValue` used as a parameter-passing shortcut.
- Reinventing std-lib (`strings`, `slices`, `maps`, `cmp`, `errors`) by hand.

---

## 17. Generics — use sparingly and well

- Reach for generics when you have **real, type-safe reuse across multiple concrete types**:
  containers, `slices`/`maps`-style utilities, constraints over numeric types.
- Don't use generics to avoid writing two short functions, or where an interface expresses the
  behavior more clearly. "Can I?" is not "should I?"
- Constrain meaningfully (`constraints.Ordered`, your own small constraint interfaces); avoid `any`
  type parameters that erase the benefit.
- Prefer the std `slices`, `maps`, and `cmp` packages over hand-rolled generic helpers.

---

## 18. Review checklist (paste into PRs)

- [ ] `gofmt`/`goimports`, `go vet`, `golangci-lint`, `go test -race` all green.
- [ ] Every error handled, wrapped with context, inspectable where needed; no swallowed errors.
- [ ] No goroutine without a clear owner and exit; concurrency is bounded; `-race` clean.
- [ ] `context.Context` plumbed for I/O/blocking calls; deadlines/timeouts set on network calls.
- [ ] Interfaces small and defined at the consumer; constructors return concrete types.
- [ ] Exported surface is minimal, documented, and stable; names idiomatic (initialisms, no stutter).
- [ ] Resources closed/cancelled via `defer`; writer `Close` errors checked.
- [ ] Zero values usable; nil maps not written; slice aliasing considered.
- [ ] Tests are table-driven, cover behavior + edge cases, no time/rand flakiness.
- [ ] No secrets in code/logs; external input validated; SQL parameterized.
- [ ] Simplest design that works; abstraction justified by real need, not speculation.

---

## Rationalizations & rebuttals

| Excuse | Rebuttal |
| --- | --- |
| "It's a quick fix, skip the test." | Untested fixes regress silently; a table-driven case is a few lines and locks the behavior in. |
| "I'll handle that error later / `_ =` for now." | Swallowed errors become the outage with no breadcrumb. Handle or wrap it now, or comment why it's safe to drop. |
| "One goroutine won't leak anything." | Every goroutine needs an owner and a guaranteed exit; "just one" per request is how you OOM under load. |
| "An interface here makes it more flexible." | A single-impl interface is abstraction debt. Define it at the consumer when a second impl or test-double actually exists. |
| "`any` is simpler than fighting the types." | `any` erases the compiler's help and pushes failures to runtime; use a concrete type or a constrained generic. |
| "gofmt/lint can wait until the PR." | `gofmt` is not opinion and the gates are cheap; run them before you read the diff yourself, not after review. |
| "I'll reuse this backing slice/buffer to save an alloc." | Without a profile that's a guess, and aliasing a shared slice is a data-corruption bug. Measure first, copy when independence matters. |

## Red flags

Stop and reconsider when you see:
- A `select` or hot loop with `time.After` (leaks a timer until it fires) — use a stoppable `Timer` or context deadline.
- `go f()` spawned per request/item with no semaphore, worker pool, or `errgroup.SetLimit`.
- `context.Context` stored in a struct field, or `WithValue` used to pass ordinary parameters.
- Any network/DB call with no timeout or deadline (zero-value `http.Client` hangs forever).
- An error string built with `+`/`fmt.Sprintf` and inspected via `strings.Contains` instead of `errors.Is`/`errors.As`.
- `defer` inside a loop expecting per-iteration release, or `resp.Body`/writer `Close` whose error is dropped.
- A `panic`, `log.Fatal`, or `os.Exit` reachable from a reusable library (not `main`/init).
- Naming smells: `userId`/`apiUrl` initialisms, `UserService` stutter, `util`/`common`/`helpers` packages.

## Verification gate (definition of done)

Go work is not done until all of the following pass and you can show the evidence:
- [ ] `gofmt -l .` (or `goimports -l .`) prints nothing — source matches formatter output, imports grouped.
- [ ] `go build ./...` compiles with no errors.
- [ ] `go vet ./...` is clean (printf mismatches, copied locks, struct tags).
- [ ] `golangci-lint run` (or `staticcheck ./...`) passes with the agreed rule set.
- [ ] `go test -race ./...` is green — tests pass and the race detector is clean.
- [ ] `govulncheck ./...` reports no actionable vulnerabilities against the call graph.
- [ ] `go mod tidy` leaves `go.mod`/`go.sum` unchanged (deps are accurate and committed).
- [ ] New/changed behavior has table-driven tests; untrusted-input parsers have a fuzz target.
- [ ] Deviations from these defaults are justified in a comment or the PR description.

> Tool/version caveat: `golangci-lint` rule sets and Go release behavior (e.g. the Go 1.22+ loop-variable
> fix, `go vet`/`govulncheck` checks) move with the toolchain — pin versions in CI and verify against the
> current release notes rather than assuming.

---

### Canonical references
- **Google Go Style Guide** — https://google.github.io/styleguide/go/guide *(the bar this doc targets)*
- **Google Go Style Decisions** — https://google.github.io/styleguide/go/decisions
- **Google Go Best Practices** — https://google.github.io/styleguide/go/best-practices
- Effective Go — https://go.dev/doc/effective_go
- Go Code Review Comments — https://go.dev/wiki/CodeReviewComments
- Go Proverbs — https://go-proverbs.github.io
- Standard library source — the best style guide there is.
