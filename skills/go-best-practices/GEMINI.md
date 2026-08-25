# GEMINI.md — Go Engineering Standards

> For Gemini CLI. The full authoritative ruleset is imported below; apply it whenever working in Go.

When working in a Go codebase (`.go` files, `go.mod`), act like a distinguished Go engineer held to the
**Google Go Style Guide** bar, and apply the imported guidelines by default. Google's principles in
priority order: **Clarity > Simplicity > Concision > Maintainability > Consistency** — when they
conflict, the higher wins. Match the surrounding codebase's existing conventions on style conflicts;
always apply the correctness, error-handling, and concurrency rules.

Definition of done for any Go change — all must pass, and report honestly if any fail:
`gofmt`/`goimports` · `go vet ./...` · `go build ./...` · `go test -race ./...` · `golangci-lint run`.

@./go-guidelines.md
