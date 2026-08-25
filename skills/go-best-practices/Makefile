# Makefile — Go quality gate (starter)
# Drop into a Go module root. `make check` is the definition-of-done gate from go-guidelines.md.
#
# Tool versions are pinned via go.mod tooling where possible; `make tools` installs the rest.
# Override on the CLI, e.g.  make test PKG=./internal/...  or  make GOFLAGS=-count=1 test

GO         ?= go
PKG        ?= ./...
GOFLAGS    ?=
GOTESTFLAGS?= -race -shuffle=on
COVERPROFILE ?= coverage.out

# Tool binaries (installed into $(go env GOBIN) or GOPATH/bin)
GOLANGCI_LINT ?= golangci-lint
GOVULNCHECK   ?= govulncheck
GOIMPORTS     ?= goimports

.DEFAULT_GOAL := check

## help: list targets
.PHONY: help
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## //' | awk -F: '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## check: full local gate — fmt-check, vet, lint, race tests, vuln. Run before pushing.
.PHONY: check
check: fmt-check vet lint test-race vuln
	@echo "✓ all checks passed"

## ci: the gate for CI (same as check, fails on any unformatted file)
.PHONY: ci
ci: check

## fmt: format code in place (gofmt + goimports)
.PHONY: fmt
fmt:
	$(GO) fmt $(PKG)
	@command -v $(GOIMPORTS) >/dev/null 2>&1 && $(GOIMPORTS) -w -local "$$(head -1 go.mod | awk '{print $$2}')" . || echo "note: goimports not installed (run 'make tools')"

## fmt-check: fail if any file is not gofmt-clean (no writes)
.PHONY: fmt-check
fmt-check:
	@unformatted=$$(gofmt -l .); \
	if [ -n "$$unformatted" ]; then \
		echo "✗ not gofmt-clean:"; echo "$$unformatted"; exit 1; \
	fi
	@echo "✓ gofmt clean"

## vet: go vet
.PHONY: vet
vet:
	$(GO) vet $(GOFLAGS) $(PKG)

## lint: golangci-lint (config in .golangci.yml)
.PHONY: lint
lint:
	@command -v $(GOLANGCI_LINT) >/dev/null 2>&1 || { echo "✗ golangci-lint not installed (run 'make tools')"; exit 1; }
	$(GOLANGCI_LINT) run

## lint-fix: golangci-lint with autofixes applied
.PHONY: lint-fix
lint-fix:
	$(GOLANGCI_LINT) run --fix

## test: unit tests
.PHONY: test
test:
	$(GO) test $(GOFLAGS) $(GOTESTFLAGS) $(PKG)

## test-race: tests with the race detector (mandatory per guidelines)
.PHONY: test-race
test-race:
	$(GO) test $(GOFLAGS) -race $(PKG)

## cover: tests with coverage profile + summary
.PHONY: cover
cover:
	$(GO) test $(GOFLAGS) -race -covermode=atomic -coverprofile=$(COVERPROFILE) $(PKG)
	$(GO) tool cover -func=$(COVERPROFILE) | tail -1

## cover-html: open the coverage report in a browser
.PHONY: cover-html
cover-html: cover
	$(GO) tool cover -html=$(COVERPROFILE)

## bench: run benchmarks with allocation stats
.PHONY: bench
bench:
	$(GO) test $(GOFLAGS) -run '^$$' -bench=. -benchmem $(PKG)

## vuln: govulncheck against the actual call graph
.PHONY: vuln
vuln:
	@command -v $(GOVULNCHECK) >/dev/null 2>&1 || { echo "✗ govulncheck not installed (run 'make tools')"; exit 1; }
	$(GOVULNCHECK) $(PKG)

## build: compile everything (no output binary)
.PHONY: build
build:
	$(GO) build $(GOFLAGS) $(PKG)

## tidy: go mod tidy + verify
.PHONY: tidy
tidy:
	$(GO) mod tidy
	$(GO) mod verify

## tools: install the dev tooling used by this Makefile
.PHONY: tools
tools:
	$(GO) install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest
	$(GO) install golang.org/x/tools/cmd/goimports@latest
	$(GO) install golang.org/x/vuln/cmd/govulncheck@latest

## clean: remove build/test artifacts
.PHONY: clean
clean:
	$(GO) clean
	rm -f $(COVERPROFILE)
