# Go Patterns — Before / After

Concrete snippets for the highest-impact rules in `go-guidelines.md`. Imitate the **GOOD** column.
Each example is self-contained and compiles in spirit (imports elided for brevity).

---

## 1. Error wrapping

**Rules:** wrap with `%w` at the end of the message; lowercase, no "failed to", no trailing
punctuation; don't log *and* return; give callers something to inspect with `errors.Is`/`errors.As`.

```go
// ❌ BEFORE
func loadConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, errors.New("Failed to read config file.") // caps, "failed to", punctuation, loses cause
    }
    var c Config
    if err := json.Unmarshal(data, &c); err != nil {
        log.Printf("error parsing config: %v", err) // logs AND returns -> double report
        return nil, err                             // bare: caller can't tell where it came from
    }
    return &c, nil
}
```

```go
// ✅ AFTER
// ErrEmptyConfig is returned when the config file exists but has no content.
var ErrEmptyConfig = errors.New("config is empty")

func loadConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("read config %q: %w", path, err) // %w at end, lowercase, keeps cause
    }
    if len(data) == 0 {
        return nil, ErrEmptyConfig // sentinel the caller can match
    }
    var c Config
    if err := json.Unmarshal(data, &c); err != nil {
        return nil, fmt.Errorf("parse config %q: %w", path, err) // wrap, don't log — caller owns logging
    }
    return &c, nil
}
```

```go
// Caller inspects the structured error instead of matching strings:
cfg, err := loadConfig(path)
switch {
case errors.Is(err, ErrEmptyConfig):
    cfg = defaultConfig() // recover from a known condition
case err != nil:
    var perr *fs.PathError
    if errors.As(err, &perr) {
        return fmt.Errorf("config unreadable at %s: %w", perr.Path, err)
    }
    return err
}
```

---

## 2. Receiver consistency (and not copying a lock)

**Rules:** pointer receiver when the method mutates, the struct is large, or it holds a no-copy field
(`sync.Mutex`); make a type's methods **all** pointer or **all** value — never mixed.

```go
// ❌ BEFORE
type Counter struct {
    mu sync.Mutex
    n  int
}

func (c Counter) Inc()       { c.mu.Lock(); c.n++; c.mu.Unlock() } // value rx: COPIES the mutex, mutation is lost
func (c *Counter) Value() int { return c.n }                       // mixed pointer/value receivers
```

```go
// ✅ AFTER
type Counter struct {
    mu sync.Mutex
    n  int
}

// All pointer receivers, consistent short name `c`, lock never copied.
func (c *Counter) Inc() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.n++
}

func (c *Counter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.n
}
```

> `go vet` flags the copied lock; the mixed-receiver bug it won't catch — reviewers must.

---

## 3. Table-driven tests with `go-cmp`

**Rules:** named subtests via `t.Run`; specify struct field names in cases; compare with
`cmp.Diff` (readable diffs); `t.Parallel()` for independent tests; never `t.Fatal` from a goroutine.

```go
// ✅ table-driven + cmp.Diff
func TestSplitHostPort(t *testing.T) {
    t.Parallel()

    tests := []struct {
        name    string
        in      string
        want    HostPort
        wantErr bool
    }{
        {name: "host and port", in: "example.com:8080", want: HostPort{Host: "example.com", Port: 8080}},
        {name: "ipv6",          in: "[::1]:80",         want: HostPort{Host: "::1", Port: 80}},
        {name: "missing port",  in: "example.com",      wantErr: true},
    }

    for _, tc := range tests {
        t.Run(tc.name, func(t *testing.T) {
            t.Parallel()

            got, err := SplitHostPort(tc.in)
            if (err != nil) != tc.wantErr {
                t.Fatalf("SplitHostPort(%q) error = %v, wantErr = %v", tc.in, err, tc.wantErr)
            }
            if tc.wantErr {
                return
            }
            if diff := cmp.Diff(tc.want, got); diff != "" {
                t.Errorf("SplitHostPort(%q) mismatch (-want +got):\n%s", tc.in, diff)
            }
        })
    }
}
```

**Shared validation returns `error` — it does not take `*testing.T`** (Google's rule): keeps failure
reporting in the test.

```go
// ✅ helper returns error; the test decides how to report it
func validUser(u User) error {
    if u.ID == "" {
        return errors.New("empty ID")
    }
    if !strings.Contains(u.Email, "@") {
        return fmt.Errorf("invalid email %q", u.Email)
    }
    return nil
}

// in a test:
if err := validUser(got); err != nil {
    t.Errorf("validUser(%+v): %v", got, err)
}
```

For `cmp` on unexported fields, pass an option explicitly rather than reaching into internals:
```go
cmp.Diff(want, got, cmpopts.IgnoreUnexported(HostPort{}))
```

---

## 4. Options pattern

Pick by caller profile. **Variadic functional options** when *most callers need none*:

```go
// ✅ functional options
type Server struct {
    addr    string
    timeout time.Duration
    tls     *tls.Config
    logger  *slog.Logger
}

// Option configures a Server. Zero options yields working defaults.
type Option func(*Server)

func WithTimeout(d time.Duration) Option { return func(s *Server) { s.timeout = d } }
func WithTLS(c *tls.Config) Option       { return func(s *Server) { s.tls = c } }
func WithLogger(l *slog.Logger) Option   { return func(s *Server) { s.logger = l } }

func NewServer(addr string, opts ...Option) *Server {
    s := &Server{
        addr:    addr,
        timeout: 30 * time.Second, // sensible defaults live in the constructor
        logger:  slog.Default(),
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// NewServer(":8080")                                   // common case: no options
// NewServer(":8080", WithTimeout(5*time.Second), WithTLS(cfg))
```

**Option struct** (last param) when *most callers set one or more* — self-documenting, defaults omittable:

```go
// ✅ option struct
type DialConfig struct {
    Timeout   time.Duration // zero => no timeout
    KeepAlive time.Duration
    TLS       *tls.Config
}

func Dial(addr string, cfg DialConfig) (*Conn, error) {
    if cfg.Timeout == 0 {
        cfg.Timeout = 30 * time.Second
    }
    // ...
}

// Dial("db:5432", DialConfig{Timeout: 5 * time.Second, KeepAlive: time.Minute})
```

> Avoid the anti-pattern of a long positional list (`New(addr, timeout, tls, logger, retries, ...)`):
> call sites become unreadable and every new knob breaks the signature.

---

## 5. Goroutine lifetime & context (bonus — highest bug impact)

**Rules:** every goroutine has an owner and a guaranteed exit; honor `ctx.Done()`; bound concurrency.

```go
// ❌ BEFORE: leaks forever, ignores cancellation, unbounded
func (w *Worker) Start() {
    go func() {
        for {
            job := <-w.in
            w.process(job) // never returns; no way to stop; errors swallowed
        }
    }()
}
```

```go
// ✅ AFTER: caller owns lifetime, stops on context, surfaces errors
func (w *Worker) Run(ctx context.Context) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case job := <-w.in:
            if err := w.process(ctx, job); err != nil {
                return fmt.Errorf("process job %d: %w", job.ID, err)
            }
        }
    }
}
```

**Bounded fan-out** with `errgroup` — any error cancels siblings, concurrency is capped:

```go
// ✅ bounded parallel work
func fetchAll(ctx context.Context, urls []string) ([]Result, error) {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(8) // cap concurrency — never unbounded `go` per item

    results := make([]Result, len(urls))
    for i, url := range urls {
        i, url := i, url // safe on any toolchain
        g.Go(func() error {
            r, err := fetch(ctx, url)
            if err != nil {
                return fmt.Errorf("fetch %s: %w", url, err)
            }
            results[i] = r // distinct index per goroutine => no shared-write race
            return nil
        })
    }
    if err := g.Wait(); err != nil {
        return nil, err
    }
    return results, nil
}
```

---

## 6. Naming quick-reference

| ❌ Before | ✅ After | Why |
|-----------|---------|-----|
| `GetUserName()` | `UserName()` | getters omit `Get` |
| `userId`, `apiUrl`, `httpServer` | `userID`, `apiURL`, `httpServer` → `HTTPServer` (exported) | initialisms keep one case |
| `widget.NewWidget()` | `widget.New()` | no package-name stutter |
| `user.UserService` | `user.Service` | no stutter |
| `const MAX_RETRIES = 3` | `const MaxRetries = 3` | MixedCaps, not SCREAMING_SNAKE |
| `var userSlice []User` | `var users []User` | omit type words |
| `func (this *Tray) ...` / `func (tray *Tray)` | `func (t *Tray) ...` | short, consistent receiver |
| `func numUsers() int` | `func userCount() int` | read as a noun phrase |
