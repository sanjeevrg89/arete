# TDD Worked Examples — AI infra / ML

Imitate these. Each is self-contained and correct in spirit (imports/boilerplate elided). They show
the **red → green → refactor** loop on real infra/ML tasks, an **eval-as-test** gate, and a controller
**envtest** sketch.

---

## 1. Red → green → refactor on an infra task

**Task (one acceptance criterion from `[[task-planning-decomposition]]`):** given a model with `N`
layers and `P` pipeline stages, compute the layers-per-stage assignment. Contract: every layer
assigned exactly once; stages differ in size by at most 1 (remainder spread over the first stages);
`P ≤ N` and `P ≥ 1` else error.

### RED — write the failing test first

```go
// pipeline_split_test.go  (run it now: it fails to COMPILE because SplitLayers doesn't exist — good)
func TestSplitLayers(t *testing.T) {
    t.Parallel()
    tests := []struct {
        name        string
        layers, stages int
        want        []int // count per stage
        wantErr     bool
    }{
        {name: "even",        layers: 8, stages: 4, want: []int{2, 2, 2, 2}},
        {name: "remainder",   layers: 10, stages: 4, want: []int{3, 3, 2, 2}}, // remainder on front stages
        {name: "single stage", layers: 5, stages: 1, want: []int{5}},
        {name: "stage per layer", layers: 3, stages: 3, want: []int{1, 1, 1}},
        {name: "too many stages", layers: 2, stages: 3, wantErr: true},
        {name: "zero stages",  layers: 4, stages: 0, wantErr: true},
    }
    for _, tc := range tests {
        t.Run(tc.name, func(t *testing.T) {
            t.Parallel()
            got, err := SplitLayers(tc.layers, tc.stages)
            if (err != nil) != tc.wantErr {
                t.Fatalf("SplitLayers(%d,%d) err = %v, wantErr %v", tc.layers, tc.stages, err, tc.wantErr)
            }
            if tc.wantErr {
                return
            }
            if diff := cmp.Diff(tc.want, got); diff != "" {
                t.Errorf("SplitLayers(%d,%d) mismatch (-want +got):\n%s", tc.layers, tc.stages, diff)
            }
        })
    }
}
```

Run `go test -race ./...` → **RED** (won't compile / fails). Confirm it fails for the right reason.

### GREEN — least code that passes

```go
// SplitLayers returns the number of layers assigned to each of `stages` pipeline stages.
// The first (layers % stages) stages get one extra layer.
func SplitLayers(layers, stages int) ([]int, error) {
    if stages < 1 || stages > layers {
        return nil, fmt.Errorf("invalid split: %d layers over %d stages", layers, stages)
    }
    base, extra := layers/stages, layers%stages
    out := make([]int, stages)
    for i := range out {
        out[i] = base
        if i < extra {
            out[i]++
        }
    }
    return out, nil
}
```

Run `go test -race ./...` → **GREEN**.

### REFACTOR — improve on green

The invariant "counts sum to `layers`" is the real contract; assert it once as a helper so future
cases inherit it, and add it to the table loop. No behavior change, tests stay green.

```go
func sumEquals(counts []int, want int) error {
    s := 0
    for _, c := range counts {
        s += c
    }
    if s != want {
        return fmt.Errorf("counts sum to %d, want %d", s, want)
    }
    return nil
}
// in the test, after the diff check:
if err := sumEquals(got, tc.layers); err != nil {
    t.Errorf("SplitLayers(%d,%d): %v", tc.layers, tc.stages, err)
}
```

Run → still GREEN. **Commit** ("pipeline: split layers across stages, remainder on front stages").
Loop to the next criterion.

---

## 2. Eval-as-test — threshold-gated, deterministic seed

**Task:** a RAG answerer changed (new retrieval). You cannot assert exact tokens. Gate the change with
**contracts** (deterministic checks on the output) + a **metric threshold** on a fixed, versioned eval
set, fully seeded so the gate is reproducible. See `[[ml-evaluation-evals]]` for metric choice.

```python
# test_rag_eval.py — runs in CI as the eval gate
import json, pytest

EVAL_SET = "evals/rag_qa_v3.jsonl"   # versioned, pinned; do not edit casually
SEED = 1234
THRESHOLDS = {"faithfulness": 0.90, "answer_relevance": 0.85}
BASELINE = json.load(open("evals/baseline_v3.json"))  # last accepted scores
MAX_REGRESSION = 0.01

@pytest.fixture(scope="module")
def answers():
    cases = [json.loads(l) for l in open(EVAL_SET)]
    # Determinism: pinned model+version, temperature 0, fixed seed.
    client = make_answerer(model="my-rag@2026-05-01", temperature=0.0, seed=SEED)
    return [(c, client.answer(c["question"])) for c in cases]

# --- Contracts: DETERMINISTIC checks on a stochastic output (these are unit-grade) ---
@pytest.mark.parametrize("strict", [True])
def test_output_contracts(answers, strict):
    for case, out in answers:
        assert out.text.strip(), f"empty answer for {case['id']}"
        # grounding contract: every cited doc id must be in the retrieved set
        assert set(out.cited_doc_ids) <= set(out.retrieved_doc_ids), f"hallucinated citation in {case['id']}"
        assert len(out.text) <= 1200, f"answer too long for {case['id']}"
        assert not contains_pii(out.text), f"PII leak in {case['id']}"

# --- Property (metamorphic): an irrelevant trailing sentence must not change the label ---
def test_irrelevant_context_invariance():
    base = answer_label("Is the sky blue?")
    perturbed = answer_label("Is the sky blue? By the way, I had coffee.")
    assert base == perturbed

# --- Metric threshold + no-regression gate (this is the quality regression test) ---
def test_metric_thresholds(answers):
    scores = score_faithfulness_and_relevance(answers)  # judge with bias mitigations, seeded
    for metric, floor in THRESHOLDS.items():
        assert scores[metric] >= floor, f"{metric}={scores[metric]:.3f} below floor {floor}"
        drop = BASELINE[metric] - scores[metric]
        assert drop <= MAX_REGRESSION, f"{metric} regressed {drop:.3f} vs baseline (max {MAX_REGRESSION})"
```

Why this is a real test, not vibes: the output **contracts** are deterministic and fail hard on
hallucinated citations / PII / empty answers; the **metric** gate fails the build if quality drops
below the floor or regresses past the band; everything is **seeded and pinned** so the gate is
reproducible. A flaky gate here means the seed/version isn't pinned tightly enough — fix that, don't
loosen the threshold blindly.

---

## 3. Controller test with `envtest` (sketch)

**Task:** a reconciler that, for each `TrainingJob` CR, creates a child `Job` and sets a `Ready`
condition. Test that the cluster **converges** and that reconcile is **idempotent** — against a real
apiserver + etcd, no kubelet. See `[[kubernetes-controller-expert]]` for the reconcile properties.

```go
var _ = Describe("TrainingJob controller", func() {
    ctx := context.Background()

    It("creates the child Job and converges (idempotently)", func() {
        tj := &trainv1.TrainingJob{
            ObjectMeta: metav1.ObjectMeta{Name: "demo", Namespace: "default"},
            Spec:       trainv1.TrainingJobSpec{Replicas: 2, Image: "trainer:latest"},
        }
        Expect(k8sClient.Create(ctx, tj)).To(Succeed())

        // Behavior, not implementation: assert the child Job EXISTS (poll, never sleep).
        childKey := types.NamespacedName{Name: "demo-worker", Namespace: "default"}
        Eventually(func() error {
            return k8sClient.Get(ctx, childKey, &batchv1.Job{})
        }).WithTimeout(10 * time.Second).Should(Succeed())

        // Status condition flips to Ready.
        Eventually(func() bool {
            got := &trainv1.TrainingJob{}
            if err := k8sClient.Get(ctx, client.ObjectKeyFromObject(tj), got); err != nil {
                return false
            }
            return meta.IsStatusConditionTrue(got.Status.Conditions, "Ready")
        }).WithTimeout(10 * time.Second).Should(BeTrue())

        // Idempotency: a fresh reconcile must NOT create a second Job / must be a no-op.
        var jobs batchv1.JobList
        Expect(k8sClient.List(ctx, &jobs, client.InNamespace("default"))).To(Succeed())
        Expect(jobs.Items).To(HaveLen(1))
    })
})
```

`envtest` provisions the apiserver + etcd in-process (via `setup-envtest`), so this runs at
unit-test speed in CI. Note the discipline: assert **observable cluster state** with `Eventually`
(level-triggered, eventually-consistent), never a `time.Sleep`, and explicitly assert **idempotency**
— the single most common reconciler bug is creating duplicates on re-reconcile.
