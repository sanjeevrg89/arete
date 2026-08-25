# Third-Party Notices

Arete vendors third-party skills verbatim under `skills/vendored/<upstream>/`. Each keeps its
upstream license; this file records provenance so updates can be synced deliberately.

## mattpocock/skills

- **Source:** https://github.com/mattpocock/skills
- **Upstream commit:** `6654f6b60cd9d5be8b54c6fafe44346dabeb3b76` (main)
- **Location here:** `skills/vendored/mattpocock/<skill-name>/` — 25 skills (18 engineering, 7 productivity), copied verbatim
- **License:** MIT (reproduced below)

These skills cover the *process* layer of engineering with an agent — grilling/spec flows,
TDD loop, code review, bug diagnosis — and complement arete's first-party *domain-depth*
skills (Kubernetes, GKE, ML infrastructure). Where a first-party skill covers similar
ground for AI-infra work specifically (e.g. `test-driven-development`, `code-review-discipline`,
`verification-and-debugging`), both descriptions are scoped so a model routes to the right one.

### Updating

```bash
git -C /tmp/mp-skills pull        # or re-clone github.com/mattpocock/skills
rsync -a --delete /tmp/mp-skills/skills/engineering/  /tmp/stage/
# copy each <name>/ dir into skills/vendored/mattpocock/<name>/, then:
#   - update the upstream commit hash in this file
#   - python scripts/validate.py   (vendored skills are validated loosely: frontmatter + unique name)
```

---

MIT License

Copyright (c) 2026 Matt Pocock

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
