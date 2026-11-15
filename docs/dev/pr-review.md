# Pull request review

## Goal

Give the maintainer a merge decision they can act on for the latest PR head. Determine whether the change is necessary, whether it follows upstream design, and whether tests through the real entry point prove its behavior.

## Scope

Review the latest PR head against the current target branch. Read the repository's `AGENTS.md`, the PR description, linked issues, CI results, and all prior review comments, including inline threads. For shared-interface changes, inspect the upstream counterpart's recent history as well as its current state.

## Checks

### 1. Necessity

Reproduce the problem on the target branch. Check whether this repository owns the fix and whether an overlapping PR or upstream component already solves it. If the change is unjustified, recommend closing.

### 2. Upstream consistency

- **Interface-change drift check.** Treat changes to public or shared interfaces, including API signatures, config keys, protocol semantics, module boundaries, and lifecycle, as requirement drift unless the PR's stated motivation is upstream synchronization. For each change, require a linked requirement or issue, or an upstream counterpart identified by file, symbol, or commit. If neither is present, flag the change and request that it be reverted or moved to a separately justified PR. A real bug alone does not justify changing a shared interface or rewriting a model.
- **Upstream interface-sync check.** For each changed feature with an upstream counterpart, inspect the history of the touched file or symbol. If upstream has a newer interface change, cite the upstream commit SHA or link, compare it with the PR, and request timely synchronization or a documented reason to defer. In read-only mode, include the request in the review output. Submit a PR comment only with explicit authorization.
- Cite the upstream counterpart by file, symbol, or commit for each significant design choice.
- Flag deviations in placement, naming, interfaces, protocol semantics, module boundaries, and lifecycle.
- Confirm that callers of shared paths remain compatible, original model behavior is preserved, and future upstream syncs are no harder.

### 3. Correctness

Trace the changed entry points and their callers. Look for regressions, unhandled inputs, and breakage of existing data or state.

### 4. Size

Identify production code that can be removed or replaced by existing implementations, including single-use wrappers, dead code, and duplicated logic.

### 5. Tests and validation

- Check that tests exercise the changed behavior through the real entry point and fail without the fix.
- Flag tests that only assert static structure or wiring, duplicate combinations, oversized GPU matrices, and missing device-count gates.
- Verify that all validation records match the current PR head.
- Classify each CI failure as an implementation problem, an outdated test, or an infrastructure failure.

### 6. Description

Verify that the PR body matches the current diff and includes a reproduction, compatibility notes, or performance data where relevant.

## Re-review

Mark each prior finding **Resolved**, **Partially resolved**, or **Unresolved**, with evidence from the latest head. Review commits added since the last review for new regressions. Reuse earlier verification for code that has not changed.

Check upstream for interface commits added since the last review. Mark prior sync findings **Resolved**, **Partially resolved**, or **Unresolved** with commit evidence.

## Output

Write the review in English, ready to post.

1. **Verdict first.** Use **Ready to merge**, **Changes requested**, or **Recommend closing**, with a one-sentence reason.
2. **Findings.** Order findings by severity. Start each finding with its severity tag, `[Blocker]`, `[Should fix]`, or `[Optional]`, followed by `file:line`. Supply each of the following fields.
   - **Trigger** — the input or condition that causes the problem.
   - **Impact** — what breaks and for whom.
   - **Evidence** — the reproduction, upstream reference, or test output.
   - **Fix** — the concrete change.
3. **Cleanup.** List code to remove and tests that must be kept.
