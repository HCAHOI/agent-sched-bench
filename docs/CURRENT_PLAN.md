# Test Pruning Plan

## Goal

Aggressively prune `tests/` in a one-way pass. Default action is delete.
Keep only tests that protect core pipeline correctness, meaningful edge cases,
or real end-to-end behavior.

## Steps

1. Inventory and classification
   Status: in_progress
   Scope:
   - Enumerate every test file and collected test function.
   - Check for dead imports or tests targeting removed behavior.
   - Mark each test as keep or delete using the mission criteria.

2. File-level pruning
   Status: pending
   Scope:
   - Delete entire files when more than 70 percent of their tests fall in the
     delete bucket.
   - Do not leave a file behind with only one surviving test.

3. Shared test cleanup
   Status: pending
   Scope:
   - Remove dead fixtures or shared test helpers that become unreferenced after
     file deletion.
   - Confirm whether `conftest.py` cleanup is needed.

4. Verification
   Status: pending
   Scope:
   - Run the remaining test suite.
   - Fix only breakage caused by deleted tests or dead test-only fixtures.

5. Independent review
   Status: pending
   Scope:
   - Spawn a strict reviewer sub-agent for the deletion diff before finalizing.

## Notes

- No test rewrites or improvements in this pass.
- No import fixing for dead tests: delete instead.
- Final report must include deleted files count, deleted test count, kept test
  count, and a one-line justification for each kept test.
