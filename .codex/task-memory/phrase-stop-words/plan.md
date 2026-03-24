# Phrase Query Stop Words

- Goal: verify whether phrase queries still match when the inverted index is built with `remove_stop_words=true`, especially for queries like `want the apple` vs `want an apple`.
- Steps:
- Add an end-to-end Rust dataset test that reproduces the requested cases.
- Run the targeted test to confirm current behavior.
- Only change implementation if the test fails.

- Status:
- Completed.
