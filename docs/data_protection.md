# Data protection notes

- Source NACC files and all participant-level or derived data must remain on approved local storage.
- Do not commit source files, raw identifiers, cohort manifests, split membership, visit sequences, or preprocessing outputs.
- Generated outputs replace raw participant/site identifiers with keyed HMAC-SHA256 `subject_key` values. Pseudonymization does not make these data public or non-sensitive.
- The shared composite-key secret must be at least 32 random bytes, stable for an authorized study run, and stored outside the repository with mode `0600`. Never print it, include it in the config, or commit it. The previously tracked Workstream 2 key has been replaced locally; the GitHub history still requires coordinated purge and secret rotation before any push.
- Clinical and MRI missing-value codes are recorded from the NACC UDSv4 and Mixed Protocol MRI Researcher's Data Dictionaries in the public schemas. Do not change these codes based on observed distributions.
- `NACC-DATA/` is excluded by `.gitignore`; verify with `git check-ignore` before any staging operation. Do not stage archives containing NACC exports.
- Review `git status --short`, `git diff --cached --name-only`, and `.gitignore` checks before every commit.
