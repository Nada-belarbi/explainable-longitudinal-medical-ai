"""Retired: NACCID-only rekeying cannot align composite identities.

Use nacc_progression.identity.subject_key(site, participant, new_private_secret)
in both workstreams. Regenerate the approved partitions using the new recipe;
never randomly reassign participants during key rotation.
"""
if __name__ == '__main__':
    raise SystemExit('Retired NACCID-only rekey tool. Use the shared composite '
                     'identity boundary in python -m nacc_progression.run_final.')
