# Preserved build-source verification

- Repository revision: `2a14eb656bd983c37b60e92852083524348511d8`
- Preserved theory-bridge PAR SHA-256: `77dabda68715e206ba15f676fd4360ef061ab815eafbeb12de0d1bcc849ad177`
- Preserved full PAR SHA-256: `2a4d3ca7bf44262cb7629526d983031432c9134fa438109a649314339d4ee700`
- Theory-bridge inventory: 31,304 archive entries; 115 commit-tracked project entries; 0 byte mismatches; 0 duplicate project entry names.
- Full inventory: 31,294 archive entries; 105 commit-tracked project entries; 0 byte mismatches; 0 duplicate project entry names.
- `policy_improvement_full_backend.py` is byte-identical in both preserved PARs and revision `2a14eb6`.
- The executed Experiment 1B parity checks at lines 6100-6127 raise `Exp1bSealedEvaluationError` for mixture-identity, constructed-centering, trainer-parity, and trainer-centering-defect failures. They do not print and continue.

The complete file-by-file inventory and SHA-256 comparison is in `embedded_source_inventory.json`.
