# Download and verify

Use the assets of release **v1.0.0**. The source-code ZIP generated automatically
by GitHub is a Git-tree snapshot; it does not contain every large complex field.

1. Download Reproducibility_Code_Tables_614564_v1.0.0.zip.
2. Download all Reproducibility_Raw_Fields_614564_v1.0.0_part*.zip files.
3. Check each archive against SHA256SUMS_ASSETS.txt (macOS: `shasum -a 256`).
4. Extract every ZIP into the same new folder. The internal paths are relative
   to the package root. Each ZIP is independently readable.
5. From that folder run `python3 code/verify_release.py` to check every retained
   file against DISTRIBUTION_MANIFEST.json. This verifies file integrity, not
   scientific accuracy. The separate REPRODUCE.md describes numerical checks.

If using a Git clone, extract only the raw-field ZIPs into the clone root; the
other files are already present. Use a separate working copy for commands that
write results so the archived evidence remains unchanged.
