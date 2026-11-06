# Maintaining the interview materials

From the repository root, run `python task_2/build_read_only.py` with NumPy installed.
It refreshes the read-only Python sources, specification and evidence from the
hands-on folder, checks source hashes against the recorded run, and rebuilds
`outputs/task-2-read-only.zip` and the pack's `MANIFEST.json`.

The read-only README and response template are maintained separately. Rebuild the
pack after editing them so the manifest and ZIP stay current. Keep its pipeline
diagram aligned with the hands-on README. Refresh the published ZIP when releasing
pack changes. These are preparation steps, not part of the candidate's exercise.

To rebuild the Task 1 PDF, install ReportLab and run `python task_1/build_pdf.py`.
