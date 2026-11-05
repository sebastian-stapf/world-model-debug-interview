# PhD coding interview

Two tasks in Python and PyTorch: implement a few-shot classifier, then investigate
an action-conditioned flow-matching world model. Task 2 has two alternative formats;
choose one for the interview.

| Task | Candidate materials | Suggested time |
| --- | --- | --- |
| **Task 1: Few-shot classification** | [Prompt and starter](task_1/) · [PDF handout](task_1/few-shot-classification.pdf) | 35-45 minutes |
| **Task 2A: Read-only debugging** | [Browse independently; no code execution](task_2/read_only/) | 60-75 minutes |
| **Task 2B: Hands-on debugging** | [Run code, implement fixes and write tests](task_2/hands_on/) | 90 minutes |

```text
task_1/
  README.md                     Few-shot classification prompt
  few-shot-classification.pdf   Candidate handout
  matching.py                   Python starter
  build_pdf.py                  Optional handout authoring script
task_2/
  README.md                     Choose an interview format
  read_only/                    Self-contained browsing pack
    README.md                   Independent investigation instructions
    RESPONSE_TEMPLATE.md        Optional written submission
    source/                     Source listings as plain text
    evidence/                   Recorded figures, metrics and raw-data excerpt
  hands_on/                     Runnable debugging project
    README.md                   Setup, investigation and testing instructions
    world_model/                Dataset, dataloader, model, trainer, evaluation
    data/                       Bundled synthetic Pong episodes
    assets/                     Measured diagnostic evidence
    scripts/                    Collection, diagnosis and UBELIX submission
    tests/                      Smoke and infrastructure tests
```

Both Task 2 formats contain the **same five intentional bugs** and use the same
recorded UBELIX experiment. The read-only pack requires no installation, terminal,
checkpoint or code execution. The hands-on version retains the original runnable
exercise; use `task_2/hands_on/` as its working directory.

The Task 1 starter is deliberately unimplemented. Task 2's public tests check its
infrastructure and are not a correctness test for the five planted bugs.

## Task 2 evidence preview

The model's regression loss falls while its generated trajectories remain poor.
These are real outputs from the supplied candidate code, trained on UBELIX.
Each version includes the complete evidence and its interpretation conventions.

![Recorded loss and evaluation curves](task_2/hands_on/assets/diagnostics.png)

![Recorded and generated trajectories](task_2/hands_on/assets/generations.png)

The [trained candidate checkpoint](https://github.com/sebastian-stapf/world-model-debug-interview/releases/tag/v1.0-interview)
is available for Task 2B. The [Task 1 PDF release](https://github.com/sebastian-stapf/world-model-debug-interview/releases/tag/v1.1-interview)
remains available as a direct download.
