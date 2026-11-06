# Reproduce the supplied run (optional)

Run these commands from `task_2/hands_on/` after completing the README setup.
The interview does not require retraining or downloading a checkpoint.

For the package versions used in the supplied run:

```bash
python -m pip install -r requirements-tested.txt -e '.[dev]'
```

## Checkpoint and diagnostics

Download the supplied checkpoint and run records from the
[public release](https://github.com/sebastian-stapf/world-model-debug-interview/releases/tag/v1.0-interview),
then reproduce the observation pack:

```bash
mkdir -p outputs
gh release download v1.0-interview --pattern candidate-run.zip --dir outputs
python -m zipfile -e outputs/candidate-run.zip outputs
python -m scripts.diagnose --checkpoint outputs/candidate/model.pt --output outputs/diagnostics
python -m world_model.eval --checkpoint outputs/candidate/model.pt
```

The release ZIP extracts to `outputs/candidate/`. Saved figures already allow
offline inspection without the checkpoint. Diagnostic generation takes longer
on CPU than the smoke tests; it is optional during the interview.

To regenerate the data or train after making changes:

```bash
python -m scripts.make_data
python -m scripts.make_data --output data/heldout_pong.npz --episodes 32 --seed 90000
python -m world_model.trainer --device cpu --steps 20 --output outputs/smoke
# Full run on a GPU:
python -m world_model.trainer --device cuda --output outputs/run
python -m scripts.diagnose --device cuda --checkpoint outputs/run/model.pt --output outputs/diagnostics
```

Use the CPU command only for a small pipeline check; the published experiment
uses the full configuration in [`configs/default.json`](configs/default.json).

## UBELIX

Create a Python environment on UBELIX, install this project's dependencies, and
submit from the `task_2/hands_on/` directory. Adjust the account and GPU type for your own
allocation. Create the log directory before submission:

```bash
mkdir -p outputs
export WM_PYTHON=/absolute/path/to/your/environment/bin/python
sbatch scripts/ubelix.sbatch
```

The script requests one RTX 4090 and runs data collection, training and diagnosis
on the compute node. See the
[UBELIX GPU guide](https://knowit.unibe.ch/en/docs/hpc/runjobs/scheduled-jobs/gpus/)
for current resource syntax. The experiment's actual environment is recorded with
the release; small numerical differences across devices are expected.
