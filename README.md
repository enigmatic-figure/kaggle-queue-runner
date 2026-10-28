# Kaggle Queue Runner

Submit one stable notebook. After Kaggle allocates hardware, it fetches the current workload list from a public GitHub repository you control. Checks and useful work run on that same allocation. You can add jobs while the notebook waits, or while another job is running.

The queued notebook contains only the repository URL, branch and runtime limits. Workload code is fetched after allocation. Each job gets its own detached Git snapshot, and its receipt records the exact commit, script hash, arguments and outcome. Editing GitHub does not change a running job.

## Start

1. Use the public repository at https://github.com/enigmatic-figure/kaggle-queue-runner, or fork it and change the bootstrap repository URL.
2. Edit `workloads.json` and the scripts under `workloads/`. Use a fresh job ID when changing an executed job.
3. Import `kaggle/queue_runner.ipynb` into Kaggle, enable internet and select the desired accelerator. The supplied metadata targets a private v5e-8 notebook.
4. Attach optional Kaggle Secrets described below, then submit the notebook once. Afterward, update GitHub to change its pending work; leave the queued notebook version alone.

The existing standalone preflight notebook cannot gain this behavior without a new version. This package does not cancel or replace it automatically. Replacing a queued notebook may restart its wait.

The default workload is an 8,192-member real-coordinate geometry trial. It checks and optimizes within one job; it is not the full 29.55-million-member optimization. More workloads can be committed while the worker waits. `workloads/example_cpu.py` is a small disabled template.

## Workload list

```json
{
  "version": 1,
  "artifact_sink": null,
  "jobs": [
    {"id": "prepare-v1", "script": "workloads/prepare.py", "timeout_seconds": 1800},
    {"id": "train-v1", "script": "workloads/train.py", "args": ["--epochs", "4"],
     "depends_on": ["prepare-v1"], "timeout_seconds": 14400,
     "min_remaining_seconds": 3600, "secrets": ["HF_TOKEN"]}
  ]
}
```

Primary jobs run sequentially. An optional resident inference service can share explicitly reserved resources with compatible jobs. The branch is fetched again before selecting each job, and every 30 seconds while idle. Completed IDs are skipped within that run. Failed or interrupted jobs are not silently retried; give a deliberate retry a new ID. `on_failure` is `continue` or `stop`. `enabled: false` leaves a job disabled.

Receipts belong to a run. This version does not implement distributed job leases or exactly-once execution across simultaneous workers. Before submitting a new allocation, remove/disable old jobs or deliberately restore the run directory with `--resume`. An interrupted receipt is retained rather than assumed complete. Checkpoint restoration is workload-specific.

Default limits are eight hours after allocation and five minutes of idle grace for newly added work. These are configurable in the bootstrap. They do not extend Kaggle's session limits or preserve an allocation after Kaggle ends it. The worker exits when its queue remains idle; it does not manufacture activity to hold hardware.

## Workload interface and checkpoints

The base runner uses Python's standard library. The notebook creates an isolated engine environment with the optional storage dependency, leaving the Kaggle model/JAX environment intact. Workloads execute using the original Kaggle Python (`QUEUE_WORKLOAD_PYTHON`). Each workload owns its dependencies and can create its own environment when versions conflict. The included geometry example installs its explicit Hugging Face and Faiss dependencies. It uses the runtime's existing TPU JAX installation.

Every workload receives:

- `QUEUE_OUTPUT_DIR`: its output directory.
- `QUEUE_SHARED_DIR`: reusable data shared by jobs in this allocation.
- `QUEUE_JOB_ID` and `QUEUE_SOURCE_COMMIT`: provenance.
- `QUEUE_CHECKPOINT_REQUEST`: cooperative request flag for the workload to poll.
- Only the named Kaggle Secrets listed in that job's `secrets` field.

```python
from queue_runner.sdk import output_dir, progress, checkpoint, checkpoint_requested

progress(stage="training", epoch=1)
closed = output_dir() / "epoch-001"
closed.mkdir()
# Write and CLOSE the checkpoint files here.
checkpoint("epoch-001")
# Poll checkpoint_requested() at safe boundaries for operator-requested saves.
```

A sealed checkpoint must never be modified. Use a new directory for the next checkpoint. The ready marker contains each file's SHA256. Cooperative checkpoint requests do not make an arbitrary training script checkpoint automatically.

Kaggle retains notebook run outputs and logs. For publication during execution, configure an existing Hugging Face bucket and attach `HF_TOKEN`:

```json
"artifact_sink": {"bucket": "IntellAgents/wiki727-suffix-atlas-20261025", "prefix": "queue-runs"}
```

Install the `hub` extra in the runner environment for this option. The publisher uploads only explicitly sealed checkpoint directories and selected redacted telemetry. It downloads and hashes each new checkpoint payload before publishing its ready marker. Networking credentials and Tailscale state live outside the Kaggle output directory. `artifact-sync.json` records publication errors; a finished job alone does not prove its cloud checkpoint was published.

## Optional private Tailscale connection

For CPU analytics alongside a four-chip vLLM TPU service, see [Mixed workloads](docs/mixed-workloads.md). The package includes an isolated, hash-locked model environment, a pinned Qwen3-8B profile, and DuckDB/Arrow/CMA-ES CPU tooling. Inference is disabled by default. Jobs without explicit resource declarations keep exclusive access to all eight chips, including the current geometry trial. The stable Kaggle notebook is unchanged.

The Linux amd64 worker can run a pinned Tailscale binary in userspace mode, with in-memory node state and a private daemon socket. It needs no TUN device. The binary archive is SHA256-verified. The auth key is passed through a temporary restricted file, then deleted. Use an ephemeral worker auth key with the access your tailnet permits.

Attach these Kaggle Secrets to the stable notebook once:

| Label | Purpose |
|---|---|
| `TS_AUTHKEY` | Ephemeral worker enrollment key. |
| `QUEUE_CONTROL_TOKEN` | Shared bearer token for the worker's control API. |
| `QUEUE_NOTIFY_URL` | Private controller URL, such as `http://<controller-tailnet-ip>:8766/notify`. |
| `QUEUE_NOTIFY_TOKEN` | Shared token for readiness callbacks. |
| `HF_TOKEN` | Optional checkpoint publication. |

`QUEUE_CONTROLTOKEN` is also accepted as the notebook secret label for the control token.

Create control and notification tokens locally without printing their values:

```powershell
python -m queue_runner.cli tokens --directory "$env:USERPROFILE/.cache/kaggle-queue-runner"
```

Copy the values from those private files into Kaggle Secrets; do not paste them into notebook code or GitHub. Kaggle Secrets currently require configuration in the notebook UI. The default `--network auto` continues useful work if networking is unconfigured; use `--network required` when private control is essential.

The worker forwards tailnet TCP port 8765 to a loopback-only HTTP control server. The connection is encrypted by Tailscale; there is no public listener or Funnel. Tailnet policy must permit the operator to reach that port. A bearer token is additionally required for every endpoint.

## Readiness notifications and controls

Run a private receiver on your machine:

```powershell
python -m queue_runner.cli listen --root "$env:USERPROFILE/.cache/kaggle-queue-runner/inbox" --token-file "$env:USERPROFILE/.cache/kaggle-queue-runner/notify.token"
tailscale serve --bg --tcp=8766 tcp://127.0.0.1:8766
```

Add this handler alongside any existing Serve configuration; do not reset unrelated handlers. Set `QUEUE_NOTIFY_URL` to this machine's Tailscale address and port. The worker reports allocation, readiness, job start/end and runner completion. Notifications are retained in `inbox.jsonl` and per-worker JSON files. Delivery failures are recorded and do not stop compute.

An optional `--agent-id <registered-agent-name>` forwards a short readiness message through `ai-devkit agent send`. That requires a running, registered AI DevKit agent. Tailscale and the inbox alone cannot wake an inactive chat. No AI DevKit agents were registered on the development machine when this package was built, so this integration is provided but not enabled there.

Use the `control_url` received in the private inbox:

```powershell
python -m queue_runner.cli control --url http://<worker-tailnet-ip>:8765 --token-file <control.token> status
python -m queue_runner.cli control --url http://<worker-tailnet-ip>:8765 --token-file <control.token> logs --job-id train-v1
python -m queue_runner.cli control --url http://<worker-tailnet-ip>:8765 --token-file <control.token> checkpoint
```

Controls: `reload`, `pause`, `resume`, `checkpoint`, `cancel`, `stop`. Pause affects the next job; cancel terminates the current job; stop drains the current job and ends the runner. Reload requests a fresh queue read when the scheduler is next idle. There is no unauthenticated shell endpoint. Repository write access is intentionally authority to execute workload code with that job's named secrets.

## Validation

Run `python -m pytest -q`. Tests use actual child processes and Git repositories, including adding a second job while the first is running. They cover immutable snapshots, receipts, timeouts, secret redaction, authenticated controls and callbacks, dependency validation, checkpoint sealing and manifest-last publication with an injected storage boundary.

Linux userspace startup, authenticated tailnet enrollment, Kaggle execution and agent delivery are separate integration checks. Passing local unit tests does not establish those live connections. See `VALIDATION.md` for recorded evidence.
