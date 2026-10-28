# CPU analytics alongside resident TPU inference

The queue worker can keep one vLLM service resident across compatible primary jobs. Primary jobs still run sequentially. The model is a separate process group, virtual environment, cache, and device partition. It can serve local data-generation code or authenticated callers on the tailnet.

The existing queued notebook does not need a new version. It clones this repository after allocation. No second Kaggle submission or reservation is involved.

## Resource layout

Google's [v5e architecture documentation](https://docs.cloud.google.com/tpu/docs/v5e) specifies two NUMA nodes on an eight-chip VM and 16 GB nominal HBM per chip. The previous actual Kaggle allocation exposed eight devices, about 405.7 GB host RAM, and a 96-CPU allowed affinity set on a 224-logical-CPU host. A four-chip partition has about 64 GB nominal HBM in total; weights, KV cache, activations and runtime allocations share it.

The starter profile reserves physical TPU chips 0–3 and eight allowed CPUs on NUMA node 0. A compatible primary CPU job uses NUMA node 1. The scheduler reads Linux sysfs chip-to-NUMA associations and intersects each node's CPUs with the container's allowed affinity. It does not assume all host CPUs are available or infer chip locality from CPU count. Unavailable or contradictory locality information defers inference and leaves the primary workload running.

CPU affinity alone does not select TPU chips. Every process receives its device visibility before importing JAX or vLLM. The four-chip service uses `TPU_VISIBLE_CHIPS`, `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1` and `TPU_PROCESS_BOUNDS=1,1,1`; it also gets a separate libtpu process port. The [upstream process-partition example](https://github.com/vllm-project/tpu-inference/blob/v0.32.0/examples/disagg/run_disagg_single_host.sh) demonstrates the same family of runtime controls. The particular four-chip/Kaggle combination remains a live integration check.

These are cooperating process reservations, not a container security boundary. CPU affinity is enforced for descendants. Host RAM remains shared, with Linux's normal allocation policy; this implementation does not enforce NUMA memory binding or a cgroup RAM cap. DuckDB has its own explicit memory and spill budget. An arbitrary primary program must honor its resource declaration.

## Enable a service

`workloads.json` includes this disabled entry:

```json
"services": [
  {"id": "qwen3-8b-v1", "profile": "profiles/qwen3-8b-v5e4.json", "enabled": false}
]
```

The supplied profile pins `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, TP=4, 4096-token context, four concurrent sequences, and 75% memory utilization. Its startup deadline is 30 minutes; its lifetime limit is two hours. These are initial budgets, not measured capacity or latency claims. Change the service ID for an explicit retry after a failed launch. A changed profile also gets a new process and provenance receipt.

Set `enabled` to true when a compatible useful primary workload is ready. Changing GitHub affects the next job boundary; it does not hot-swap a running service or current job. The service survives adjacent compatible jobs with an unchanged ID/profile. A job without `resources` reserves all eight chips and all CPUs conservatively. It stops the service before the primary child starts. The existing `geometry-8192-v1` job retains that exclusive behavior.

For a CPU task, declare a separate host partition:

```json
{
  "id": "parquet-audit-v1",
  "script": "workloads/cpu_task.py",
  "args": ["--script", "workloads/parquet_audit.py", "--", "--input", "/tmp/prepared-data/part-*.parquet", "--memory-gb", "32"],
  "resources": {"tpu_chips": [], "numa_node": 1, "cpu_threads": 24},
  "timeout_seconds": 14400,
  "min_remaining_seconds": 1800
}
```

The input is illustrative: select real prepared Parquet files before enabling this job. It writes row counts, null counts, and numeric min/max/mean/standard deviation into a sealed checkpoint. Large scans spill under `/tmp`, subject to actual available storage. Its 32 GB budget is a DuckDB setting, not a session-wide memory guarantee.

`resources.tpu_chips=[]` sets JAX and PJRT to CPU and clears TPU visibility for that child. Its wrapper applies Linux CPU affinity before importing workload dependencies. BLAS/OpenMP/NumExpr thread counts follow the declared CPU budget. Libraries with their own worker pools still need an explicit pool size; avoid 24 workers each starting 24 BLAS threads.

`workloads/cpu_task.py` creates an independent CPU environment with `cma==4.5.0`, `duckdb==2.0.0`, `pyarrow==26.0.0`, and `numpy==2.5.4`. Use the same wrapper with a repository script implementing the actual CMA-ES objective. No placeholder optimization is scheduled.

## Reproducible inference environment

The official [vLLM TPU installation](https://docs.vllm.ai/projects/tpu/en/v0.31.0/getting_started/installation/) provides the `vllm-tpu` package. The current upstream project is transitioning to the name vLLM JAX; this integration pins the already published `vllm-tpu==0.32.0` release.

`environments/vllm-tpu-py312.lock` contains all 240 resolved packages and wheel hashes for Linux CPython 3.12. The installer uses `--require-hashes --only-binary=:all:` and CPU PyTorch wheels. It does not upgrade Kaggle's JAX installation. The resolved model stack uses JAX/JAXlib 0.11.2, libtpu 0.0.48, and PyTorch 2.10.0+cpu. These differ from the geometry environment deliberately.

The cache and virtual environment live under `/tmp/queue-model-cache`; actual available space is checked before installation. The lock digest identifies the environment. A successful `pip check`, installed freeze, selected-device report, source commit, profile hash and CPU/chip reservations are retained under the run's `services/` directory. A short-lived selected-device probe exits before the serving process starts.

Optional inference setup failures do not abort the primary CPU task. A failed service is not automatically restarted in a loop. The entire process group is terminated at an incompatible job boundary or runner shutdown. The ordinary queue/session time limits still apply; inference does not keep an otherwise idle allocation alive indefinitely.

## Private API, status and notifications

The local OpenAI-compatible API binds to `127.0.0.1:8000`. The worker's existing Tailscale daemon can forward private TCP port 8000 to it, alongside the unchanged queue control port 8765. The API uses `VLLM_API_KEY`, supplied from the queue control token. No key is written to the profile, command line or publication receipts. Tailnet policy must allow the operator to reach port 8000.

The authenticated queue `/status` response includes `inference`, including phase, original source commit and the private API URL. Phases include starting, installing, device_check, loading, ready, deferred, failed and stopped. Readiness requires an authenticated `/v1/models` response; this verifies model discovery, not text-generation quality. A `service_ready` callback carries the private API URL to the existing receiver. Failure and shutdown are also reported.

Use the existing token file on your machine:

```python
import json, urllib.request
from pathlib import Path
token = (Path.home()/'.cache/kaggle-queue-runner/control.token').read_text().strip()
request = urllib.request.Request(
    'http://<worker-tailnet-ip>:8000/v1/chat/completions',
    data=json.dumps({'model':'Qwen/Qwen3-8B', 'messages':[
        {'role':'user','content':'Summarize these experiment results: ...'}]}).encode(),
    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)
```

A primary dataset-generation task can call the loopback API. Give that job `"secrets":["QUEUE_CONTROL_TOKEN"]` to receive its API credential explicitly; the `QUEUE_CONTROLTOKEN` Kaggle label is also accepted. It must wait for readiness or check status before making requests. The model service itself provides inference. Tool access, worktree modification policies and a resident agent loop are separate tasks to build on the API.

## Evidence and remaining checks

- Linux tests exercise real child processes, CPU affinity, visibility, residence across CPU jobs, unloading before an exclusive TPU job, failed launches and queue integration, using a lightweight substitute server. They do not execute a TPU model.
- The exact 240-package hash lock resolves successfully on Linux CPython 3.12. Dependency resolution does not prove model execution.
- `tools/smoke_cpu.py` installs the real CPU environment, checks a five-row Parquet table's statistics and sealed checkpoint, and runs a small deterministic CMA-ES test without accelerators.
- Pending allocation: observed four-chip visibility, topology mapping inside Kaggle, actual vLLM loading/generation, simultaneous workload memory/latency, and private remote inference. Enable a controlled compatible workload for that check before relying on throughput or large-model capacity.
