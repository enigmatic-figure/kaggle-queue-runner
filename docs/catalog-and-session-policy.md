# Selectable deployments and an allocation-sized useful-work queue

Model choice, deployment layout, task backlog, and session policy are separate JSON documents. They are stored in this repository and read from the same pinned Git commit as a job. Nothing requires Prisma, a hosted database, or one particular agent framework. `catalog.resolve` accepts a JSON reader callback; another configuration store can implement that boundary later without changing the document format.

The runner currently implements the vLLM JAX adapter with a hash-locked environment. The `backend` field identifies that adapter explicitly. Additional engine adapters can be added; an unknown adapter is rejected rather than silently treated as compatible.

## Files to select work

| File | Purpose |
|---|---|
| `catalog/models.json` | Nine model entries with immutable Hugging Face revisions, reported parameter metadata, roles and validation status. |
| `catalog/sets.json` | Named deployment sets and named backlog documents. |
| `catalog/selection.json` | Active default set, selected backlogs, idle policy and checkpoint reserve. |
| `profiles/*.json` | Chip/CPU placement, API port, precision, pooling/generation runner, context and time budgets. |
| `backlogs/model-profiling.json` | Eight finite useful profiling jobs, each selecting its own deployment set. |
| `catalog/agent-policy.json` | Preferred agent alias, private trace format and optional teacher configuration. |

Qwen3-Coder-30B-A3B-Instruct is the **preferred agent model**, without being a required model for other roles. Each job can select a different `service_set`. Set selection is independent of the public API protocol and the session trace format.

The committed selection remains inactive: no model set or profiling backlog has been activated for the already queued geometry run. Its notebook and queued version have not been replaced.

## Candidate configurations

These are prepared candidates, not verified v5e deployments. All have local TPU validation marked `not_run`. The [pinned upstream model matrix](https://docs.vllm.ai/projects/tpu/en/v0.32.0/recommended_models/) contains useful compatibility evidence, but its tested hardware and model variants must not be confused with a local Kaggle result.

| Set | Initial layout | Role / qualification |
|---|---|---|
| `qwen3-coder-30b` | Eight chips, BF16 | Default agent/teacher candidate; tool-call parser enabled. |
| `dual-embedding` | Two independent Qwen3-Embedding-0.6B processes, one chip each | Separate CPU slices, API ports and libtpu ports; compare latency/correctness and parallel throughput later. |
| `qwen3-embedding-4b` | One chip, BF16 | Pooling/embedding candidate; fit and kernels need measurement. |
| `qwen3-embedding-8b` | Four chips, BF16 | Larger embedding reference. |
| `gpt-oss-20b` | Four chips, native checkpoint / auto dtype | Experimental: MXFP4 checkpoint handling on v5e is unverified. |
| `gemma4-moe` | Eight chips, BF16 | Gemma 4 26B-A4B instruction model. |
| `gemma4-dense` | Eight chips, BF16 | Gemma 4 31B instruction model. |
| `qwen35-9b` | Four chips, BF16 | General-purpose candidate; v5e path needs testing. |
| `qwen35-397b` | Blocked resident profile | An offload/compression research entry, excluded from the profiling backlog. |

The [Coder card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) reports about 30.5B total parameters and 3.3B active per token. Its BF16 weights alone are roughly 61 GB. The initial eight-chip layout leaves useful headroom; a four-chip quantized variant needs a separately pinned checkpoint and a validated TPU kernel/loader. A quantized local GPU deployment is not evidence that the BF16 checkpoint fits a four-chip TPU budget.

For [Qwen3.5-397B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), total weight storage still depends on the full model. The pinned Hugging Face metadata reports about 403.4B parameters including additional components: about 807 GB at BF16, or a theoretical 202 GB at four bits before scales and runtime overhead. That cannot be resident in roughly 64 GB HBM. Host/expert offloading is a separate implementation and bandwidth experiment. The blocked entry preserves it on the shortlist without triggering a huge unsupported download.

One-, four-, and eight-chip layouts are supported by the planner. For partial slices it verifies the observed chip-to-NUMA relationship; an eight-chip profile deliberately spans both chip groups. CPU affinity remains explicit. Each resident process needs disjoint CPUs/chips and unique API/libtpu ports. The full deployment set is checked before any child starts. Changing the set stops the old processes before starting their replacements.

## Prepare a future payload

For example, prepare a nine-hour budget with the default agent and profiling backlog:

```powershell
python tools/prepare_run.py --hours 9 --output C:/QueuePlans/first-catalog-session
```

This creates a notebook, Kaggle metadata and `selection.to-publish.json`. It submits nothing, changes no running queue, and does not claim that the requested duration or weekly quota is available. Use the duration actually permitted for the chosen allocation. The existing queued version requests eight hours; new GitHub configuration cannot extend that server-side request. Kaggle's [notebook documentation](https://www.kaggle.com/docs/notebooks) describes notebook execution limits, which must be checked again when choosing the next run.

Review the staged selection and publish it to `catalog/selection.json` when configuring the intended run. Its useful-work policy is:

```json
{
  "active_set": "qwen3-coder-30b",
  "backlogs": ["model-profiling"],
  "idle_mode": "await_work",
  "serve_when_idle": true,
  "idle_resources": {"tpu_chips": [], "numa_node": 1, "cpu_threads": 1},
  "checkpoint_reserve_seconds": 300
}
```

`--set <name>` chooses another default. Repeated `--backlog <name>` selects finite queues; `--backlog none` prepares API availability without that profiling queue. Build additional backlog documents for real graph optimization, CPU CMA-ES objectives, spaCy annotation with a defined label schema, RL teacher requests or dataset enrichment. The runner does not invent task objectives or repeat completed jobs to manufacture activity.

In `await_work` mode, an empty primary queue no longer exits after five minutes. It continues to accept new GitHub work and can make the selected model set the primary use of the TPU through its private API. Normal polling waits do not extend Kaggle's limits. An explicit stop or the session deadline still ends the worker. `drain` preserves the older finite-queue behavior.

The scheduler subtracts the checkpoint reserve before assigning job timeouts and stops dispatching near the deadline, leaving time for service shutdown and artifact synchronization. Individual jobs still implement their own checkpoint format. This reserve does not make an arbitrary task resumable or guarantee survival of a platform termination. Existing resident models are retained near the deadline even when there is no longer enough time to start a new model.

## Profiling and task roles

The prepared backlog runs the current geometry trial and then selects each eligible model set for `workloads/profile_services.py`. The script waits for authenticated endpoint readiness, sends a small deterministic prompt set, measures startup wait and end-to-end request latency, checks embedding dimensions/norms or records generation responses/usage, and seals each measurement plus a summary. Load failures are recorded as outcomes; they are not model-quality scores.

The default request count is three; `--requests 12` uses the full short prompt collection. These are startup and API probes, not saturated throughput, first-token latency, retrieval quality, peak-memory or agent benchmarks. Those require dedicated workload inputs. The backlog is best effort within the real session budget; later jobs can be deferred if time is insufficient.

Each job's `service_set` controls which model layout it needs. The default selected set is resumed when the primary queue is idle. Thus the same vLLM deployment can support a primary API session, a secondary annotator, or a teacher without embedding any one role into the engine implementation.

## Session logs and optional teachers

`queue_runner.inference_client.Session` is a standard-library, OpenAI-compatible client. It writes `queue-session-jsonl-v1` records containing source/model metadata, requests, returned responses, elapsed times and explicit tool-result records. It does not log authorization headers and redacts its known API credential. The trace file is private and is not automatically sealed or published by the profiling job. Preserve task outcomes, failed attempts, teacher involvement and human corrections when creating training examples; an unreviewed successful API response is not automatically a correct training target.

This is an interchange boundary for an agent or data worker. A terminal-capable agent adapter still needs to be selected and tested with the actual endpoint. [Qwen Code supports custom providers](https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/) and [session export](https://qwenlm.github.io/qwen-code-docs/en/developers/qwen-serve-protocol/); it is one possible consumer, alongside other frameworks. The runner does not currently execute an autonomous terminal loop.

`catalog/agent-policy.json` reserves a provider-independent teacher endpoint, model alias, secret label and per-task escalation budget. It is disabled and contains no endpoint or credentials. The future agent/task policy must decide when remote escalation is allowed and account for those calls. No external teacher calls, RL updates or fine-tuning are launched by preparing this catalog.

The callback address remains in the private Kaggle Secret and local receiver configuration. It is not written into the public model catalog. Existing Tailscale controls and API bearer authentication apply to every model instance.
