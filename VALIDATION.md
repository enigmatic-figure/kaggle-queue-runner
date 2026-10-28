# Validation, 2026-10-28

Observed results, rather than deployment assumptions:

| Check | Result |
|---|---|
| Windows Python 3.13 tests | 12 passed; one Linux-only test skipped. |
| WSL Ubuntu / Linux tests | All 13 passed. |
| Full worker CLI smoke test | Two sequential jobs completed and produced closed checkpoints. No external credentials or accelerator allocation used. |
| Dynamic Git queue test | A second job committed while the first was running was fetched and executed afterward. The first retained its original commit snapshot and was not repeated. |
| Storage boundary test | Payload hashes verified before publishing the checkpoint marker; repeat publications skipped; modified closed payload rejected. Uses an injected storage client. |
| Private callback | A synthetic callback reached the running Windows receiver through its Tailscale Serve TCP handler. This is a controller self-test, not a Kaggle allocation event. |
| Linux Tailscale binary | Official 1.104.4 archive SHA256 verified; userspace daemon started with in-memory state and reported NeedsLogin. No authenticated enrollment or host-tailnet change. |
| Notebook | Structure and Python code validated. Stable bootstrap fetches GitHub only after allocation. |
| Wheel | Built successfully with pip wheel. |
| Vendored numerical source | All four source-file SHA256 values match vendor/SOURCE.json. |
| Public publication | Published commit `212271c6857de7f13e24f9d5f6c9723f48ea4097`; anonymous clone succeeded and its tracked tree exactly matched the local source. |
| Published-copy verification | All 13 Linux tests passed; full CLI smoke completed two sequential jobs and two closed checkpoints; vendored hashes verified. |

The original standalone pair-force check has also completed on an actual Kaggle eight-device TPU v5e allocation: force error 1.43e-8 maximum, distance error 1.05e-7 maximum against the Float64 reference. It did not execute this generic runner, its owner-force reduction, or the real-data trial, and its allocation ended. It is not an end-to-end throughput result.

The generic notebook is a private saved draft at https://www.kaggle.com/code/finnbogiabrahamson/kaggle-queue-runner . Saving the draft did not request an allocation. Its public source is https://github.com/enigmatic-figure/kaggle-queue-runner . Attach the networking secrets to enable private controls and callbacks before submission.

Pending integration checks:

- Actual Kaggle execution of the generic bootstrap and geometry workload.
- Kaggle worker enrollment into the tailnet with the supplied ephemeral key.
- Worker-to-controller notification over that authenticated connection.
- Delivery to a registered AI DevKit agent. None were registered on this development machine.
- Real cloud checkpoint publication by this package. The storage protocol is tested locally; no publishing secret is embedded in the code.

The private receiver is time-limited to 48 hours. Its tokens and inbox are outside this repository. No completed full-corpus geometry optimization or retrieval improvement is claimed.
