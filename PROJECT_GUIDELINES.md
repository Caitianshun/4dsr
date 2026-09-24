# Public repository guidelines

## Publication scope

Publish only the project's non-HOI super-resolution source code and necessary text configuration. The current full-scene route covers N3DV and MeetRoom; the ZJU single-person experiment is retained only as a labeled historical non-HOI branch. Keep the source code's experimental behavior and provenance intact.

Never publish datasets, images/videos, pretrained weights, checkpoints, evaluation outputs, logs, caches, paper PDFs, personal credentials, or copied third-party installation/source trees. Keep HOI-specific code and configurations out of this repository, including scripts that depend on HODome/HOI-M3, human/object instance masks, or HOI model priors. Refer to upstream code by version and preserve needed local patches with clear attribution.

## Update procedure

After a non-HOI SR code/configuration change is complete and its relevant checks pass, publish it in the same work session using the source workspace's `python3 scripts/publish_generic_sr.py --publish`. The script copies an explicit allowlist into a separate clean clone, commits and pushes `main`, and verifies the remote commit. Check the proposed file list before expanding its allowlist. If the push fails, report the unsynced state and retry after resolving the cause. Do not force-push or call a local commit an upload.

The publication command operates on finished changes; it does not watch every file save or copy partially written files. New non-HOI source directories require a deliberate allowlist update and a fresh scope check. The private workspace's `AGENTS.md` is the operational rule for future research tasks; this public file records the repository boundary without private infrastructure details.
