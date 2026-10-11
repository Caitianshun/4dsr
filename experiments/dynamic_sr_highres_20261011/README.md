# High-resolution dataset validation (fresh namespace)

This execution adapter follows the local Claude plan and the user's requested
dataset/headroom validation. It preserves historical code and results.

Run CPU `selfcheck.py` first. Use `prepare_colmap_dataset.py --z1-only --z1-rung
half` to check calibration/mapping in prepared HR pixel units. Only requested
rungs passing geometric and patch NCC gates can be prepared. Full-rung failures
remain archived and do not justify relaxed thresholds.

Data preparation validates full actual decode counts and packet presentation
timestamps, retains source frame numbers, uses published COLMAP calibration,
preserves complete backgrounds, and chooses its 300-frame window from one
training camera's half-rung LR only. Both rungs share that window. It stores
all LR, cam00 HR, and ten cam01 HR diagnostic frames; other train HR is absent.
`initialize_highres.py` calls the unchanged train-LR-only SIFT triangulation.

`register_highres.py lr/sr` creates consumers for the unchanged cullretrain
training interface. Non-N3DV uses author dynerf/default.py unchanged: LR coarse
3000 + fine14000 steps with batch4. N3DV Cook/Cut use their scene batch2.
SR refinement uses6000same-time three-view rows. DeskGames20train cameras reuse
the native balanced algorithm. Long36032train cameras require a disclosed
sampling interface adaptation: rotate20-camera subsets per frame; all32cameras
appear, corresponding-column exposures match, 6000/9600camera-time keys are
covered. All arms share exactly one schedule; no equal cross-dataset compute
claim follows from equal iteration counts.

`teacher_highres.py plan/cost/run/accept` uses the frozen original untiled
SwinIR generator (tile0; pinned generator/network/checkpoint SHA). Cost generates
three train LR images, records actual time and peak memory, and full run resumes
the generator's own hash-checked cache. Only CPU acceptance of all teacher
PNG/receipt identities enables SR registration. The full training-teacher
population remains complete even for Long360's subset SR-refinement schedule.

`expand_hr_highres.py` writes separate privileged train HR for oracleHR and
HRref. Each frame must re-degrade to the exact prepared LR PNG SHA. `entry.py`
dispatches lr/sr/eval to unchanged cullretrain and reference-source switches to
unchanged refs.py. It rejects output overwrite and preserves failure receipts.

The minimal matrix is LR parent17k, matched6000continuations λ0, SRλ0.2,
oracleHR; SRrecon17k trains from frozen SR pixels, and HRref17k trains from
privileged HR pixels. Their budgets differ from parent17k+SR6k and must be
reported explicitly. All GPU work requires supervising allocation and gate
acceptance. Scripts are local research assets; no dataset/media or results are
publicly redistributed.
