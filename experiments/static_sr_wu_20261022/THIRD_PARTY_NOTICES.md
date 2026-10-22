# Third-party method provenance

These modules are explicit adaptations to the Wu time-conditioned Gaussian backbone. They are not original algorithms of this project and are not untouched author reproductions. Full author checkouts, networks, checkpoints, datasets and compiled binaries are not distributed here.

| Adapter | Author repository and pinned commit | Retained license or notice |
| --- | --- | --- |
| SplatSuRe and contributing-pixel preparation | https://github.com/pranav-asthana/SplatSuRe `df40abb8d92a02656268cb0a0b97acc2495b9307` | `methods/splatsure_UPSTREAM_LICENSE.txt` |
| IE-SRGS losses, priors and RADE bridge | https://github.com/ZheXues/IE-SRGS `74d60d5bb0f49974ac0efa3497dbbd9e246480f4` | `methods/iesrgs_NOTICE.txt`, `methods/iesrgs_UPSTREAM_LICENSE.md` |
| Sequence Matters sequence construction and PSRT bridge | https://github.com/DHPark98/SequenceMatters `06d9e7fa359f65e6fdb6b96e48786d6028de0caf` | `methods/sequence_matters_UPSTREAM_LICENSE.md` |
| 3DSR diffusion feedback and Mip bridge | https://github.com/Consistent3DSR/3DSR `8c70a9f64301b4de255672e4aa7da42acc0001a2` | `methods/threedsr_UPSTREAM_LICENSE.md` |

Author functions and teacher networks loaded at runtime remain subject to their respective upstream licenses. StableSR, SwinIR, PSRT, Depth Anything V2 and their dependencies must be obtained separately. No blanket project license supersedes those upstream terms. See the README for source recipe differences and dynamic adaptation boundaries.
