"""Apply confirmed compiler, missing-statistic and empty-TR fixes to Multi4D.

Clone https://github.com/BatFaceWayne/Multi4D and checkout the COMMIT before
calling. Install README reconstruction dependencies in a separate Python env;
compile its own simple-knn, rasterizer and hybrid rasterizer. This helper does
not change the original method, install semantic tools, or touch the Wu env.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from build_adapter import COMMIT


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path)
    a=p.parse_args();assert subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()==COMMIT
    edits=[('submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h','cstdint','CUDA 12.8: explicit fixed-width integer declarations'),
        ('diff_gaussian_rasterization_hybrid/cuda_rasterizer/rasterizer_impl.h','cstdint','CUDA 12.8: explicit fixed-width integer declarations'),
        ('submodules/simple-knn/simple_knn.cu','cfloat','CUDA 12.8: explicit FLT_MAX declaration')]
    result=[]
    for name,header,why in edits:
        path=a.upstream/name;original=subprocess.check_output(['git','-C',str(a.upstream),'show',COMMIT+':'+name])
        patched=(f'#include <{header}> // {why}\n').encode()+original
        assert path.read_bytes() in [original,patched],f'Unexpected third-party modification: {name}'
        if path.read_bytes()!=patched:path.write_bytes(patched)
        result.append(dict(path=name,sha256=sha(path)))
    replacements={
        'diff_gaussian_rasterization_hybrid/cuda_rasterizer/forward.cu':[
            ('  // All threads that treat valid pixel write out their final', '''  // BACKWARD::PerGaussianRenderCUDA reads this per-tile cutoff. The
  // released forward never wrote it, so allocation contents could zero or
  // truncate every hybrid gradient. Reduce the actual last contributor.
  __shared__ uint32_t tile_last_contributor[BLOCK_SIZE];
  tile_last_contributor[block.thread_rank()] = inside ? last_contributor : 0;
  block.sync();
  for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
    if (block.thread_rank() < offset)
      tile_last_contributor[block.thread_rank()] = max(
          tile_last_contributor[block.thread_rank()],
          tile_last_contributor[block.thread_rank() + offset]);
    block.sync();
  }
  if (block.thread_rank() == 0) max_contrib[tile_id] = tile_last_contributor[0];

  // All threads that treat valid pixel write out their final'''),
            ('    uint32_t *tiles_touched, bool prefiltered) {\n  preprocessCUDA<',
             '    uint32_t *tiles_touched, bool prefiltered) {\n  if (P == 0) return; // Empty TR leaves persistent branches active.\n  preprocessCUDA<')],
        'diff_gaussian_rasterization_hybrid/cuda_rasterizer/backward.cu':[
            ('    glm::vec4 *dL_drot_r, float *dL_dopacity) {\n  // Propagate',
             '    glm::vec4 *dL_drot_r, float *dL_dopacity) {\n  if (P == 0) return; // Avoid a zero-block CUDA launch for empty TR.\n  // Propagate')],
        'diff_gaussian_rasterization_hybrid/rasterize_points.cu':[
            ('  if (P != 0) {\n    CudaRasterizer::Rasterizer::backward(',
             '  if (P + P_static != 0) {\n    CudaRasterizer::Rasterizer::backward('),
            ('  int M = 0;\n  if (sh.size(0) != 0) {\n    M = sh.size(1);\n  }',
             '  int M = sh.dim() >= 2 ? sh.size(1) : 0; // Keep empty TR SH gradient shape.')]}
    for name,changes in replacements.items():
        path=a.upstream/name;original=subprocess.check_output(['git','-C',str(a.upstream),'show',COMMIT+':'+name]).decode();patched=original;allowed=[original]
        for before,after in changes:
            assert patched.count(before)==1,(name,before)
            patched=patched.replace(before,after)
            allowed.append(patched)
        assert path.read_text() in allowed,f'Unexpected modification: {name}'
        if path.read_text()!=patched:path.write_text(patched)
        result.append(dict(path=name,sha256=sha(path)))
    a.receipt.write_text(json.dumps(dict(commit=COMMIT,patches=result),indent=2)+'\n')


if __name__=='__main__':main()
