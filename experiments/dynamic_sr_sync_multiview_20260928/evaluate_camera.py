"""Compose the inherited evaluator adapter and add identity/ROI hooks only."""
import importlib.util
import types
from dv_common import ROOT,HERE
from evaluation_adapter import validate_model,roi_metrics


def main():
    path=ROOT/'experiments/dynamic_sr_controlled_headroom_20260926/evaluate.py'
    spec=importlib.util.spec_from_file_location('inherited_controlled_evaluation',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    # Preserve every original controlled adapter substitution and metric operation.
    # The proxy only inserts metadata validation and fixed two-frame ROI diagnostics.
    source_path=original.ORIGINAL
    class SourceProxy:
        def read_text(self):
            s=source_path.read_text()
            needle='    detail_metadata = model.checkpoint.get(\'detail_supervision\')'
            assert s.count(needle)==1
            s=s.replace(needle,'    sync_family = validate_sync_model(model)\n'+needle)
            needle="                legacy.write_rgb(args.out / 'predictions'"
            assert s.count(needle)==1
            s=s.replace(needle,"                if observation['frame_index'] in [40,80]:\n                    row['roi_float'] = sync_roi_metrics(legacy,pred,gt,roi_protocol['regions_by_camera_xyxy_exclusive'].get(camera,{}),metric,args.lpips_device)\n"+needle)
            return s
        def __str__(self):return str(source_path)
    original.ORIGINAL=SourceProxy()
    # The existing adapter executes its generated module. Supply hooks through
    # its local ModuleType factory; no monkey patch of global types or metrics.
    def module_factory(name):
        m=types.ModuleType(name);m.validate_sync_model=validate_model;m.sync_roi_metrics=roi_metrics
        return m
    original.types=types.SimpleNamespace(ModuleType=module_factory)
    original.__file__=str(HERE/'evaluate_camera.py')
    original.main()


if __name__=='__main__':main()
