#!/usr/bin/env python3
"""CPU proof that the Pillow dtype adaptation preserves native RGB bytes."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace


def main(a):
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0,str(a.runtime.resolve()))
    helper=ModuleType("utils.general_utils")
    source=a.runtime/"utils/general_utils.py"
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="PILtoTorch"]
    scope={"torch":torch,"np":np}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),"exec"),scope)
    helper.PILtoTorch=scope["PILtoTorch"]
    sys.modules["utils.general_utils"]=helper
    loader=importlib.import_module("utils.data_utils")
    manifest=json.loads(a.manifest.read_text())
    rows=[r for r in manifest["observations"] if r["split"]=="train"
          and r["camera_id"] in (manifest["splits"]["train"][0],manifest["splits"]["train"][-1])
          and r["frame_index"] in (0,149,299)]
    checked=[]
    for row in rows:
        path=a.manifest.parent/row["lr_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row["lr_sha256"]
        with Image.open(path) as image:
            rgba=np.asarray(image.convert("RGBA"))
        norm=rgba/255.0
        arr=norm[:,:,:3]*norm[:,:,3:4]
        signed=np.asarray(arr*255.,dtype=np.byte)
        unsigned=np.asarray(arr*255.,dtype=np.uint8)
        if signed.tobytes()!=unsigned.tobytes():
            raise ValueError("Image dtype adaptation changed native raw bytes")
        expected=scope["PILtoTorch"](Image.frombytes("RGB",tuple(manifest["resolutions"]["lr"]),signed.tobytes()),tuple(manifest["resolutions"]["lr"]))
        camera=SimpleNamespace(meta_only=True,image_path=str(path),resolution=tuple(manifest["resolutions"]["lr"]),
                               image_width=manifest["resolutions"]["lr"][0],image_height=manifest["resolutions"]["lr"][1])
        actual,_=loader.CameraDataset([camera],False)[0]
        if not torch.equal(actual,expected):
            raise ValueError("Actual native CameraDataset differs from original raw RGB semantics")
        checked.append({"camera":row["camera_id"],"frame":row["frame_index"],"source_sha256":row["lr_sha256"],
                        "dtype_bytes_equal":True,"native_loader_tensor_equal":True,"shape":list(actual.shape)})
    report={"status":"passed","GPU_calls":0,"parameter_updates":0,
            "manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            "native_loader_sha256":hashlib.sha256((a.runtime/"utils/data_utils.py").read_bytes()).hexdigest(),
            "checked":checked,"interpretation":"uint8 preserves the original signed-int8 raw 8-bit RGB bytes, while allowing current Pillow to construct the image"}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"passed","actual_native_loader_samples":len(checked)}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
