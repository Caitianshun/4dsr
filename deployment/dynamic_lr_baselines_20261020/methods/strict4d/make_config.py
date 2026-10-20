#!/usr/bin/env python3
"""Keep native 30k/batch4 schedules; substitute paths and exact registered LR."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--runtime", type=Path, required=True)
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--config", type=Path, required=True)
a = p.parse_args()
receipt = json.loads((a.dataset / "stage_receipt.json").read_text())
scene = receipt["scene"]
text = (a.runtime / "configs/dynerf" / (scene + ".yaml")).read_text()
text = text.replace(f'  source_path: "data/N3V/{scene}"', "  source_path: " + json.dumps(str(a.dataset.resolve())))
text = text.replace(f'  model_path: "output/N3V/{scene}"', "  model_path: " + json.dumps(str(a.output.resolve())))
text = text.replace("  resolution: 2", "  resolution: 1")
text = text.replace("exhaust_test: True", "exhaust_test: False")
# Test metrics are computed only after the terminal checkpoint, by a separate
# heldout renderer. Native training never sees cam00 pixels or selects on them.
text += "\ntest_iterations: []\n"
a.config.parent.mkdir(parents=True, exist_ok=True)
a.config.write_text(text)
print(a.config)
