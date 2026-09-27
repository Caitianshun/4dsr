"""Default RGB call is a direct delegation to the original accepted renderer.

The historical refactor remains unmodified and is retained as diagnostic data.
"""
from context import motion

RENDERER_ID = 'legacy_direct_v1'

def render_model(model,camera):
    return motion.render_model(model,camera)
