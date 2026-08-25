"""Build-time import and compatibility checks for the model-server image."""

import accelerate
import diffusers
import torch
import torchvision
import transformers

from internnav.agent import internvla_n1_agent_realworld
from internnav.model.basemodel.internvla_n1.nextdit_crossattn_traj import (
    NextDiTCrossAttn,
    NextDiTCrossAttnConfig,
)
from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2


assert torch.__version__.startswith('2.6.0'), torch.__version__
assert torchvision.__version__.startswith('0.21.0'), torchvision.__version__

# Instantiation exercises Diffusers' gradient-checkpointing integration, which
# a plain module import does not cover.
nextdit = NextDiTCrossAttn(NextDiTCrossAttnConfig())
del nextdit

print(
    'InternNav model-server imports OK:',
    torch.__version__,
    torchvision.__version__,
    torch.version.cuda,
    transformers.__version__,
    diffusers.__version__,
    accelerate.__version__,
    DepthAnythingV2.__name__,
    internvla_n1_agent_realworld.__name__,
)
