import torch as th
import torch.nn as nn

from utils.debug import *
from utils.state_parsing import StateParsing


class Actor(nn.Module):
    def __init__(self, args, cnn, cnn_coarse, preference_predictor=None) -> None:
        super(Actor, self).__init__()
        self.args = args
        self.cnn = cnn
        self.cnn_coarse = cnn_coarse
        self.preference_predictor = preference_predictor
        self.merge = nn.Conv2d(2, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

        self.state_parsing = StateParsing(args)
        self.grid = args.grid

    def forward(self, x):
        # cnn_input contains all masks (6 channels)
        cnn_input = x[:, 1 + self.grid * self.grid * 2: 1 + self.grid * self.grid * 8].reshape(-1, 6, self.args.grid, self.args.grid)
        position_mask = self.state_parsing.state2position_mask(x, next_next_macro=False)
        position_mask = position_mask.flatten(start_dim=1, end_dim=2)
        cnn_res = self.cnn(cnn_input)

        old_canvas = self.state_parsing.state2canvas(x, new=False).unsqueeze(1)
        new_canvas = self.state_parsing.state2canvas(x, new=True).unsqueeze(1)
        structural_mask = self.state_parsing.state2structural_mask(x, next_next_macro=False).unsqueeze(1)
        next_structural_mask = self.state_parsing.state2structural_mask(x, next_next_macro=True).unsqueeze(1)
        wire_mask = self.state_parsing.state2wire_mask(x, next_next_macro=False).unsqueeze(1)
        next_wire_mask = self.state_parsing.state2wire_mask(x, next_next_macro=True).unsqueeze(1)
        coarse_input = th.cat([old_canvas,
                               new_canvas,
                               structural_mask,
                               wire_mask,
                               next_structural_mask,
                               next_wire_mask],
                              dim=1)
        coarse_res, _ = self.cnn_coarse(coarse_input)
        cnn_res = self.merge(th.cat([cnn_res, coarse_res], dim=1))

        # CarePlace preference-aware mask weighting
        use_perception = getattr(self.args, 'use_perception_guidance', False)

        if use_perception and self.preference_predictor is not None:
            # Compute preferences with gradient flow through predictor
            alpha_w, alpha_s = self.preference_predictor.predict_from_state_batch(
                x, self.state_parsing
            )
            # Detach for mask computation (paper: "alpha_w is detached during reward combination")
            # For policy mask, we DON'T detach so gradients flow back
            wire_weight = alpha_w.view(-1, 1)
            struct_weight = alpha_s.view(-1, 1)
        else:
            # Read pre-computed alphas from state (no gradient path)
            alpha_w = x[:, -3]
            alpha_s = x[:, -2]
            wire_weight = alpha_w.view(-1, 1)
            struct_weight = alpha_s.view(-1, 1)

        mask2 = wire_weight * wire_mask.flatten(start_dim=1) + \
                struct_weight * structural_mask.flatten(start_dim=1) + \
                position_mask * 10
        mask_min = mask2.min(dim=1, keepdim=True).values + self.args.soft_coefficient
        mask2 = mask2.le(mask_min).logical_not().float()

        x_out = cnn_res.reshape(-1, self.grid * self.grid)
        x_out = th.where(position_mask + mask2 >= 1.0, -1.0e10, x_out.double())
        x_out = self.softmax(x_out)
        return x_out
