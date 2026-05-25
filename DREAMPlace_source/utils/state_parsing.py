import numpy as np
import torch as th


class StateParsing:
    def __init__(self, args) -> None:
        self.args = args
        self.grid = args.grid

    # Mapping from benchmark name to problem ID
    PROBLEM_IDS = {
        'superblue1': 0, 'superblue3': 1, 'superblue4': 2, 'superblue5': 3,
        'superblue7': 4, 'superblue10': 5, 'superblue16': 6, 'superblue18': 7,
    }

    def get_state(self,
        place_idx,
        old_canvas,
        new_canvas,
        structural_mask,
        position_mask,
        wire_mask,
        next_structural_mask,
        next_position_mask,
        next_wire_mask,
        size_x,
        size_y,
        alpha_w=0.5,
        alpha_s=0.5,
        problem_id=0,
    ):
        """Construct state vector with CarePlace preference weights appended."""
        return np.concatenate((
                    np.array([place_idx]),
                    old_canvas.flatten(),
                    new_canvas.flatten(),
                    structural_mask.flatten(),
                    position_mask.flatten(),
                    wire_mask.flatten(),
                    next_structural_mask.flatten(),
                    next_position_mask.flatten(),
                    next_wire_mask.flatten(),
                    np.array([size_x / self.grid, size_y / self.grid]),
                    np.array([alpha_w, alpha_s]),
                    np.array([problem_id]),
                ), axis=0
                )

    def state2canvas(self, state, new=True):
        if len(state.shape) == 1:
            if new:
                return state[1 + self.grid * self.grid: 1 + self.grid * self.grid * 2].reshape(self.grid, self.grid)
            else:
                return state[1: 1 + self.grid * self.grid].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            if new:
                return state[:, 1 + self.grid * self.grid: 1 + self.grid * self.grid * 2].reshape(-1, self.grid, self.grid)
            else:
                return state[:, 1: 1 + self.grid * self.grid].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2structural_mask(self, state, next_next_macro=False):
        """Structural mask (distance-to-boundary). Replaces old regular_mask."""
        if len(state.shape) == 1:
            if next_next_macro:
                return state[1 + self.grid * self.grid * 5: 1 + self.grid * self.grid * 6].reshape(self.grid, self.grid)
            else:
                return state[1 + self.grid * self.grid * 2: 1 + self.grid * self.grid * 3].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            if next_next_macro:
                return state[:, 1 + self.grid * self.grid * 5: 1 + self.grid * self.grid * 6].reshape(-1, self.grid, self.grid)
            else:
                return state[:, 1 + self.grid * self.grid * 2: 1 + self.grid * self.grid * 3].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2position_mask(self, state, next_next_macro=False):
        if len(state.shape) == 1:
            if next_next_macro:
                return state[1 + self.grid * self.grid * 6: 1 + self.grid * self.grid * 7].reshape(self.grid, self.grid)
            else:
                return state[1 + self.grid * self.grid * 3: 1 + self.grid * self.grid * 4].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            if next_next_macro:
                return state[:, 1 + self.grid * self.grid * 6: 1 + self.grid * self.grid * 7].reshape(-1, self.grid, self.grid)
            else:
                return state[:, 1 + self.grid * self.grid * 3: 1 + self.grid * self.grid * 4].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2wire_mask(self, state, next_next_macro=False):
        if len(state.shape) == 1:
            if next_next_macro:
                return state[1 + self.grid * self.grid * 7: 1 + self.grid * self.grid * 8].reshape(self.grid, self.grid)
            else:
                return state[1 + self.grid * self.grid * 4: 1 + self.grid * self.grid * 5].reshape(self.grid, self.grid)
        elif len(state.shape) == 2:
            if next_next_macro:
                return state[:, 1 + self.grid * self.grid * 7: 1 + self.grid * self.grid * 8].reshape(-1, self.grid, self.grid)
            else:
                return state[:, 1 + self.grid * self.grid * 4: 1 + self.grid * self.grid * 5].reshape(-1, self.grid, self.grid)
        else:
            raise NotImplementedError

    def state2preference(self, state):
        """Extract preference weights [alpha_w, alpha_s] from state vector."""
        if len(state.shape) == 1:
            return state[-3], state[-2]
        elif len(state.shape) == 2:
            return state[:, -3], state[:, -2]
        else:
            raise NotImplementedError

    def state2problem_id(self, state):
        """Extract problem ID from state vector."""
        if len(state.shape) == 1:
            return int(state[-1])
        elif len(state.shape) == 2:
            return state[:, -1].long()
        else:
            raise NotImplementedError

