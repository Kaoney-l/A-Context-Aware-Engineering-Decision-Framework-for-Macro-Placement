import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MacroInteractionGraph:
    """Builds a macro interaction graph from the netlist hypergraph for GNN encoding."""

    def __init__(self, node_info, node_to_net_dict, net_info, macro_list):
        self.node_info = node_info
        self.node_to_net_dict = node_to_net_dict
        self.net_info = net_info
        self.macro_list = macro_list
        self._build()

    def _build(self):
        n = len(self.macro_list)
        self.node_to_idx = {name: i for i, name in enumerate(self.macro_list)}
        self.idx_to_node = {i: name for i, name in enumerate(self.macro_list)}

        # Build adjacency: edge weight = number of shared nets
        adj = np.zeros((n, n), dtype=np.float32)
        for i, mi in enumerate(self.macro_list):
            for j, mj in enumerate(self.macro_list):
                if i >= j:
                    continue
                shared = len(self.node_to_net_dict[mi] & self.node_to_net_dict[mj])
                if shared > 0:
                    adj[i, j] = shared
                    adj[j, i] = shared

        self.adj = adj
        self.num_nodes = n

        # Node features: [width, height, area, order_id, degree_normalized]
        node_feats = []
        max_area = 0
        max_degree = 0
        for name in self.macro_list:
            w = self.node_info[name]['x']
            h = self.node_info[name]['y']
            area = w * h
            max_area = max(max_area, area)
            deg = len(self.node_to_net_dict[name])
            max_degree = max(max_degree, deg)

        for name in self.macro_list:
            w_norm = self.node_info[name]['x'] / max(w for m in self.macro_list
                                                     for w in [self.node_info[m]['x']])
            h_norm = self.node_info[name]['y'] / max(w for m in self.macro_list
                                                     for w in [self.node_info[m]['y']])
            area_norm = (self.node_info[name]['x'] * self.node_info[name]['y']) / max_area
            order_id = self.node_info[name].get('id', 0) / max(1, n)
            deg_norm = len(self.node_to_net_dict[name]) / max(1, max_degree)
            node_feats.append([w_norm, h_norm, area_norm, order_id, deg_norm])

        self.node_features = np.array(node_feats, dtype=np.float32)

        # Normalized adjacency with self-loops
        deg = adj.sum(axis=1) + 1.0
        deg_inv_sqrt = np.power(deg, -0.5)
        self.adj_norm = adj * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]

    def get_graph_data(self, device='cpu'):
        return {
            'node_features': th.from_numpy(self.node_features).float().to(device),
            'adj_norm': th.from_numpy(self.adj_norm).float().to(device),
            'adj': th.from_numpy(self.adj).float().to(device),
            'num_nodes': self.num_nodes,
            'node_to_idx': self.node_to_idx,
            'idx_to_node': self.idx_to_node,
        }


class GNNEncoder(nn.Module):
    """GNN encoder that produces topology-aware node embeddings via message passing."""

    def __init__(self, in_dim=5, hidden_dim=64, out_dim=32, num_layers=2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.self_weights = nn.ModuleList()
        self.neigh_weights = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_dim if i == 0 else hidden_dim
            out_ch = hidden_dim if i < num_layers - 1 else out_dim
            self.self_weights.append(nn.Linear(in_ch, out_ch, bias=False))
            self.neigh_weights.append(nn.Linear(in_ch, out_ch, bias=False))

    def forward(self, node_features, adj_norm):
        x = node_features
        for l in range(self.num_layers):
            neigh_msg = th.mm(adj_norm, x)
            x = F.relu(self.self_weights[l](x) + self.neigh_weights[l](neigh_msg))
        return x


class TopologicalInteractionDescriptor(nn.Module):
    """Computes explicit neighborhood interaction features (Eq. 4 in paper)."""

    def __init__(self, embed_dim=32, hidden_dim=64, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, gnn_embeds, adj, node_to_idx, macro_name):
        idx = node_to_idx[macro_name]
        h_i = gnn_embeds[idx]

        neighbors = (adj[idx] > 0).nonzero(as_tuple=True)[0]
        if len(neighbors) == 0:
            return th.zeros(self.mlp[-2].out_features, device=gnn_embeds.device)

        nei_embeds = gnn_embeds[neighbors]
        deg_i = len(neighbors)

        interactions = []
        for j in neighbors:
            h_j = gnn_embeds[j]
            deg_j = (adj[j] > 0).sum().float()
            feat = th.cat([
                h_i, h_j,
                th.tensor([deg_i], device=gnn_embeds.device).float(),
                th.tensor([deg_j], device=gnn_embeds.device).float(),
            ])
            interactions.append(self.mlp(feat))

        return th.stack(interactions).mean(dim=0)


class SpatialEncoder(nn.Module):
    """CNN + GAP encoder for spatial state features (Eq. 5a in paper)."""

    def __init__(self, in_channels=4, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, masks):
        x = self.cnn(masks)
        x = self.gap(x)
        return x.view(x.size(0), -1)


class PreferencePredictor(nn.Module):
    """Macro preference prediction module that infers per-macro optimization preferences."""

    def __init__(self, args, gnn_encoder=None):
        super().__init__()
        self.args = args
        self.grid = args.grid
        self.gnn_encoder = gnn_encoder

        embed_dim = args.preference_gnn_out_dim if hasattr(args, 'preference_gnn_out_dim') else 32
        topo_out_dim = args.preference_topo_out_dim if hasattr(args, 'preference_topo_out_dim') else 32
        spatial_dim = args.preference_spatial_dim if hasattr(args, 'preference_spatial_dim') else 64

        if self.gnn_encoder is None:
            self.gnn_encoder = GNNEncoder(
                in_dim=args.preference_gnn_in_dim if hasattr(args, 'preference_gnn_in_dim') else 5,
                hidden_dim=args.preference_gnn_hidden if hasattr(args, 'preference_gnn_hidden') else 64,
                out_dim=embed_dim,
                num_layers=args.preference_gnn_layers if hasattr(args, 'preference_gnn_layers') else 2,
            )

        self.topo_desc = TopologicalInteractionDescriptor(
            embed_dim=embed_dim,
            hidden_dim=args.preference_topo_hidden if hasattr(args, 'preference_topo_hidden') else 64,
            out_dim=topo_out_dim,
        )

        self.spatial_encoder = SpatialEncoder(
            in_channels=4,
            hidden_dim=spatial_dim,
        )

        combined_dim = topo_out_dim + spatial_dim
        self.pred_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

        self._graph_cache = {}

    def build_graph_for_problem(self, problem):
        """Cache GNN embeddings for a problem instance.

        Also builds a mapping from placement order index (place_idx, matching
        macro_to_place sorted by node_id_to_name) to GNN node index, so that
        batch prediction can correctly look up topology embeddings.
        """
        key = problem.benchmark
        if key in self._graph_cache:
            return self._graph_cache[key]

        macro_list = list(problem.macro_pos.keys())
        graph = MacroInteractionGraph(
            node_info=problem.node_info,
            node_to_net_dict=problem.node_to_net_dict,
            net_info=problem.net_info,
            macro_list=macro_list,
        )
        graph_data = graph.get_graph_data(device=next(self.parameters()).device)

        with th.no_grad():
            gnn_embeds = self.gnn_encoder(
                graph_data['node_features'],
                graph_data['adj_norm'],
            )

        # Build place_idx -> GNN node index mapping
        # macro_to_place is sorted by node_id_to_name (see PlaceEnv.set_place_order)
        macro_to_place_sorted = sorted(macro_list, key=lambda x: problem.node_id_to_name.index(x))
        place_idx_to_gnn_idx = {}
        for place_idx, macro_name in enumerate(macro_to_place_sorted):
            gnn_idx = graph_data['node_to_idx'][macro_name]
            place_idx_to_gnn_idx[place_idx] = gnn_idx

        self._graph_cache[key] = {
            'graph': graph,
            'graph_data': graph_data,
            'gnn_embeds': gnn_embeds,
            'place_idx_to_gnn_idx': place_idx_to_gnn_idx,
        }
        return self._graph_cache[key]

    def forward(self, macro_name, view_mask, position_mask, wire_mask, structural_mask,
                graph_cache_entry):
        """
        Predict preference weights [alpha_w, alpha_s] for a macro.

        Args:
            macro_name: name of the current macro
            view_mask, position_mask, wire_mask, structural_mask: current mask tensors
            graph_cache_entry: cached graph data from build_graph_for_problem

        Returns:
            [alpha_w, alpha_s] preference weights (after softmax)
        """
        graph_data = graph_cache_entry['graph_data']
        gnn_embeds = graph_cache_entry['gnn_embeds']

        # Topological interaction representation (Eq. 4)
        topo_feat = self.topo_desc(
            gnn_embeds, graph_data['adj'],
            graph_data['node_to_idx'], macro_name,
        )

        # Spatial encoding (Eq. 5a): stack 4 masks
        if view_mask.dim() == 2:
            masks = th.stack([view_mask, position_mask, wire_mask, structural_mask], dim=0).unsqueeze(0)
        else:
            masks = th.stack([view_mask, position_mask, wire_mask, structural_mask], dim=1)
        spatial_feat = self.spatial_encoder(masks)  # [1, spatial_dim]

        # Combine and predict (Eq. 5b)
        combined = th.cat([topo_feat.unsqueeze(0), spatial_feat], dim=1)
        logits = self.pred_head(combined)
        alpha = F.softmax(logits, dim=1)  # [alpha_w, alpha_s]

        return alpha.squeeze(0)  # [2]

    def predict_from_state_batch(self, state_batch, state_parsing):
        """Predict preferences from a batch of states during PPO update.

        This runs inside the actor's forward pass so gradients flow back
        to the preference predictor through the PPO loss.

        Uses problem_id from state to look up the correct cached GNN embeddings.

        Args:
            state_batch: [B, state_dim] tensor of states
            state_parsing: StateParsing instance

        Returns:
            alpha_w: [B] tensor of wirelength preference weights
            alpha_s: [B] tensor of structural preference weights
        """
        B = state_batch.shape[0]
        G = self.grid
        device = state_batch.device

        # Extract masks from state batch
        view_mask = state_parsing.state2canvas(state_batch, new=True).to(device)
        position_mask = state_parsing.state2position_mask(state_batch, next_next_macro=False).to(device)
        wire_mask = state_parsing.state2wire_mask(state_batch, next_next_macro=False).to(device)
        structural_mask = state_parsing.state2structural_mask(state_batch, next_next_macro=False).to(device)

        # Stack: [B, 4, G, G]
        masks = th.stack([view_mask, position_mask, wire_mask, structural_mask], dim=1)
        spatial_feats = self.spatial_encoder(masks)  # [B, spatial_dim]

        # Get problem IDs to look up cached GNN embeddings
        problem_ids = state_parsing.state2problem_id(state_batch)  # [B]
        place_idx = state_batch[:, 0].long()  # [B]

        # Build per-sample topology features
        # Map problem_id -> benchmark name for cache lookup
        id_to_benchmark = {v: k for k, v in state_parsing.PROBLEM_IDS.items()}
        topo_feats = []

        for b in range(B):
            pid = problem_ids[b].item()
            benchmark = id_to_benchmark.get(pid, None)

            if benchmark is not None and benchmark in self._graph_cache:
                cache_entry = self._graph_cache[benchmark]
                gnn_embeds = cache_entry['gnn_embeds']
                adj = cache_entry['graph_data']['adj']
                # Map place_idx (placement order) to GNN node index
                place_to_gnn = cache_entry.get('place_idx_to_gnn_idx', {})
                raw_pidx = place_idx[b].item()
                gnn_idx = place_to_gnn.get(raw_pidx, min(raw_pidx, gnn_embeds.shape[0] - 1))
            else:
                # Fallback: zero topology features
                topo_feats.append(th.zeros(self.topo_desc.mlp[-2].out_features, device=device))
                continue

            h_i = gnn_embeds[gnn_idx]  # [embed_dim]
            neighbors = (adj[gnn_idx] > 0).nonzero(as_tuple=True)[0]
            if len(neighbors) > 0:
                h_neigh = gnn_embeds[neighbors].mean(dim=0)
            else:
                h_neigh = th.zeros_like(h_i)

            deg_i = len(neighbors)
            deg_neigh_avg = 0.0
            for j in neighbors:
                deg_neigh_avg += (adj[j] > 0).sum().float().item()
            deg_neigh_avg = deg_neigh_avg / max(1, len(neighbors))

            feat = th.cat([
                h_i, h_neigh,
                th.tensor([float(deg_i)], device=device),
                th.tensor([deg_neigh_avg], device=device),
            ])
            topo_feat = self.topo_desc.mlp(feat)
            topo_feats.append(topo_feat)

        topo_feats = th.stack(topo_feats)  # [B, topo_out_dim]

        # Combine and predict (Eq. 5b)
        combined = th.cat([topo_feats, spatial_feats], dim=1)
        logits = self.pred_head(combined)
        alpha = F.softmax(logits, dim=1)  # [B, 2]

        return alpha[:, 0], alpha[:, 1]


class RunningMeanStd:
    """Tracks running mean and standard deviation for online normalization."""

    def __init__(self, epsilon=1e-8, momentum=0.1):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon
        self.momentum = momentum

    def update(self, x):
        x = float(x)
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.var = 0.0
        else:
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.var = (1.0 - 1.0 / self.count) * self.var + delta * delta2 / (self.count - 1 + 1e-10)

    def normalize(self, x):
        return (x - self.mean) / (th.sqrt(th.tensor(self.var)) + self.epsilon)

    def normalize_scalar(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + self.epsilon)


class DynamicGroupCalibration(nn.Module):
    """Dynamic group calibration that normalizes reward components online (Eq. 10 in paper)."""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.wire_stats = RunningMeanStd(epsilon)
        self.structural_stats = RunningMeanStd(epsilon)
        self.epsilon = epsilon

    def update(self, wire_reward, structural_reward):
        self.wire_stats.update(wire_reward)
        self.structural_stats.update(structural_reward)

    def normalize_wire(self, wire_reward):
        return self.wire_stats.normalize_scalar(wire_reward)

    def normalize_structural(self, structural_reward):
        return self.structural_stats.normalize_scalar(structural_reward)

    def calibrate(self, wire_reward, structural_reward, alpha_w, alpha_s):
        """Calibrate and combine rewards using dynamic group normalization.

        Args:
            wire_reward: raw wirelength reward (scalar)
            structural_reward: raw structural reward (scalar)
            alpha_w: predicted wirelength preference weight
            alpha_s: predicted structural preference weight

        Returns:
            calibrated combined reward
        """
        r_wire_norm = self.normalize_wire(wire_reward)
        r_struct_norm = self.normalize_structural(structural_reward)
        return alpha_w * r_wire_norm + alpha_s * r_struct_norm
