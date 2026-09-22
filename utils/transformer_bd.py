import os
import numpy as np

import config


class HexapodTransformerModel:
    """Torch model used for latent-based behavior descriptors."""

    def __init__(
        self,
        state_dim=73,
        d_model=64,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=2000,
    ):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(state_dim, d_model)
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
                self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fitness_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

            def forward(self, states):
                x = self.input_proj(states)
                batch_size, seq_len, _ = x.shape
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                x = x + self.pos_embedding[:, :seq_len + 1]
                x = self.transformer(x)
                cls_output = x[:, 0]
                fitness_pred = self.fitness_head(cls_output).squeeze(-1)
                return fitness_pred, cls_output

        self.model = _Model()


class TransformerBDExtractor:
    """Loads a trained transformer and exposes latent-vector BD extraction."""

    def __init__(self):
        import torch
        from multiprocessing import current_process

        cfg = config.get_transformer_bd_config('transformer_latent')
        self.latent_indices = cfg.get("latent_indices", [0, 1])
        device_pref = str(cfg.get("device", "auto")).lower()
        in_main_process = current_process().name == "MainProcess"

        # Forked multiprocessing workers cannot safely initialize CUDA.
        if device_pref == "cpu":
            resolved_device = "cpu"
        elif device_pref == "cuda":
            resolved_device = "cuda" if torch.cuda.is_available() and in_main_process else "cpu"
        else:  # auto
            resolved_device = "cuda" if torch.cuda.is_available() and in_main_process else "cpu"

        self.device = torch.device(resolved_device)

        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError("TRANSFORMER_BD_CONFIG['model_path'] is required for BD_MODE='transformer_latent'.")

        self.model_path = model_path
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.join(config.PROJECT_ROOT, self.model_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Transformer model not found: {self.model_path}. "
                "Train and save a checkpoint, then set TRANSFORMER_BD_CONFIG['model_path']."
            )

        # Ensure full checkpoint loading (avoid PyTorch 2.6+ weights_only=True default)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # If the checkpoint contains a saved config for the model, prefer those
        # parameters when constructing the model to avoid shape mismatches.
        ckpt_cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
        # some checkpoints store state_dim at top-level
        if isinstance(checkpoint, dict) and 'state_dim' in checkpoint:
            ckpt_cfg['state_dim'] = checkpoint['state_dim']

        model_wrapper = HexapodTransformerModel(
            state_dim=ckpt_cfg.get('state_dim', cfg.get('state_dim', 73)),
            d_model=ckpt_cfg.get('d_model', cfg.get('d_model', 64)),
            nhead=ckpt_cfg.get('nhead', cfg.get('nhead', 4)),
            num_layers=ckpt_cfg.get('num_layers', cfg.get('num_layers', 4)),
            dim_feedforward=ckpt_cfg.get('dim_feedforward', cfg.get('dim_feedforward', 256)),
            dropout=ckpt_cfg.get('dropout', cfg.get('dropout', 0.1)),
            max_seq_len=ckpt_cfg.get('max_seq_len', cfg.get('max_seq_len', 2000)),
        )
        self.model = model_wrapper.model.to(self.device)

        if isinstance(checkpoint, dict) and ("state_dict" in checkpoint or "model_state_dict" in checkpoint):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()
        # Attempt to load a PCA projector located next to the model checkpoint
        # or from a user-specified path in the config under 'pca_path'. If found,
        # keep it and apply it to the full CLS latent in compute_bd().
        pca_path = cfg.get('pca_path', None)
        if not pca_path:
            # default: look for transformer_pca.pkl in the checkpoint directory
            model_dir = os.path.dirname(self.model_path)
            default_pca = os.path.join(model_dir, 'transformer_pca.pkl')
            pca_path = default_pca if os.path.exists(default_pca) else None

        self.pca = None
        if pca_path:
            try:
                import pickle
                with open(pca_path, 'rb') as pf:
                    self.pca = pickle.load(pf)
            except Exception:
                # if PCA fails to load, silently ignore and fall back to raw latent indices
                self.pca = None

    def compute_bd(self, state_sequence: np.ndarray) -> np.ndarray:
        """Return selected latent coordinates for one trajectory.

        Args:
            state_sequence: Array of shape (T, state_dim)
        """
        import torch


        if state_sequence.ndim != 2:
            raise ValueError(f"Expected state_sequence shape (T, D), got {state_sequence.shape}")

        seq_T, seq_D = state_sequence.shape

        # determine expected input feature dimension from the model's input projection
        try:
            expected_D = int(self.model.input_proj.in_features)
        except Exception:
            expected_D = None

        # If the simulator provides a larger state vector (e.g. 73) but the
        # transformer was trained on a smaller subset (e.g. 31), try a sensible
        # heuristic: extract the contiguous block [torques(18), contacts(6),
        # base_pos(3), base_orn(4)] which lives before the final base velocities
        # (base_lin_vel + base_ang_vel = 6). This yields 31 dims and matches the
        # training schema used by the transformer training notebook.
        if expected_D is not None and seq_D != expected_D and seq_D >= expected_D + 6:
            # apply heuristic slice: start = seq_D - (expected_D + 6), end = seq_D - 6
            start = seq_D - (expected_D + 6)
            end = seq_D - 6
            if end - start == expected_D:
                state_sequence = state_sequence[:, start:end]
                seq_D = state_sequence.shape[1]

        with torch.no_grad():
            states_t = torch.tensor(state_sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, latent = self.model(states_t)

        latent_np = latent.squeeze(0).detach().cpu().numpy()

        # If a PCA projector is available, apply it to the full CLS latent and
        # return the projected coordinates.
        if self.pca is not None:
            try:
                proj = self.pca.transform(latent_np.reshape(1, -1))[0]
                return proj
            except Exception:
                # Fall back to returning selected latent indices if PCA fails
                pass

        if np.max(self.latent_indices) >= latent_np.shape[0]:
            raise ValueError(
                f"latent_indices {self.latent_indices} out of bounds for latent size {latent_np.shape[0]}"
            )

        return latent_np[self.latent_indices]

    def compute_latent(self, state_sequence: np.ndarray) -> np.ndarray:
        """Return the full CLS latent vector for one trajectory."""
        import torch

        if state_sequence.ndim != 2:
            raise ValueError(f"Expected state_sequence shape (T, D), got {state_sequence.shape}")

        with torch.no_grad():
            states_t = torch.tensor(state_sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, latent = self.model(states_t)

        return latent.squeeze(0).detach().cpu().numpy()
