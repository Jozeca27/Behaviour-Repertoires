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
        bd_dim=8,
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
                self.shared_norm = nn.LayerNorm(d_model)

                # 8D behavior descriptor head (learned)
                self.bd_head = nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Linear(64, bd_dim),
                )

                # Fitness prediction head consumes BD (not CLS)
                self.fitness_head = nn.Sequential(
                    nn.Linear(bd_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, states):
                x = self.input_proj(states)

                batch_size, seq_len, _ = x.shape

                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

                if seq_len + 1 > self.pos_embedding.shape[1]:
                    raise ValueError(
                        f'Sequence length {seq_len} exceeds max_seq_len {self.pos_embedding.shape[1] - 1}'
                    )

                x = x + self.pos_embedding[:, :seq_len + 1]

                x = self.transformer(x)

                cls_output = x[:, 0]
                cls_output = self.shared_norm(cls_output)

                bd = self.bd_head(cls_output)

                fitness_pred = self.fitness_head(bd).squeeze(-1)

                return fitness_pred, bd

        self.model = _Model()


class TransformerBDExtractor:
    """Loads a trained transformer and exposes latent-vector BD extraction."""

    def __init__(self):
        import torch
        from multiprocessing import current_process

        cfg = config.get_transformer_bd_config('transformer_8d')
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
            raise ValueError("TRANSFORMER_BD_CONFIG['model_path'] is required for BD_MODE='transformer_8d'.")

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
            bd_dim=ckpt_cfg.get('bd_dim', cfg.get('bd_dim', 8)),
        )
        self.model = model_wrapper.model.to(self.device)

        if isinstance(checkpoint, dict) and ("state_dict" in checkpoint or "model_state_dict" in checkpoint):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()
       
    def compute_bd(self, state_sequence: np.ndarray) -> np.ndarray:
        """
        Compute learned 8D transformer behavior descriptor.
        """

        import torch

        if state_sequence.ndim != 2:
            raise ValueError(
                f"Expected state_sequence shape (T, D), "
                f"got {state_sequence.shape}"
            )

        seq_T, seq_D = state_sequence.shape

        try:
            expected_D = int(
                self.model.input_proj.in_features
            )
        except Exception:
            expected_D = None

        # --------------------------------------------------
        # INPUT DIMENSION FIX
        # --------------------------------------------------

        if (
            expected_D is not None
            and seq_D != expected_D
            and seq_D >= expected_D + 6
        ):

            start = seq_D - (expected_D + 6)
            end = seq_D - 6

            if end - start == expected_D:

                state_sequence = (
                    state_sequence[:, start:end]
                )

        with torch.no_grad():

            states_t = torch.tensor(
                state_sequence,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            _, bd = self.model(states_t)

        bd_np = (
            bd.squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        return bd_np.astype(np.float32)
    
    def compute_latent( self, state_sequence: np.ndarray) -> np.ndarray:

        return self.compute_bd(state_sequence)
