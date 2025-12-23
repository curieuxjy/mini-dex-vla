"""
DexMachina용 VLA Diffusion Policy Model

양손 Allegro Hand 조작을 위한 확장된 모델
- State dim: 410
- Action dim: 44
- 더 큰 hidden dimensions
"""

import torch
import torch.nn as nn
from .encoders import (
    ImageEncoderTinyCNN,
    ImageEncoderLarger,
    TextEncoderTinyGRU,
    StateEncoderMLP,
    BimanualStateEncoderMLP,
)
from .fusion import FusionMLP
from .diffusion_head import DiffusionConfig, DiffusionPolicyHead, ActionDenoiseModel


class LargerActionDenoiseModel(nn.Module):
    """
    더 큰 action space를 위한 확장된 denoising model
    """
    def __init__(self, cfg: DiffusionConfig, time_emb_dim=64, hidden_dim=512):
        super().__init__()
        from .diffusion_head import SinusoidalTimeEmbedding

        self.cfg = cfg
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = cfg.action_dim + time_emb_dim + cfg.cond_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, cfg.action_dim),
        )

    def forward(self, x_t, t, cond):
        t_emb = self.time_emb(t)
        x = torch.cat([x_t, t_emb, cond], dim=-1)
        eps_pred = self.net(x)
        return eps_pred


class LargerDiffusionPolicyHead(DiffusionPolicyHead):
    """
    더 큰 action space를 위한 확장된 Diffusion Policy Head
    """
    def __init__(self, cfg: DiffusionConfig, time_emb_dim=64, hidden_dim=512):
        # 부모 클래스의 __init__을 호출하지 않고 직접 초기화
        nn.Module.__init__(self)

        self.cfg = cfg
        self.denoise_model = LargerActionDenoiseModel(
            cfg,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim
        )

        from .diffusion_head import make_beta_schedule
        betas, alphas, alpha_bar = make_beta_schedule(cfg)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)


class VLADexMachinaPolicy(nn.Module):
    """
    DexMachina 환경을 위한 VLA Diffusion Policy

    기본 설정:
    - state_dim: 410
    - action_dim: 44
    - d_model: 256
    - diffusion_T: 32
    """
    def __init__(
        self,
        vocab_size: int,
        state_dim: int = 410,
        action_dim: int = 44,
        d_model: int = 256,
        diffusion_T: int = 32,
        use_larger_encoder: bool = True,
        state_hidden_dims: list = None,
        diffusion_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # Image encoder
        if use_larger_encoder:
            self.img_encoder = ImageEncoderLarger(d_model=d_model)
        else:
            self.img_encoder = ImageEncoderTinyCNN(d_model=d_model)

        # Text encoder
        self.txt_encoder = TextEncoderTinyGRU(
            vocab_size=vocab_size,
            d_word=64,
            d_model=d_model
        )

        # State encoder (BimanualStateEncoderMLP for large state)
        if state_hidden_dims is None:
            state_hidden_dims = [512, 256]

        self.state_encoder = BimanualStateEncoderMLP(
            state_dim=state_dim,
            d_model=d_model,
            hidden_dims=state_hidden_dims,
            dropout=dropout,
        )

        # Fusion
        self.fusion = FusionMLP(d_model=d_model)

        # Diffusion head
        cfg = DiffusionConfig(
            T=diffusion_T,
            action_dim=action_dim,
            cond_dim=d_model,
        )

        self.diffusion_head = LargerDiffusionPolicyHead(
            cfg,
            time_emb_dim=64,
            hidden_dim=diffusion_hidden_dim,
        )

    def encode_obs(self, image, text_tokens, state):
        """
        Encode observations into fused context

        Args:
            image: (B, 3, H, W)
            text_tokens: (B, T_text)
            state: (B, state_dim)

        Returns:
            fused_context: (B, d_model)
        """
        img_token = self.img_encoder(image)
        txt_token = self.txt_encoder(text_tokens)
        state_token = self.state_encoder(state)
        fused_context = self.fusion(img_token, txt_token, state_token)
        return fused_context

    def loss(self, image, text_tokens, state, actions):
        """
        Compute diffusion loss

        Args:
            image: (B, 3, H, W)
            text_tokens: (B, T_text)
            state: (B, state_dim)
            actions: (B, action_dim)

        Returns:
            loss: scalar
        """
        cond = self.encode_obs(image, text_tokens, state)
        return self.diffusion_head.loss(actions, cond)

    def act(self, image, text_tokens, state):
        """
        Generate action via diffusion sampling

        Args:
            image: (B, 3, H, W)
            text_tokens: (B, T_text)
            state: (B, state_dim)

        Returns:
            actions: (B, action_dim)
        """
        cond = self.encode_obs(image, text_tokens, state)
        actions = self.diffusion_head.sample(cond)
        return actions

    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dexmachina_model(
    vocab_size: int,
    state_dim: int = 410,
    action_dim: int = 44,
    model_size: str = "base",
) -> VLADexMachinaPolicy:
    """
    Factory function to create DexMachina VLA model

    Args:
        vocab_size: Vocabulary size for text encoder
        state_dim: State dimension (default: 410)
        action_dim: Action dimension (default: 44)
        model_size: "small", "base", or "large"

    Returns:
        VLADexMachinaPolicy model
    """
    configs = {
        "small": {
            "d_model": 128,
            "diffusion_T": 16,
            "state_hidden_dims": [256, 128],
            "diffusion_hidden_dim": 256,
            "use_larger_encoder": False,
        },
        "base": {
            "d_model": 256,
            "diffusion_T": 32,
            "state_hidden_dims": [512, 256],
            "diffusion_hidden_dim": 512,
            "use_larger_encoder": True,
        },
        "large": {
            "d_model": 512,
            "diffusion_T": 50,
            "state_hidden_dims": [1024, 512, 256],
            "diffusion_hidden_dim": 1024,
            "use_larger_encoder": True,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from {list(configs.keys())}")

    cfg = configs[model_size]

    return VLADexMachinaPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        **cfg,
    )
