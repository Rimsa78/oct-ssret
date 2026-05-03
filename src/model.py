"""OCT-SSRet: pretrained CNN + bottleneck (Mamba-style selective SSM, optional
Transformer block, or none) for 4-class retinal disease classification on
OCT B-scans.

This is a deliberate port of the paper-2 (DC-SSRet) hybrid CNN-Mamba
architecture from fundus to OCT. The OCT modality differs from fundus in:
  - single-channel input (we replicate to 3-channel for ImageNet-pretrained features)
  - high aspect ratio (typical 768x496) handled by letterbox-resize at the data layer
  - lower texture diversity (mostly grayscale layered structure)
  - 4-class disease label (not ordinal severity), so we drop the CORAL head
    and use a single class-balanced focal-CE head.

The architectural hypothesis under test: a small bidirectional selective-SSM
bottleneck inserted on the post-CNN token sequence improves cross-cohort
zero-shot transfer, mirroring the paper-2 finding on fundus DR.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

from ssm_block import BiSelectiveSSMBlock, MLPBottleneckBlock


class OCTSSRet(nn.Module):
    def __init__(self,
                 num_classes: int = 4,
                 backbone: str = "resnet18",
                 use_ssm: bool = True,
                 ssm_layers: int = 2,
                 ssm_state: int = 16,
                 ssm_bidirectional: bool = True,
                 use_transformer: bool = False,
                 transformer_layers: int = 2,
                 transformer_heads: int = 8,
                 use_mlp: bool = False,
                 mlp_layers: int = 2,
                 mlp_expand: int = 4,
                 dropout: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        self.use_ssm = use_ssm
        self.use_transformer = use_transformer
        self.use_mlp = use_mlp

        if backbone == "resnet18":
            w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = resnet18(weights=w); dim = 512
            self.backbone = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4,
            )
        elif backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = resnet50(weights=w); dim = 2048
            self.backbone = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4,
            )
        elif backbone == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = efficientnet_b0(weights=w); dim = 1280
            self.backbone = net.features
        elif backbone == "swin_v2_t":
            from torchvision.models import swin_v2_t, Swin_V2_T_Weights
            w = Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
            net = swin_v2_t(weights=w); dim = 768
            # net.children() = features, norm, permute, avgpool, flatten, head
            # We want features only, then permute (B,H,W,C) -> (B,C,H,W) for our pipeline
            self._swin_features = net.features
            self._swin_norm = net.norm
            self.backbone = "_swin"  # special-cased in forward
        elif backbone == "vit_b_16":
            # Vision-Transformer B/16 returns (B, N+1, 768) including CLS; we drop CLS
            # and reshape into a feature map for the same downstream pipeline.
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            w = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self._vit = vit_b_16(weights=w); dim = 768
            self.backbone = None  # special-cased in forward
        else:
            raise ValueError(backbone)
        self.backbone_name = backbone
        self.final_dim = dim

        if use_ssm:
            self.ssm_blocks = nn.ModuleList([
                BiSelectiveSSMBlock(d_model=dim, d_state=ssm_state, expand=2,
                                    dropout=0.1, bidirectional=ssm_bidirectional)
                for _ in range(ssm_layers)
            ])
        if use_mlp:
            self.mlp_blocks = nn.ModuleList([
                MLPBottleneckBlock(d_model=dim, expand=mlp_expand, dropout=0.1)
                for _ in range(mlp_layers)
            ])
        if use_transformer:
            # Standard Transformer-encoder bottleneck: same drop-in role as the
            # Mamba block but quadratic-in-tokens. For head-to-head comparison.
            enc_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=transformer_heads, dim_feedforward=4 * dim,
                dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(dim, num_classes)
        nn.init.normal_(self.cls_head.weight, std=0.02)
        nn.init.zeros_(self.cls_head.bias)

    def forward(self, x):
        if self.backbone_name == "swin_v2_t":
            f = self._swin_features(x)              # (B, H, W, C) for Swin-V2 (channels-last)
            f = self._swin_norm(f)
            feat_map = f.permute(0, 3, 1, 2)        # (B, C, H, W)
            B, D, Hf, Wf = feat_map.shape
            tokens = feat_map.flatten(2).transpose(1, 2)
        elif self.backbone_name == "vit_b_16":
            # ViT path: ResNet-pre-resize to 224, run patch embed + encoder
            if x.shape[-1] != 224:
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            tokens = self._vit._process_input(x)            # (B, N, D)
            cls = self._vit.class_token.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = self._vit.encoder(tokens)              # (B, N+1, D)
            tokens = tokens[:, 1:]                          # drop CLS
            B, N, D = tokens.shape
            Hf = Wf = int(N ** 0.5)
            feat_map = tokens.transpose(1, 2).reshape(B, D, Hf, Wf)
        else:
            feat_map = self.backbone(x)                     # (B, dim, h, w)
            B, D, Hf, Wf = feat_map.shape
            tokens = feat_map.flatten(2).transpose(1, 2)    # (B, h*w, D)
        if self.use_ssm:
            for blk in self.ssm_blocks:
                tokens = blk(tokens)
        if self.use_transformer:
            tokens = self.transformer(tokens)
        if self.use_mlp:
            for blk in self.mlp_blocks:
                tokens = blk(tokens)
        pooled = tokens.mean(dim=1)
        pooled = self.norm(pooled)
        logits = self.cls_head(self.dropout(pooled))
        return {
            "logits_cls": logits,
            "feat_pooled": pooled,
            "feat_map": feat_map.permute(0, 2, 3, 1),       # (B, h, w, D) for Grad-CAM
        }


# -----------------------------------------------------------------------------
# Class-balanced focal loss (Cui et al. 2019 + Lin et al. 2017)
# -----------------------------------------------------------------------------

def class_balanced_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                              freq: torch.Tensor, beta: float = 0.999,
                              gamma: float = 2.0) -> torch.Tensor:
    """logits: (B, K) raw, targets: (B,) int, freq: (K,) class counts."""
    K = logits.shape[-1]
    eff_n = 1.0 - torch.pow(beta, freq.float())
    weights = (1.0 - beta) / eff_n
    weights = (weights / weights.sum()) * K                  # mean-1 normalised
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
    log_p_t = log_p.gather(1, targets.view(-1, 1)).squeeze(1)
    focal = (1 - p_t).pow(gamma) * (-log_p_t)
    w = weights[targets]
    return (w * focal).mean()
