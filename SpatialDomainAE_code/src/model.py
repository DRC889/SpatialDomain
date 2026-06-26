"""
SpatialStroke: Dual-graph GAT with attention fusion for spatial domain
identification in ischemic stroke.

Main model — SpatialDomainNet:
    1. Expression → MLP Encoder → h_local
    2. h_local → GAT(spatial_graph) → h_spatial
       h_local → GAT(feature_graph) → h_feature
    3. AttentionFusion(h_spatial, h_feature) → h_fused  (per-spot learned weight)
    4. h_fused → Classifier → domain labels

Key design choices (inspired by SpatialGlue):
    - Dual graph: spatial k-NN + expression-correlation k-NN
    - 1-layer GAT per graph (not stacked) → avoids over-smoothing
    - Attention fusion learns per-spot how much spatial vs feature context to use
    - GAT (not GCN) enables boundary-aware neighbor attention

Ablation models for systematic comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# =============================================================================
# Building blocks
# =============================================================================

class ExpressionEncoder(nn.Module):
    """Gene expression encoder (MLP)."""

    def __init__(self, n_genes: int, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        Wh = self.W(h)
        src, dst = edge_index[0], edge_index[1]
        cat_feat = torch.cat([Wh[src], Wh[dst]], dim=-1)
        e = self.leaky_relu(self.attn(cat_feat).squeeze(-1))
        e_max = torch.zeros(N, device=h.device)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
        e_exp = torch.exp(e - e_max[dst])
        e_sum = torch.zeros(N, device=h.device)
        e_sum.scatter_add_(0, dst, e_exp)
        alpha = e_exp / (e_sum[dst] + 1e-8)
        alpha = self.dropout(alpha)
        out = torch.zeros(N, Wh.size(1), device=h.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, Wh.size(1)),
                         alpha.unsqueeze(-1) * Wh[src])
        return out


class MultiHeadGAT(nn.Module):
    """Multi-head graph attention with residual connection."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        assert out_dim % n_heads == 0
        head_dim = out_dim // n_heads
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_dim, head_dim, dropout)
            for _ in range(n_heads)
        ])
        self.norm = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        head_outs = [head(h, edge_index) for head in self.heads]
        out = torch.cat(head_outs, dim=-1)
        out = F.elu(out + self.residual(h))
        return self.norm(out)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of two embeddings (SpatialGlue-style).
    Learns per-spot weight alpha to combine emb1 and emb2.

    Output: alpha * emb1 + (1 - alpha) * emb2
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_omega = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.u_omega = Parameter(torch.FloatTensor(out_dim, 1))
        nn.init.xavier_uniform_(self.w_omega)
        nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1: torch.Tensor,
                emb2: torch.Tensor) -> tuple:
        # Stack: (N, 2, dim)
        stacked = torch.stack([emb1, emb2], dim=1)
        # Attention scores: (N, 2)
        v = torch.tanh(torch.matmul(stacked, self.w_omega))
        vu = torch.matmul(v, self.u_omega).squeeze(-1)
        alpha = F.softmax(vu, dim=1)
        # Weighted combination: (N, dim)
        combined = (stacked * alpha.unsqueeze(-1)).sum(dim=1)
        return combined, alpha


# =============================================================================
# Main model: SpatialDomainNet (Dual-graph GAT + Attention Fusion)
# =============================================================================

class SpatialDomainNet(nn.Module):
    """
    Dual-graph GAT with attention fusion for spatial domain identification.

    Architecture:
        Expression → MLP Encoder → h_local (hidden_dim)
        h_local → GAT(spatial_graph) → h_spatial
        h_local → GAT(feature_graph) → h_feature
        AttentionFusion(h_spatial, h_feature) → h_fused
        h_fused → Classifier → logits

    Innovation:
        - Dual graph: spatial k-NN (physical proximity) +
                      feature k-NN (expression correlation)
        - Single-layer GAT per graph (avoids over-smoothing)
        - Attention fusion learns per-spot optimal weighting
        - GAT provides boundary-aware neighbor attention
          (unlike GCN which weights all neighbors equally)
    """

    def __init__(self, n_genes: int, n_classes: int, hidden_dim: int = 256,
                 n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        # Expression encoder
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)

        # Single-layer GAT for each graph type
        self.gat_spatial = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.gat_feature = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)

        # Attention-based fusion (spatial vs feature)
        self.attention = AttentionFusion(hidden_dim, hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, expr: torch.Tensor,
                edge_spatial: torch.Tensor,
                edge_feature: torch.Tensor) -> torch.Tensor:
        h = self.encoder(expr)
        h_spatial = self.gat_spatial(h, edge_spatial)
        h_feature = self.gat_feature(h, edge_feature)
        h_fused, _ = self.attention(h_spatial, h_feature)
        return self.classifier(h_fused)

    def forward_with_attention(self, expr: torch.Tensor,
                               edge_spatial: torch.Tensor,
                               edge_feature: torch.Tensor) -> tuple:
        """Forward pass that also returns attention weights for visualization."""
        h = self.encoder(expr)
        h_spatial = self.gat_spatial(h, edge_spatial)
        h_feature = self.gat_feature(h, edge_feature)
        h_fused, alpha = self.attention(h_spatial, h_feature)
        logits = self.classifier(h_fused)
        return logits, alpha

    def get_embeddings(self, expr: torch.Tensor,
                       edge_spatial: torch.Tensor,
                       edge_feature: torch.Tensor) -> torch.Tensor:
        """Get spatial embeddings for downstream analysis."""
        self.eval()
        with torch.no_grad():
            h = self.encoder(expr)
            h_spatial = self.gat_spatial(h, edge_spatial)
            h_feature = self.gat_feature(h, edge_feature)
            h_fused, _ = self.attention(h_spatial, h_feature)
        return h_fused


class SpatialDomainAE(nn.Module):
    """
    Unsupervised variant: Dual-graph GAT Autoencoder.

    Same dual-graph GAT + attention fusion core, but trained with
    reconstruction loss instead of classification loss.
    Learned embeddings are clustered post hoc with Leiden in the benchmark.

    Architecture:
        Expression → Encoder → DualGAT → Fusion → h_fused (latent)
        h_fused → Decoder → reconstructed expression
        Loss: MSE(reconstructed, original)
    """

    def __init__(self, n_genes: int, latent_dim: int = 30,
                 hidden_dim: int = 256, n_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        # Expression encoder
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)

        # Single-layer GAT per graph
        self.gat_spatial = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.gat_feature = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)

        # Attention fusion
        self.attention = AttentionFusion(hidden_dim, hidden_dim)

        # Bottleneck to latent
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, n_genes),
        )

    def encode(self, expr, edge_spatial, edge_feature):
        h = self.encoder(expr)
        h_s = self.gat_spatial(h, edge_spatial)
        h_f = self.gat_feature(h, edge_feature)
        h_fused, alpha = self.attention(h_s, h_f)
        z = self.to_latent(h_fused)
        return z, alpha

    def forward(self, expr, edge_spatial, edge_feature):
        z, alpha = self.encode(expr, edge_spatial, edge_feature)
        recon = self.decoder(z)
        return recon, z, alpha

    def get_embeddings(self, expr, edge_spatial, edge_feature):
        self.eval()
        with torch.no_grad():
            z, alpha = self.encode(expr, edge_spatial, edge_feature)
        return z


# =============================================================================
# Ablation variants
# =============================================================================

class ExprOnlyNet(nn.Module):
    """Ablation: expression MLP only (no graph)."""

    def __init__(self, n_genes: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, 256, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, expr, edge_spatial=None, edge_feature=None):
        # Older analysis scripts used the signature (img, expr, edge).
        if expr is None and edge_spatial is not None:
            expr = edge_spatial
        return self.classifier(self.encoder(expr))


class SpatialGATNet(nn.Module):
    """Ablation: expression + spatial GAT only (no feature graph)."""

    def __init__(self, n_genes: int, n_classes: int, hidden_dim: int = 256,
                 n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)
        self.gat = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, expr, edge_spatial, edge_feature=None):
        h = self.encoder(expr)
        h = self.gat(h, edge_spatial)
        return self.classifier(h)


class FeatureGATNet(nn.Module):
    """Ablation: expression + feature GAT only (no spatial graph)."""

    def __init__(self, n_genes: int, n_classes: int, hidden_dim: int = 256,
                 n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)
        self.gat = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, expr, edge_spatial=None, edge_feature=None):
        h = self.encoder(expr)
        h = self.gat(h, edge_feature)
        return self.classifier(h)


# =============================================================================
# Legacy supervised variants used by exploratory scripts
# =============================================================================

class ExprGraphNet(nn.Module):
    """Expression + spatial GAT classifier used in early coarse analyses."""

    def __init__(self, n_genes: int, n_classes: int, hidden_dim: int = 256,
                 n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)
        self.gat1 = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.gat2 = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, img, expr=None, edge_index=None):
        if expr is None:
            expr = img
        h = self.encoder(expr)
        h = self.gat1(h, edge_index)
        h = self.gat2(h, edge_index)
        return self.classifier(h)


class NoGraphNet(ExprOnlyNet):
    """Backward-compatible name for an expression-only classifier."""

    def forward(self, img, expr=None, edge_index=None):
        if expr is None:
            expr = img
        return super().forward(expr)


class ImageOnlyNet(nn.Module):
    """Classifier using pre-extracted image embeddings only."""

    def __init__(self, n_classes: int, backbone_dim: int = 768,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, img, expr=None, edge_index=None):
        return self.classifier(self.img_proj(img))


class ConcatGraphNet(nn.Module):
    """Classifier that concatenates image and expression features before GAT."""

    def __init__(self, n_genes: int, n_classes: int, backbone_dim: int = 768,
                 hidden_dim: int = 256, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.expr_proj = ExpressionEncoder(n_genes, hidden_dim, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
        )
        self.gat = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, img, expr, edge_index):
        h_img = self.img_proj(img)
        h_expr = self.expr_proj(expr)
        h = self.fuse(torch.cat([h_img, h_expr], dim=-1))
        h = self.gat(h, edge_index)
        return self.classifier(h)


class SpatialCLIPNet(nn.Module):
    """
    Lightweight contrastive image-expression graph classifier.

    This class keeps the older coarse-analysis scripts runnable. The main
    manuscript model is SpatialDomainAE above.
    """

    def __init__(self, n_genes: int, n_classes: int, backbone_dim: int = 768,
                 embed_dim: int = 128, gat_dim: int = 256, n_heads: int = 4,
                 dropout: float = 0.3, modality_dropout: float = 0.0):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.img_proj = nn.Sequential(
            nn.Linear(backbone_dim, gat_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(gat_dim),
            nn.Dropout(dropout),
            nn.Linear(gat_dim, embed_dim),
        )
        self.expr_encoder = nn.Sequential(
            ExpressionEncoder(n_genes, gat_dim, dropout),
            nn.Linear(gat_dim, embed_dim),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * embed_dim, gat_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(gat_dim),
        )
        self.gat = MultiHeadGAT(gat_dim, gat_dim, n_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gat_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def contrastive_loss(self, z_img, z_expr):
        z_img = F.normalize(z_img, dim=-1)
        z_expr = F.normalize(z_expr, dim=-1)
        logits = self.logit_scale.exp() * z_img @ z_expr.T
        target = torch.arange(z_img.size(0), device=z_img.device)
        loss_i = F.cross_entropy(logits, target)
        loss_e = F.cross_entropy(logits.T, target)
        return 0.5 * (loss_i + loss_e)

    def _apply_modality_dropout(self, z_img, z_expr):
        if not self.training or self.modality_dropout <= 0:
            return z_img, z_expr
        keep_img = (torch.rand(z_img.size(0), 1, device=z_img.device)
                    > self.modality_dropout).float()
        keep_expr = (torch.rand(z_expr.size(0), 1, device=z_expr.device)
                     > self.modality_dropout).float()
        return z_img * keep_img, z_expr * keep_expr

    def forward(self, z_img, z_expr, edge_index, mode="multimodal"):
        if mode == "image":
            z_expr = torch.zeros_like(z_expr)
        elif mode == "expression":
            z_img = torch.zeros_like(z_img)
        else:
            z_img, z_expr = self._apply_modality_dropout(z_img, z_expr)

        h = self.fuse(torch.cat([z_img, z_expr], dim=-1))
        h = self.gat(h, edge_index)
        return self.classifier(h)


class DualGCNNet(nn.Module):
    """Ablation: dual-graph GCN (not GAT) + attention fusion.
    Uses simple graph convolution (SpatialGlue-style) instead of GAT."""

    def __init__(self, n_genes: int, n_classes: int, hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)
        # GCN: just a linear layer, graph aggregation done via adj @ h
        self.gcn_spatial_w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gcn_feature_w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.attention = AttentionFusion(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def _gcn_aggregate(self, h, edge_index, W, norm):
        """Simple GCN aggregation using edge_index."""
        N = h.size(0)
        Wh = W(h)
        src, dst = edge_index[0], edge_index[1]
        # Mean aggregation
        out = torch.zeros_like(Wh)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, Wh.size(1)), Wh[src])
        deg = torch.zeros(N, device=h.device)
        deg.scatter_add_(0, dst, torch.ones(len(src), device=h.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        out = out / deg
        out = F.elu(out + Wh)  # residual
        return norm(out)

    def forward(self, expr, edge_spatial, edge_feature):
        h = self.encoder(expr)
        h_s = self._gcn_aggregate(h, edge_spatial, self.gcn_spatial_w, self.norm_s)
        h_f = self._gcn_aggregate(h, edge_feature, self.gcn_feature_w, self.norm_f)
        h_fused, _ = self.attention(h_s, h_f)
        return self.classifier(h_fused)


# =============================================================================
# Autoencoder ablation variants (for unsupervised embedding comparison)
# =============================================================================

class _AEDecoder(nn.Module):
    """Shared decoder for all AE variants."""

    def __init__(self, latent_dim: int, hidden_dim: int, n_genes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, n_genes),
        )

    def forward(self, z):
        return self.net(z)


class SpatialGATAE(nn.Module):
    """Ablation AE: spatial GAT only (no feature graph)."""

    def __init__(self, n_genes: int, latent_dim: int = 30,
                 hidden_dim: int = 256, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)
        self.gat = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = _AEDecoder(latent_dim, hidden_dim, n_genes)

    def encode(self, expr, edge_spatial, edge_feature=None):
        h = self.encoder(expr)
        h = self.gat(h, edge_spatial)
        return self.to_latent(h)

    def forward(self, expr, edge_spatial, edge_feature=None):
        z = self.encode(expr, edge_spatial)
        return self.decoder(z), z, None

    def get_embeddings(self, expr, edge_spatial, edge_feature=None):
        self.eval()
        with torch.no_grad():
            return self.encode(expr, edge_spatial)


class FeatureGATAE(nn.Module):
    """Ablation AE: feature GAT only (no spatial graph)."""

    def __init__(self, n_genes: int, latent_dim: int = 30,
                 hidden_dim: int = 256, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)
        self.gat = MultiHeadGAT(hidden_dim, hidden_dim, n_heads, dropout)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = _AEDecoder(latent_dim, hidden_dim, n_genes)

    def encode(self, expr, edge_spatial=None, edge_feature=None):
        h = self.encoder(expr)
        # Use edge_feature if provided, otherwise edge_spatial as fallback
        edge = edge_feature if edge_feature is not None else edge_spatial
        h = self.gat(h, edge)
        return self.to_latent(h)

    def forward(self, expr, edge_spatial=None, edge_feature=None):
        z = self.encode(expr, edge_spatial, edge_feature)
        return self.decoder(z), z, None

    def get_embeddings(self, expr, edge_spatial=None, edge_feature=None):
        self.eval()
        with torch.no_grad():
            return self.encode(expr, edge_spatial, edge_feature)


class DualGCNAE(nn.Module):
    """Ablation AE: dual-graph GCN (not GAT) + attention fusion autoencoder."""

    def __init__(self, n_genes: int, latent_dim: int = 30,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)
        self.gcn_spatial_w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gcn_feature_w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.attention = AttentionFusion(hidden_dim, hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = _AEDecoder(latent_dim, hidden_dim, n_genes)

    def _gcn_aggregate(self, h, edge_index, W, norm):
        N = h.size(0)
        Wh = W(h)
        src, dst = edge_index[0], edge_index[1]
        out = torch.zeros_like(Wh)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, Wh.size(1)), Wh[src])
        deg = torch.zeros(N, device=h.device)
        deg.scatter_add_(0, dst, torch.ones(len(src), device=h.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        out = out / deg
        out = F.elu(out + Wh)
        return norm(out)

    def encode(self, expr, edge_spatial, edge_feature):
        h = self.encoder(expr)
        h_s = self._gcn_aggregate(h, edge_spatial, self.gcn_spatial_w, self.norm_s)
        h_f = self._gcn_aggregate(h, edge_feature, self.gcn_feature_w, self.norm_f)
        h_fused, alpha = self.attention(h_s, h_f)
        return self.to_latent(h_fused)

    def forward(self, expr, edge_spatial, edge_feature):
        z = self.encode(expr, edge_spatial, edge_feature)
        return self.decoder(z), z, None

    def get_embeddings(self, expr, edge_spatial, edge_feature):
        self.eval()
        with torch.no_grad():
            return self.encode(expr, edge_spatial, edge_feature)


class ExprOnlyAE(nn.Module):
    """Ablation AE: MLP only, no graph."""

    def __init__(self, n_genes: int, latent_dim: int = 30,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.encoder = ExpressionEncoder(n_genes, hidden_dim, dropout)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder = _AEDecoder(latent_dim, hidden_dim, n_genes)

    def encode(self, expr, edge_spatial=None, edge_feature=None):
        h = self.encoder(expr)
        return self.to_latent(h)

    def forward(self, expr, edge_spatial=None, edge_feature=None):
        z = self.encode(expr)
        return self.decoder(z), z, None

    def get_embeddings(self, expr, edge_spatial=None, edge_feature=None):
        self.eval()
        with torch.no_grad():
            return self.encode(expr)
