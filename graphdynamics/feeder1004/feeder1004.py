"""
feeder1004  =  V1003Super MINUS the supernode attention  (created 2026-08-26)

ABLATION. Copied verbatim from model_v1003super.py (md5 86577e27153e6124
e089576212bf1f37) and changed in exactly three places, all commented in situ
and dated 2026-08-26:

  1. attn_1 / attn_2 / attn_3 are not constructed  (~line 679)
  2. n_attn is hardcoded to 0 for the print          (~line 699)
  3. the three attn calls in the recurrent loop are commented out (~line 761)

Nothing else differs. The class is still named RecurrentBrainNetV7 so an
existing trainer can import it by changing only the module name. The
SupernodeLinearAttention class remains defined but is never instantiated.

WHY. v1003 beats the UNet baseline by 7.63% (val 0.043643 vs 0.047250, three
seeds each, non-overlapping). That number credits two mechanisms at once: the
recurrent sparse graph, and the supernode attention on top of it. This variant
isolates the graph. Attention is 12,864 params -- 0.017% of the model -- so if
it carries a large share of the win, that is worth knowing.

Below: the v1003 docstring as inherited, describing everything this file
still shares with it.

  #13 attn_1 was skipped on iteration 0 (ran 4x vs 5x for attn_2/attn_3);
      a3 now starts as zeros so every iteration takes the same branch.
      (Moot here -- there is no attn_1 -- but the zeros init is retained
      because a3 is still read before it is written on the first pass.)

  bottleneck  encoder.mid (14,164,992 params) is removed. In v1001 its output
      was discarded via torch.zeros_like, so it and encoder.down4 trained on
      nothing. down4's 8x8 output now enters the graph as a sixth collector
      level, s5 (64 nodes). The decoder's bottleneck slot is filled by t5.

  readout  t4 (16x16) and t5 (8x8) are learned nn.Linear projections over ALL
      n3 nodes rather than index slices: a 64- and a 256-node slice would have
      carried 0.89% and 3.55% of the graph state into the decoder's two most
      semantic inputs. t5 is read from recurrent pass 3, t4 from pass 4, the
      other four levels from the final pass. proj_t4 and proj_t5 are placed in
      the 1x LR group, not the 10x brain group.

  decoder  dec4 and dec3 each gained a third ResBlock(512,512), matching the
      UNet baseline's depth at those levels, and their first block was narrowed
      1536 -> 1024 because the zeroed s3/s4 channels are no longer concatenated
      at all. The two decoders are now identical, 46,829,699 params each --
      genuinely, not "identical after excluding inert weights".

  #8  the observed-region loss term is applied to `raw`, not `recon`; forward
      returns (recons, raw). On recon the term reduced to m*(x_obs - x) and had
      exactly zero gradient.

REVERTED 2026-08-16, both tried and withdrawn the same day:
  #14 clustering supernodes on the FEEDER graph. The feeder's extra edges were
      built FROM the parent's partition, so the parent partition is the correct
      one (ARI 1.0000 against a fresh coarsen; the feeder-clustered variant was
      0.6845). Back to graph_brain_mild_v146_feeder_supernode.npz.
  #15 replacing the twin path with coordinate-queried linear cross-attention.
      Linear attention sums all nodes into a head_dim x head_dim state before
      any query is applied, so with head_dim = C//n_heads = 8 the entire
      7203x32 = 230,496-number graph state reached the decoder through 288
      numbers (measured Jacobian rank ~285). The premise was also wrong: a3 is
      ordered by global tensor index, so index-adjacent nodes are siblings in
      the tensor hierarchy and the grid layout reflects the construction.
      Back to slice-and-reshape for t3/t2/t1/t0.

Original V1001Super header follows.

RecurrentBrainNet V1001Super — delayed feedback + classless supernode linear attention.

Combines v7 feeder loop3's delay (a2 sees old a1, then a1 updates) with
v10 supernode's SupernodeLinearAttention (shared QKV, no size classes, 2x init).
Norm ordering from v10 supernode (norm before attention).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# Sparse graph convolution
# ═══════════════════════════════════════════════════════════════════

class MultiChannelSparseLinear(nn.Module):
    def __init__(self, in_nodes, out_nodes, mask):
        super().__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        idx = mask.nonzero(as_tuple=False)
        self.register_buffer('row_idx', idx[:, 0].long())
        self.register_buffer('col_idx', idx[:, 1].long())
        n_edges = idx.shape[0]
        self.weight = nn.Parameter(torch.empty(n_edges))
        nn.init.uniform_(self.weight,
                         -1.0 / math.sqrt(in_nodes),
                          1.0 / math.sqrt(in_nodes))
        # ---- REMOVED 2026-08-16: the per-node bias ----
        # It added one scalar per output node, broadcast across all C channels.
        # Every one of the five sparse layers is immediately followed by a
        # LayerNorm over the channel dim (ln2, ln3, ln_fb, ln_fb1), and
        # LayerNorm(x + c*1) == LayerNorm(x) exactly, so the bias was cancelled
        # before it could affect anything. Measured mean |grad| 1e-10 to 4e-06
        # against 1e-04 for the next smallest parameter in the model -- that was
        # LayerNorm-backward float residue, not signal. 28,812 parameters that
        # could not influence the output.
        # was: self.bias = nn.Parameter(torch.zeros(out_nodes))

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            B, N, C = x.shape
            x_e = x[:, self.col_idx, :]
            w = self.weight.float().unsqueeze(0).unsqueeze(-1)
            contrib = x_e * w
            out = torch.zeros(B, self.out_nodes, C,
                              device=x.device, dtype=torch.float32)
            row_exp = self.row_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, C)
            out.scatter_add_(1, row_exp, contrib)
            # was: return out + self.bias.float().unsqueeze(0).unsqueeze(-1)
            return out


# ═══════════════════════════════════════════════════════════════════
# UNet building blocks
# ═══════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip_proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return self.skip_proj(x) + h


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(4, 64, 3, 1, 1)

        self.enc0 = nn.ModuleList([ResBlock(64, 64), ResBlock(64, 64)])
        self.down0 = Downsample(64)

        self.enc1 = nn.ModuleList([ResBlock(64, 128), ResBlock(128, 128)])
        self.down1 = Downsample(128)

        self.enc2 = nn.ModuleList([ResBlock(128, 256), ResBlock(256, 256)])
        self.down2 = Downsample(256)

        self.enc3 = nn.ModuleList([ResBlock(256, 512), ResBlock(512, 512)])
        self.down3 = Downsample(512)

        self.enc4 = nn.ModuleList([ResBlock(512, 512), ResBlock(512, 512)])
        self.down4 = Downsample(512)

        # ---- CHANGED 2026-08-15 (v1003: send down4 to the graph, eliminate only mid) ----
        # `mid` (3x ResBlock(512,512), 14,164,992 params) WAS the UNet middle - the
        # thing the graph replaces - so it is removed outright rather than computed
        # and discarded. down4's 8x8 output now feeds the collector as a 6th level,
        # so down4 trains instead of dying alongside mid.
        # self.mid = nn.ModuleList([ResBlock(512, 512), ResBlock(512, 512), ResBlock(512, 512)])

    def forward(self, x):
        h = self.stem(x)

        for block in self.enc0:
            h = block(h)
        s0 = h
        h = self.down0(h)

        for block in self.enc1:
            h = block(h)
        s1 = h
        h = self.down1(h)

        for block in self.enc2:
            h = block(h)
        s2 = h
        h = self.down2(h)

        for block in self.enc3:
            h = block(h)
        s3 = h
        h = self.down3(h)

        for block in self.enc4:
            h = block(h)
        s4 = h
        s5 = self.down4(h)          # 8x8, formerly the input to `mid`

        # ---- CHANGED 2026-08-15/16 (v1003) ----
        # The UNet's bottleneck -- three ResBlocks at 8x8, 14,164,992 params,
        # named `self.mid` in this codebase and living inside the Encoder class
        # -- is gone. The graph replaces it. down4's 8x8 output is now just the
        # deepest encoder feature, s5, and goes to the collector like any other.
        # was: for block in self.mid: h = block(h)
        #      return h, [s0, s1, s2, s3, s4]
        # was: return s5, [s0, s1, s2, s3, s4, s5]   <- returned s5 twice, and the
        #      caller bound the first copy to a variable called `bottleneck`,
        #      which this model does not have.
        return [s0, s1, s2, s3, s4, s5]


# ═══════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ---- CHANGED 2026-08-16 (v1003) ----
        # (a) third ResBlock at dec4 and dec3, matching the UNet baseline's depth
        #     at those levels (model_unetonly.py:114,118). Without it the baseline
        #     had 9,443,328 more live params and two more conv layers there.
        # (b) first block narrowed 1536 -> 1024. v1001 concatenated three tensors
        #     at these levels, [up 512 | s3/s4 512 | t3/t4 512], but the middle one
        #     was torch.zeros_like(...) -- deliberately zeroed so the graph is the
        #     only deep pathway. That allocated 5,244,928 weights reading channels
        #     that are always zero: conv1.weight[:,512:1024] 2,359,296 x2,
        #     skip_proj.weight[:,512:1024] 262,144 x2, norm1 affine 1,024 x2.
        #     They were not strictly zero-gradient (GroupNorm turns a zero channel
        #     into its bias), but their input has spatial std exactly 0, so they
        #     could only add a per-channel constant already provided by conv1.bias.
        #     The zeros are no longer concatenated at all, so the parameters do not
        #     exist. The decoder now matches the baseline block for block AND
        #     channel for channel. Behaviour is unchanged up to one detail:
        #     GroupNorm(32, 1536) put 48 channels per group, so the groups
        #     straddling the 512/1024 boundary mixed live and zeroed channels and
        #     the zeros shifted those groups' statistics. GroupNorm(32, 1024) has
        #     32 per group. Irrelevant from scratch, but it means v1003's decoder
        #     is not numerically comparable to v1001's.
        # was: self.dec4 = nn.ModuleList([ResBlock(1536, 512), ResBlock(512, 512)])
        # was: self.dec3 = nn.ModuleList([ResBlock(1536, 512), ResBlock(512, 512)])
        self.up4 = Upsample(512)
        self.dec4 = nn.ModuleList([ResBlock(1024, 512), ResBlock(512, 512), ResBlock(512, 512)])

        self.up3 = Upsample(512)
        self.dec3 = nn.ModuleList([ResBlock(1024, 512), ResBlock(512, 512), ResBlock(512, 512)])

        self.up2 = Upsample(512)
        self.dec2 = nn.ModuleList([ResBlock(768, 256), ResBlock(256, 256)])

        self.up1 = Upsample(256)
        self.dec1 = nn.ModuleList([ResBlock(384, 128), ResBlock(128, 128)])

        self.up0 = Upsample(128)
        self.dec0 = nn.ModuleList([ResBlock(192, 64), ResBlock(64, 64)])

        self.final_norm = nn.GroupNorm(32, 64)
        self.final_conv = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, h, skips, t3, t4):
        # ---- CHANGED 2026-08-16: s3/s4 are no longer passed or concatenated ----
        # They arrived as torch.zeros_like(...) and contributed nothing but 5.24M
        # parameters reading a constant. skips is now (s0, s1, s2).
        s0, s1, s2 = skips

        with torch.amp.autocast('cuda', enabled=False):
            h = self.up4(h.float())
            h = torch.cat([h, t4.float()], dim=1)
            for block in self.dec4:
                h = block(h)

            h = self.up3(h)
            h = torch.cat([h, t3.float()], dim=1)
            for block in self.dec3:
                h = block(h)

            h = self.up2(h)
            h = torch.cat([h, s2.float()], dim=1)
            for block in self.dec2:
                h = block(h)

            h = self.up1(h)
            h = torch.cat([h, s1.float()], dim=1)
            for block in self.dec1:
                h = block(h)

            h = self.up0(h)
            h = torch.cat([h, s0.float()], dim=1)
            for block in self.dec0:
                h = block(h)

            h = F.silu(self.final_norm(h))
            h = self.final_conv(h)
        return h


# ═══════════════════════════════════════════════════════════════════
# V7: Multi-scale brain ↔ UNet interface
# ═══════════════════════════════════════════════════════════════════

class BrainCollectorV7(nn.Module):
    """Maps ALL encoder levels to brain layer-1 nodes (4802 nodes, C channels).

    Each encoder level is pooled only as needed to fit the node budget.
    No upsampling — every node carries a real encoder feature.

    Node allocation (n1=4802), corrected 2026-08-16 -- this block still listed
    the v1001 numbers (s1: 1024, no s5) after v1003 added the s5 level:
      s3: 1024 (32x32, no collapse)
      s4:  256 (16x16, no collapse)
      s5:   64 (8x8,   no collapse)  <- new in v1003: down4's output, which the
                                        baseline feeds to its bottleneck instead
      s2: 2048 (64x64 -> 32x64, 2:1)
      s1:  960 (128x128 -> 30x32, ~4:1 per dim)  <- was 1024; 64 went to s5
      s0:  450 (256x256 -> 18x25, ~12:1 per dim)

    Order: [s3, s4, s5, s2, s1, s0]. Sums to 4802 exactly.
    """

    def __init__(self, n1, channels):
        super().__init__()
        self.n1 = n1
        self.channels = channels

        # ---- CHANGED 2026-08-15 (v1003: send down4 to the graph, eliminate only mid) ----
        # s5 (8x8 = 64 nodes) added so the graph sees the encoder's full depth.
        # The 64 nodes come out of s1: 1024 -> 960, pooled (30,32) instead of (32,32).
        # was: n_s3 1024 | n_s4 256 | n_s2 2048 | n_s1 1024 | n_s0 = remainder (450)
        self.n_s3 = 1024
        self.n_s4 = 256
        self.n_s5 = 64
        self.n_s2 = 2048
        self.n_s1 = 960
        self.n_s0 = n1 - self.n_s3 - self.n_s4 - self.n_s5 - self.n_s2 - self.n_s1

        self.proj_s3 = nn.Conv2d(512, channels, 1)
        self.proj_s4 = nn.Conv2d(512, channels, 1)
        self.proj_s5 = nn.Conv2d(512, channels, 1)   # 8x8 level (v1003)
        self.proj_s2 = nn.Conv2d(256, channels, 1)
        self.proj_s1 = nn.Conv2d(128, channels, 1)
        self.proj_s0 = nn.Conv2d(64, channels, 1)

        self.s2_pool = (32, 64)
        self.s1_pool = (30, 32)   # was (32,32)=1024; 64 nodes reallocated to s5
        self.s0_pool = (18, 25)

    def forward(self, s0, s1, s2, s3, s4, s5):
        with torch.amp.autocast('cuda', enabled=False):
            s0, s1, s2, s3, s4, s5 = s0.float(), s1.float(), s2.float(), s3.float(), s4.float(), s5.float()

            h_s3 = self.proj_s3(s3).flatten(2).permute(0, 2, 1)
            h_s4 = self.proj_s4(s4).flatten(2).permute(0, 2, 1)
            h_s5 = self.proj_s5(s5).flatten(2).permute(0, 2, 1)
            h_s2 = self.proj_s2(F.adaptive_avg_pool2d(s2, self.s2_pool)).flatten(2).permute(0, 2, 1)
            h_s1 = self.proj_s1(F.adaptive_avg_pool2d(s1, self.s1_pool)).flatten(2).permute(0, 2, 1)
            h_s0 = self.proj_s0(F.adaptive_avg_pool2d(s0, self.s0_pool)).flatten(2).permute(0, 2, 1)

            return torch.cat([h_s3, h_s4, h_s5, h_s2, h_s1, h_s0], dim=1)


# ---- SUPERSEDED 2026-08-15 (v1003 #15/#16) ----------------------------
# Index-sliced a3 into 5 groups, reshaped each to a grid and bilinearly
# upsampled. Assumed node index order carried image-space meaning; it does
# not. Kept for reference; replaced by BrainTwinPathV1003 below.
# class BrainTwinPathV7(nn.Module):
#     """Maps brain layer-3 output (7203 nodes) to features for ALL decoder levels.
#
#     Node groups are assigned to levels and reshaped to spatial grids,
#     expanded only as needed. No universal collapse to a tiny grid.
#
#     Node allocation (n3=7203):
#       t4:  256 nodes → 16x16, no expansion
#       t3: 1024 nodes → 32x32, no expansion
#       t2: 2048 nodes → 32x64 → 64x64, 2:1
#       t1: 2048 nodes → 32x64 → 128x128, ~4:1 per dim
#       t0: 1827 nodes → 43x43 (pad 22) → 256x256, ~6:1 per dim
#     """
#
#     def __init__(self, n3, channels):
#         super().__init__()
#         self.n3 = n3
#         self.channels = channels
#
#         self.n_t4 = 256
#         self.n_t3 = 1024
#         self.n_t2 = 2048
#         self.n_t1 = 2048
#         # ---- CHANGED 2026-08-16 ----
        # was: n_t0 = n3 - n_t3 - n_t2 - n_t1 = 2083, on a 46x46=2116 grid.
        # 2083 is prime, so no grid fits: the 33 pad slots all landed in the last
        # row (33 of 46 columns identically zero, 72% of the bottom row), and
        # bilinear upsampling to 256x256 dragged the bottom ~5 image rows toward
        # zero across the right two-thirds. v1001 had 22 pad in a 43-wide row.
        # 2048 on a 32x64 grid pads nothing. The 35 leftover nodes lose their
        # spatial slot but still reach the decoder via proj_t4 / proj_t5.
#         self.n_t0 = 2048   # (inside the superseded BrainTwinPathV7 block; commented 2026-08-16 -- it had lost its leading # and was being parsed as the last statement of BrainCollectorV7.forward)
#
#         self.grid_t4 = (16, 16)        # produced by proj_t4, not sliced
#         self.grid_t3 = (32, 32)
#         self.grid_t2 = (32, 64)
#         self.grid_t1 = (32, 64)
#         self.grid_t0 = (43, 43)
#
#         self.head4 = nn.Conv2d(channels, 512, 1)
#         self.head3 = nn.Conv2d(channels, 512, 1)
#
#         self.head2 = nn.Conv2d(channels, 256, 1)
#         self.head1 = nn.Conv2d(channels, 128, 1)
#         self.head0 = nn.Conv2d(channels, 64, 1)
#
#         for head in [self.head0, self.head1, self.head2]:
#             nn.init.zeros_(head.weight)
#             nn.init.zeros_(head.bias)
#
#     def _to_grid(self, nodes, grid_h, grid_w):
#         B, N, C = nodes.shape
#         total = grid_h * grid_w
#         if N < total:
#             pad = torch.zeros(B, total - N, C, device=nodes.device, dtype=nodes.dtype)
#             nodes = torch.cat([nodes, pad], dim=1)
#         return nodes[:, :total, :].permute(0, 2, 1).view(B, C, grid_h, grid_w)
#
#     def forward(self, a3):
#         i0 = 0
#         i1 = self.n_t4
#         i2 = i1 + self.n_t3
#         i3 = i2 + self.n_t2
#         i4 = i3 + self.n_t1
#
#         g_t4 = self._to_grid(a3[:, i0:i1, :], *self.grid_t4)
#         g_t3 = self._to_grid(a3[:, i1:i2, :], *self.grid_t3)
#         g_t2 = self._to_grid(a3[:, i2:i3, :], *self.grid_t2)
#         g_t1 = self._to_grid(a3[:, i3:i4, :], *self.grid_t1)
#         g_t0 = self._to_grid(a3[:, i4:,   :], *self.grid_t0)
#
#         t4 = self.head4(g_t4)
#         t3 = self.head3(g_t3)
#         t2 = self.head2(F.interpolate(g_t2, size=(64, 64), mode='bilinear', align_corners=False))
#         t1 = self.head1(F.interpolate(g_t1, size=(128, 128), mode='bilinear', align_corners=False))
#         t0 = self.head0(F.interpolate(g_t0, size=(256, 256), mode='bilinear', align_corners=False))
#         return t0, t1, t2, t3, t4
# ---- end superseded ---------------------------------------------------

# ---- SUPERSEDED 2026-08-16 (same day it was written) ------------------
# BrainTwinPathV1003 was a coordinate-queried LINEAR cross-attention readout,
# written to fix issue #15 (a3 node index was assumed to carry no spatial
# meaning, so reshaping it to a grid looked arbitrary). It is reverted for two
# measured reasons:
#
#  1. It destroyed the graph->decoder pathway. Linear attention sums all nodes
#     into KV (head_dim x head_dim per head) BEFORE any query is applied, and
#     head_dim was C//n_heads = 8. So the whole 7203x32 = 230,496-number brain
#     state reached the decoder through KV (4x8x8=256) + Z (4x8=32) = 288
#     numbers, shared by all 87,360 output positions. Measured: ~285 non-zero
#     singular values in the output Jacobian vs the algebraic bound of 288.
#     v1001's readout delivered all 230,496. An ~800x narrowing, and half the
#     decoder sat downstream of it.
#  2. The premise was wrong. a3 lists the colour-3 nodes in ascending GLOBAL
#     index order, and the global index is the base-7 tensor coordinate, so
#     index-adjacent nodes share their leading digits -- they are siblings in
#     the tensor hierarchy. Reshaping to a grid places hierarchy-siblings
#     adjacent. That layout reflects the construction; it is not arbitrary.
#
# Restoring v1001's slice-and-reshape, extended with a sixth level t5 (8x8) to
# feed the decoder bottleneck that `mid` used to fill. Softmax attention was
# considered and rejected: too much integrative power in the readout for this
# architecture.
# ---- end superseded ---------------------------------------------------


class BrainTwinPathV1003(nn.Module):
    """v1001's BrainTwinPathV7, extended to six levels for the removed bottleneck.

    Node groups are assigned to levels and reshaped to spatial grids, expanded
    only as needed. There is no pooling and no shared summary: every node
    reaches the decoder, 7168 of them through a spatial slice and the remaining
    35 only through proj_t4/proj_t5, which read all n3 nodes.

    The two deepest levels are PROJECTIONS over all n3 nodes, not slices:
      t5:  8x8 = 64 slots  <- proj_t5 over all 7203 nodes, from pass 3
      t4: 16x16 = 256 slots <- proj_t4 over all 7203 nodes, from pass 4
    Sliced as a 64- and a 256-node block they would have carried 0.89% and
    3.55% of the graph state into the decoder's two most semantic inputs.
    A projection lets every node reach them, applied identically per channel.

    The remaining levels are slices of the final pass, as in v1001:
      t3: 1024 nodes -> 32x32, no expansion
      t2: 2048 nodes -> 32x64 ->  64x64,  2:1
      t1: 2048 nodes -> 32x64 -> 128x128, ~4:1 per dim
      t0: 2048 nodes -> 32x64 -> 256x256, no padding (2048 fills 32x64 exactly)
    The 35 nodes past t0's slice are not given a spatial slot; they still reach
    the decoder through proj_t4 and proj_t5, which read all n3 nodes.
    """

    def __init__(self, n3, channels):
        super().__init__()
        self.n3 = n3
        self.channels = channels

        self.n_t3 = 1024
        self.n_t2 = 2048
        self.n_t1 = 2048
        # ---- CHANGED 2026-08-16 ----
        # was: n_t0 = n3 - n_t3 - n_t2 - n_t1 = 2083, on a 46x46=2116 grid.
        # 2083 is prime, so no grid fits: the 33 pad slots all landed in the last
        # row (33 of 46 columns identically zero, 72% of the bottom row), and
        # bilinear upsampling to 256x256 dragged the bottom ~5 image rows toward
        # zero across the right two-thirds. v1001 had 22 pad in a 43-wide row.
        # 2048 on a 32x64 grid pads nothing. The 35 leftover nodes lose their
        # spatial slot but still reach the decoder via proj_t4 / proj_t5.
        self.n_t0 = 2048

        self.grid_t5 = (8, 8)          # produced by proj_t5, not sliced
        self.grid_t4 = (16, 16)        # produced by proj_t4, not sliced
        self.grid_t3 = (32, 32)
        self.grid_t2 = (32, 64)
        self.grid_t1 = (32, 64)
        self.grid_t0 = (32, 64)             # 2048 exactly, pad 0

        # linear projection over the node dimension: all n3 nodes -> 64 slots
        self.proj_t5 = nn.Linear(n3, self.grid_t5[0] * self.grid_t5[1])
        self.proj_t4 = nn.Linear(n3, self.grid_t4[0] * self.grid_t4[1])
        self.head5 = nn.Conv2d(channels, 512, 1)   # new in v1003
        self.head4 = nn.Conv2d(channels, 512, 1)
        self.head3 = nn.Conv2d(channels, 512, 1)
        self.head2 = nn.Conv2d(channels, 256, 1)
        self.head1 = nn.Conv2d(channels, 128, 1)
        self.head0 = nn.Conv2d(channels, 64, 1)

        for head in [self.head0, self.head1, self.head2]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _to_grid(self, nodes, grid_h, grid_w):
        B, N, C = nodes.shape
        total = grid_h * grid_w
        if N < total:
            pad = torch.zeros(B, total - N, C, device=nodes.device, dtype=nodes.dtype)
            nodes = torch.cat([nodes, pad], dim=1)
        return nodes[:, :total, :].permute(0, 2, 1).view(B, C, grid_h, grid_w)

    def forward(self, a3, a3_t5=None, a3_t4=None):
        # ---- ADDED 2026-08-16 ----
        # a3_t5 lets the t5 (8x8 bottleneck) level be read from a DIFFERENT
        # recurrent iteration than the other five levels. All six were taken
        # from the final pass; t5 now comes from an earlier one. See
        # RecurrentBrainNetV7.t5_from_iter.
        src_t5 = a3 if a3_t5 is None else a3_t5
        src_t4 = a3 if a3_t4 is None else a3_t4
        i0 = 0
        i1 = self.n_t3
        i2 = i1 + self.n_t2
        i3 = i2 + self.n_t1

        # t5, t4: learned linear maps over ALL nodes, per channel
        h5, w5 = self.grid_t5
        h4, w4 = self.grid_t4
        g_t5 = self.proj_t5(src_t5.transpose(1, 2)).reshape(-1, self.channels, h5, w5)
        g_t4 = self.proj_t4(src_t4.transpose(1, 2)).reshape(-1, self.channels, h4, w4)

        g_t3 = self._to_grid(a3[:, i0:i1, :], *self.grid_t3)
        g_t2 = self._to_grid(a3[:, i1:i2, :], *self.grid_t2)
        g_t1 = self._to_grid(a3[:, i2:i3, :], *self.grid_t1)
        g_t0 = self._to_grid(a3[:, i3:,   :], *self.grid_t0)

        t5 = self.head5(g_t5)
        t4 = self.head4(g_t4)
        t3 = self.head3(g_t3)
        t2 = self.head2(F.interpolate(g_t2, size=(64, 64), mode='bilinear', align_corners=False))
        t1 = self.head1(F.interpolate(g_t1, size=(128, 128), mode='bilinear', align_corners=False))
        t0 = self.head0(F.interpolate(g_t0, size=(256, 256), mode='bilinear', align_corners=False))
        return t0, t1, t2, t3, t4, t5


class SupernodeLinearAttention(nn.Module):
    """Linear attention with ELU+1 kernel, local to supernodes (1015 groups).
    Shared QKV across all supernodes. Weights initialized at 2x scale."""

    def __init__(self, channels, n_heads, supernode_ids, n_supernodes):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.n_supernodes = n_supernodes

        self.qkv = nn.Linear(channels, 3 * channels)
        self.out_proj = nn.Linear(channels, channels)
        self.ln = nn.LayerNorm(channels)

        with torch.no_grad():
            self.qkv.weight.mul_(2.0)
            self.qkv.bias.mul_(2.0)
            self.out_proj.weight.mul_(2.0)
            self.out_proj.bias.mul_(2.0)

        self.register_buffer('supernode_ids', supernode_ids.long())

    def forward(self, x):
        B, N, C = x.shape
        H, D = self.n_heads, self.head_dim
        K = self.n_supernodes
        residual = x
        x = self.ln(x)

        qkv = self.qkv(x).reshape(B, N, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = F.elu(q, alpha=1.0) + 1.0
        k = F.elu(k, alpha=1.0) + 1.0

        kv_contrib = k.unsqueeze(-1) * v.unsqueeze(-2)
        sids = self.supernode_ids

        idx_kv = sids.view(1, N, 1, 1, 1).expand(B, N, H, D, D)
        KV = torch.zeros(B, K, H, D, D, device=x.device, dtype=x.dtype)
        KV.scatter_add_(1, idx_kv, kv_contrib)

        idx_k = sids.view(1, N, 1, 1).expand(B, N, H, D)
        K_sum = torch.zeros(B, K, H, D, device=x.device, dtype=x.dtype)
        K_sum.scatter_add_(1, idx_k, k)

        KV_n = KV[:, sids]
        K_sum_n = K_sum[:, sids]

        out = torch.einsum('bnhd,bnhde->bnhe', q, KV_n)
        normalizer = (q * K_sum_n).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = out / normalizer

        out = out.reshape(B, N, C)
        out = self.out_proj(out)
        return residual + out


# ═══════════════════════════════════════════════════════════════════
# Full model
# ═══════════════════════════════════════════════════════════════════

class RecurrentBrainNetV7(nn.Module):
    """V1001Super: delayed feedback + classless supernode attention + v10 norm ordering."""

    def __init__(self, graph_path, brain_channels=32, graph_init_scale=5.0, n_iters=5):
        super().__init__()
        data = np.load(graph_path)
        mask_12 = torch.from_numpy(data['mask_12'])
        mask_13 = torch.from_numpy(data['mask_13'])
        mask_23 = torch.from_numpy(data['mask_23'])
        n1, n2, n3 = int(data['n1']), int(data['n2']), int(data['n3'])
        C = brain_channels

        supernode_id_1 = torch.from_numpy(data['supernode_id_1'].astype(np.int64))
        supernode_id_2 = torch.from_numpy(data['supernode_id_2'].astype(np.int64))
        supernode_id_3 = torch.from_numpy(data['supernode_id_3'].astype(np.int64))
        n_supernodes = max(supernode_id_1.max(), supernode_id_2.max(), supernode_id_3.max()).item() + 1

        self.n1, self.n2, self.n3 = n1, n2, n3
        self.brain_channels = C
        self.n_iters = n_iters

        self.encoder = Encoder()
        self.collector = BrainCollectorV7(n1, C)
        self.twin_path = BrainTwinPathV1003(n3, C)
        self.decoder = Decoder()

        self.layer_12 = MultiChannelSparseLinear(n1, n2, mask_12)
        self.layer_13 = MultiChannelSparseLinear(n1, n3, mask_13)
        self.layer_23 = MultiChannelSparseLinear(n2, n3, mask_23)
        self.layer_32 = MultiChannelSparseLinear(n3, n2, mask_23.t().contiguous())
        self.layer_31 = MultiChannelSparseLinear(n3, n1, mask_13.t().contiguous())

        with torch.no_grad():
            for layer in [self.layer_12, self.layer_13, self.layer_23, self.layer_32, self.layer_31]:
                layer.weight.mul_(graph_init_scale)

        # ---- REMOVED 2026-08-26 (v1004 = v1003 minus attention) ----
        # Ablation: v1003 wins by 7.63% over the UNet baseline, but that number
        # mixes two mechanisms -- the recurrent sparse graph, and the supernode
        # attention layered on top of it. Removing the attention isolates the
        # graph's own contribution. Everything else is byte-identical to
        # model_v1003super.py, so the difference between the two runs is
        # attributable to these three modules and the three calls in the loop.
        # The SupernodeLinearAttention class is left defined but unused, and
        # supernode_id_1/2/3 and n_supernodes are still read for the print.
        # was: n_heads = 4
        # was: self.attn_1 = SupernodeLinearAttention(C, n_heads, supernode_id_1, n_supernodes)
        # was: self.attn_2 = SupernodeLinearAttention(C, n_heads, supernode_id_2, n_supernodes)
        # was: self.attn_3 = SupernodeLinearAttention(C, n_heads, supernode_id_3, n_supernodes)

        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.ln3 = nn.LayerNorm(C)
        self.ln_fb = nn.LayerNorm(C)
        self.ln_fb1 = nn.LayerNorm(C)

        # ---- CHANGED 2026-08-26 (v1004) ----
        # was: n_attn = sum(p.numel() for p in [
        # was:     *self.attn_1.parameters(), *self.attn_2.parameters(), *self.attn_3.parameters()])
        n_attn = 0
        n_params = sum(p.numel() for p in self.parameters())
        n_graph = sum(p.numel() for p in [
            self.layer_12.weight,
            self.layer_13.weight,
            self.layer_23.weight,
            self.layer_32.weight,
            self.layer_31.weight,
            self.ln1.weight, self.ln1.bias,
            self.ln2.weight, self.ln2.bias,
            self.ln3.weight, self.ln3.bias,
            self.ln_fb.weight, self.ln_fb.bias,
            self.ln_fb1.weight, self.ln_fb1.bias,
        ])
        n_collector = sum(p.numel() for p in self.collector.parameters())
        n_twin = sum(p.numel() for p in self.twin_path.parameters())
        # ---- REMOVED 2026-08-16 ----
        # detach_collector and collector_grad_scale were set here and read
        # nowhere in this file. train_v1003super.py had an alpha ramp writing
        # collector_grad_scale, never activated, and printed "full gradient
        # (alpha=1.0)" every epoch describing a mechanism that does not exist.
        # was: self.detach_collector = False
        # was: self.collector_grad_scale = 0.0
        # ---- ADDED 2026-08-16 ----
        # Which recurrent pass supplies each projected readout, 1-based.
        # t3/t2/t1/t0 always come from the final pass. Set both to n_iters to
        # read everything from the last pass.
        self.t5_from_iter = 3
        self.t4_from_iter = 4

        print(f"RecurrentBrainNetV1004 (v1003 minus attention): {n_params:,} params total, "
              f"{n_graph:,} graph, {n_attn:,} attn ({n_supernodes} supernodes, UNUSED), "
              f"{n_collector:,} collector, {n_twin:,} twin, "
              f"n1={n1}, n2={n2}, n3={n3}, C={C}, iters={n_iters}")

    def forward(self, x_obs, mask):
        enc_in = torch.cat([x_obs, mask], dim=1)
        with torch.amp.autocast('cuda', enabled=False):
            skips = self.encoder(enc_in.float())
        s0, s1, s2, s3, s4, s5 = skips

        with torch.amp.autocast('cuda', enabled=False):
            a1_base = self.ln1(self.collector(s0, s1, s2, s3, s4, s5))

            a1 = a1_base
            # ---- CHANGED 2026-08-15 (v1003 #13) ----
            # was: a3 = None, which sent iteration 0 down the `else` branch and
            # skipped attn_1 entirely (attn_1 ran 4x, attn_2/attn_3 5x).
            # a3 = None
            a3 = torch.zeros(a1_base.shape[0], self.n3, self.brain_channels,
                             device=a1_base.device, dtype=a1_base.dtype)
            a3_t5 = None                       # snapshot for the t5 readout
            a3_t4 = None                       # snapshot for the t4 readout
            for _it in range(self.n_iters):
                # ---- SIMPLIFIED 2026-08-16 ----
                # a3 is initialised to zeros (fix #13), so `if a3 is not None`
                # was always true and the else branch was unreachable. Removed.
                # was: if a3 is not None: ... else: a2 = gelu(ln2(layer_12(a1)))
                # DELAYED: a2 uses OLD a1
                # ---- REMOVED 2026-08-26 (v1004 = v1003 minus attention) ----
                # The three attn calls are commented out below. Ordering,
                # delay structure and LayerNorm placement are unchanged, so
                # the recurrence is still second-order in a3 with a1 carrying
                # the delayed input+feedback mixture, exactly as in v1003.
                fb = self.ln_fb(self.layer_32(a3))
                a2 = F.gelu(self.ln2(self.layer_12(a1) + fb))
                # was: a2 = self.attn_2(a2)
                # a1 updates AFTER a2
                a1 = a1_base + self.ln_fb1(self.layer_31(a3))
                # was: a1 = self.attn_1(a1)
                # a3 sees FRESH a1
                a3 = self.ln3(self.layer_23(a2) + self.layer_13(a1))
                # was: a3 = self.attn_3(a3)
                if (_it + 1) == self.t5_from_iter:
                    a3_t5 = a3
                if (_it + 1) == self.t4_from_iter:
                    a3_t4 = a3

            t0, t1, t2, t3, t4, t5 = self.twin_path(a3, a3_t5=a3_t5, a3_t4=a3_t4)

        # ---- CHANGED 2026-08-16: stop passing zeroed s3/s4 ----
        # was: skips_mod = (s0+t0, s1+t1, s2+t2, zeros_like(s3), zeros_like(s4))
        # The decoder no longer takes them; the graph remains the only deep path.
        skips_mod = (s0 + t0, s1 + t1, s2 + t2)
        # ---- CHANGED 2026-08-15 (v1003: send down4 to the graph, eliminate only mid) ----
        # was: raw = self.decoder(torch.zeros_like(bottleneck), skips_mod, t3, t4)
        # The bottleneck slot was a zero tensor, which also left decoder.up4.conv
        # with permanently zero gradient. It now gets the graph's 8x8 readout.
        raw = self.decoder(t5, skips_mod, t3, t4)
        recon = mask * x_obs + (1 - mask) * raw
        # ---- CHANGED 2026-08-15 (v1003 #8) ----
        # was: return [recon], None
        # The observed-region loss term was computed on `recon`, but
        # m*(recon - x) = m*(x_obs - x) exactly (binary m kills the raw term),
        # so it had zero gradient and 'val_obs' printed 0.0000 structurally.
        # Expose `raw` so the valid-region loss can be applied to the network
        # output, as in PConv (Liu et al. 2018) and standard inpainting losses.
        return [recon], raw
