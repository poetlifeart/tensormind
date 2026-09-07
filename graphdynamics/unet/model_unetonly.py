"""
UNet-only baseline 256x256 — the graph model with its graph replaced by an
ordinary UNet bottleneck.

Relationship to RecurrentBrainNetV1003Super, stated precisely (corrected
2026-08-16; this docstring used to claim "same encoder/decoder", which is false
for the encoder):
  decoder  IDENTICAL -- 46,829,699 params both, same blocks, same widths.
  encoder  identical EXCEPT this baseline keeps `self.mid` (3x ResBlock(512,512)
           = 14,164,992 params), the bottleneck the graph replaces. Every other
           encoder module is identical in name, shape and count (26,108,928).
  totals   baseline 87,103,619 vs graph 75,982,503. The baseline carries 14.6%
           more parameters, so a graph win is conservative and a graph loss is
           confounded with capacity.

Decoder levels 4 and 3 take the encoder skips s4 and s3 directly, where the
graph model substitutes t4 and t3 from the graph state.
(Corrected 2026-08-16: the previous two lines here said "same param count for
encoder, fewer for decoder", and both halves were false. The decoders are
identical at 46,829,699; the encoders differ by exactly the bottleneck.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Encoder(nn.Module):
    """5-level encoder: 256->128->64->32->16->8. Channels: [64, 128, 256, 512, 512]."""

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

        self.mid = nn.ModuleList([ResBlock(512, 512), ResBlock(512, 512), ResBlock(512, 512)])

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
        h = self.down4(h)

        for block in self.mid:
            h = block(h)

        return h, [s0, s1, s2, s3, s4]


class Decoder(nn.Module):
    """5-level decoder — skip connections only, no brain twins."""

    def __init__(self):
        super().__init__()
        # Level 4: 8->16, cat(512, 512) = 1024 (h + skip only)
        self.up4 = Upsample(512)
        self.dec4 = nn.ModuleList([ResBlock(1024, 512), ResBlock(512, 512), ResBlock(512, 512)])

        # Level 3: 16->32, cat(512, 512) = 1024 (h + skip only)
        self.up3 = Upsample(512)
        self.dec3 = nn.ModuleList([ResBlock(1024, 512), ResBlock(512, 512), ResBlock(512, 512)])

        # Level 2: 32->64, cat(512, 256) = 768
        self.up2 = Upsample(512)
        self.dec2 = nn.ModuleList([ResBlock(768, 256), ResBlock(256, 256)])

        # Level 1: 64->128, cat(256, 128) = 384
        self.up1 = Upsample(256)
        self.dec1 = nn.ModuleList([ResBlock(384, 128), ResBlock(128, 128)])

        # Level 0: 128->256, cat(128, 64) = 192
        self.up0 = Upsample(128)
        self.dec0 = nn.ModuleList([ResBlock(192, 64), ResBlock(64, 64)])

        self.final_norm = nn.GroupNorm(32, 64)
        self.final_conv = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, h, skips):
        s0, s1, s2, s3, s4 = skips

        with torch.amp.autocast('cuda', enabled=False):
            h = self.up4(h.float())
            h = torch.cat([h, s4.float()], dim=1)
            for block in self.dec4:
                h = block(h)

            h = self.up3(h)
            h = torch.cat([h, s3.float()], dim=1)
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


class UNetOnly(nn.Module):
    """Baseline U-Net: the graph model with an ordinary bottleneck in place of
    the graph. Encoder identical to it except this one keeps the bottleneck
    (`self.mid`, 14,164,992 params); decoder identical (46,829,699 both)."""

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"UNetOnly: {n_params:,} params")

    def forward(self, x_obs, mask):
        enc_in = torch.cat([x_obs, mask], dim=1)
        # ---- CHANGED 2026-08-16: encoder forced to fp32, matching v1003 ----
        # The decoder was already wrapped in autocast(enabled=False) but the
        # encoder was not, so under the training loop's fp16 autocast 40,273,920
        # params -- 46% of this model, and identical to the graph model's encoder
        # apart from `mid` -- computed and backpropagated in half precision, while the
        # graph model runs every block in fp32. Same modules, different
        # arithmetic, on the one component the comparison holds fixed.
        # was: bottleneck, skips = self.encoder(enc_in)
        with torch.amp.autocast('cuda', enabled=False):
            bottleneck, skips = self.encoder(enc_in.float())
        raw = self.decoder(bottleneck, skips)
        recon = mask * x_obs + (1 - mask) * raw
        # ---- CHANGED 2026-08-16 ----
        # was: return [recon], None
        # The observed-region loss term was computed on `recon`, but with binary
        # mask m*(recon - x) = m*(x_obs - x) exactly, so it had zero gradient and
        # val_obs printed 0.0000 structurally. v1003 was fixed to apply that term
        # to `raw`; this baseline was not, so the two models were optimising
        # different objectives. Exposing `raw` so both use the same loss.
        return [recon], raw
