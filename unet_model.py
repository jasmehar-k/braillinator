import string
import torch
import torch.nn as nn
import torch.nn.functional as F

PRINTABLE_CHARS = string.printable  # 100 chars
VOCAB_SIZE = len(PRINTABLE_CHARS) + 1  # index 0 = padding
MAX_LEN = 64
EMBED_DIM = 32
BOTTLENECK_CH = 256


class CharTokenizer:
    def __init__(self, char_to_idx: dict = None):
        if char_to_idx is not None:
            self.char_to_idx = char_to_idx
        else:
            self.char_to_idx = {ch: i + 1 for i, ch in enumerate(PRINTABLE_CHARS)}

    def encode(self, text: str) -> torch.Tensor:
        tokens = [self.char_to_idx.get(ch, 0) for ch in text[:MAX_LEN]]
        tokens += [0] * (MAX_LEN - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


class FiLMBlock(nn.Module):
    def __init__(self, num_channels: int, text_dim: int):
        super().__init__()
        self.gamma_fc = nn.Linear(text_dim, num_channels)
        self.beta_fc = nn.Linear(text_dim, num_channels)

    def forward(self, x: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)   text_vec: (B, text_dim)
        gamma = self.gamma_fc(text_vec).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(text_vec).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConditionalUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Text conditioning
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.embed_pool = nn.Linear(EMBED_DIM, BOTTLENECK_CH)

        # Encoder: channels [1, 32, 64, 128, 256]
        self.enc1 = EncoderBlock(1, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        # Bottleneck at 16x16 (for 256x256 input)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.film = FiLMBlock(BOTTLENECK_CH, BOTTLENECK_CH)

        # Decoder
        self.dec4 = DecoderBlock(256, 256, 128)
        self.dec3 = DecoderBlock(128, 128, 64)
        self.dec2 = DecoderBlock(64, 64, 32)
        self.dec1 = DecoderBlock(32, 32, 16)

        self.output_conv = nn.Conv2d(16, 1, 1)
        self.output_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 256, 256)   tokens: (B, MAX_LEN)
        emb = self.embedding(tokens)                          # (B, MAX_LEN, EMBED_DIM)
        text_vec = F.relu(self.embed_pool(emb.mean(dim=1)))  # (B, BOTTLENECK_CH)

        x1, s1 = self.enc1(x)   # pooled (B,32,128,128), skip (B,32,256,256)
        x2, s2 = self.enc2(x1)  # pooled (B,64,64,64),   skip (B,64,128,128)
        x3, s3 = self.enc3(x2)  # pooled (B,128,32,32),  skip (B,128,64,64)
        x4, s4 = self.enc4(x3)  # pooled (B,256,16,16),  skip (B,256,32,32)

        b = self.bottleneck(x4)          # (B, 256, 16, 16)
        b = self.film(b, text_vec)       # FiLM conditioning

        d = self.dec4(b, s4)   # (B, 128, 32, 32)
        d = self.dec3(d, s3)   # (B, 64, 64, 64)
        d = self.dec2(d, s2)   # (B, 32, 128, 128)
        d = self.dec1(d, s1)   # (B, 16, 256, 256)

        return self.output_act(self.output_conv(d))  # (B, 1, 256, 256)
