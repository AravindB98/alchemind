"""Optional deep-learning generator: a character-level SMILES language model.

This module trains an LSTM to model the "language" of SMILES strings and then
samples novel molecules from it — the neural counterpart to the genetic
generator. PyTorch is an *optional* dependency (``pip install alchemind[deep]``);
importing this module without torch raises a clear, actionable error, and the
rest of Alchemind keeps working via the genetic generator.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

try:  # torch is optional
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when torch is absent
    _TORCH_AVAILABLE = False

from ..utils.chem import canonical_smiles

BOS, EOS, PAD = "^", "$", " "


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "The char-RNN generator needs PyTorch. Install it with "
            "`pip install alchemind[deep]` (or `pip install torch`)."
        )


class SmilesVocab:
    def __init__(self, smiles: List[str]):
        chars = sorted({c for s in smiles for c in s} | {BOS, EOS, PAD})
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[BOS]] + [self.stoi[c] for c in s] + [self.stoi[EOS]]


if _TORCH_AVAILABLE:

    class CharRNN(nn.Module):
        def __init__(self, vocab_size: int, embed: int = 64, hidden: int = 256,
                     layers: int = 2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed)
            self.lstm = nn.LSTM(embed, hidden, layers, batch_first=True)
            self.fc = nn.Linear(hidden, vocab_size)

        def forward(self, x, hidden=None):
            emb = self.embed(x)
            out, hidden = self.lstm(emb, hidden)
            return self.fc(out), hidden


class CharRNNGenerator:
    """Train / sample a character-level SMILES generator."""

    def __init__(self, vocab: Optional[SmilesVocab] = None, model=None,
                 device: str = "cpu"):
        _require_torch()
        self.vocab = vocab
        self.model = model
        self.device = device

    def train(self, smiles: List[str], epochs: int = 20, lr: float = 2e-3,
              batch_size: int = 64) -> "CharRNNGenerator":
        _require_torch()
        self.vocab = SmilesVocab(smiles)
        self.model = CharRNN(len(self.vocab)).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.stoi[PAD])
        encoded = [self.vocab.encode(s) for s in smiles]
        maxlen = max(len(e) for e in encoded)
        pad = self.vocab.stoi[PAD]
        padded = torch.tensor(
            [e + [pad] * (maxlen - len(e)) for e in encoded], dtype=torch.long
        )
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(padded.size(0))
            for i in range(0, padded.size(0), batch_size):
                batch = padded[perm[i:i + batch_size]].to(self.device)
                logits, _ = self.model(batch[:, :-1])
                loss = loss_fn(logits.reshape(-1, len(self.vocab)),
                               batch[:, 1:].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def sample(self, n: int = 20, max_len: int = 120, temperature: float = 1.0,
               unique_valid: bool = True) -> List[str]:
        _require_torch()
        assert self.model is not None and self.vocab is not None, "Train or load first."
        self.model.eval()
        results: List[str] = []
        with torch.no_grad():
            results = self._sample_loop(n, max_len, temperature, unique_valid)
        return results

    def _sample_loop(self, n, max_len, temperature, unique_valid) -> List[str]:
        results: List[str] = []
        for _ in range(n * 4):
            if unique_valid and len(results) >= n:
                break
            token = torch.tensor([[self.vocab.stoi[BOS]]], device=self.device)
            hidden = None
            chars: List[str] = []
            for _ in range(max_len):
                logits, hidden = self.model(token, hidden)
                probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                idx = int(torch.multinomial(probs, 1).item())
                ch = self.vocab.itos[idx]
                if ch == EOS:
                    break
                if ch in (BOS, PAD):
                    continue
                chars.append(ch)
                token = torch.tensor([[idx]], device=self.device)
            smi = canonical_smiles("".join(chars))
            if smi and (not unique_valid or smi not in results):
                results.append(smi)
        return results[:n]

    def save(self, path: Path) -> None:
        _require_torch()
        torch.save({"state": self.model.state_dict(), "itos": self.vocab.itos}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "CharRNNGenerator":
        _require_torch()
        ckpt = torch.load(path, map_location=device)
        vocab = SmilesVocab.__new__(SmilesVocab)
        vocab.itos = ckpt["itos"]
        vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
        model = CharRNN(len(vocab)).to(device)
        model.load_state_dict(ckpt["state"])
        return cls(vocab=vocab, model=model, device=device)
