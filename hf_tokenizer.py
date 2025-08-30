"""Optional Hugging Face tokenizer wrapper with safe fallback to CharTokenizer."""
from typing import Optional, Tuple

try:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

from tokenizer import CharTokenizer


class HFWrapper:
    def __init__(self, model_name: Optional[str] = None, sample_text: Optional[str] = None):
        if HF_AVAILABLE and model_name is not None:
            # use pretrained fast tokenizer
            self.tok = PreTrainedTokenizerFast.from_pretrained(model_name)
            self.is_fast = True
        else:
            # fallback to char tokenizer built from sample_text
            assert sample_text is not None, "sample_text required for char fallback"
            self.tok = CharTokenizer.from_text(sample_text)
            self.is_fast = False

    def encode(self, text: str):
        if HF_AVAILABLE and self.is_fast:
            return self.tok.encode(text).ids
        return self.tok.encode(text)

    def decode(self, ids):
        if HF_AVAILABLE and self.is_fast:
            return self.tok.decode(ids)
        return self.tok.decode(ids)
