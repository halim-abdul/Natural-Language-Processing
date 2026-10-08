"""Tokenizer compatibility layer.

The original coursework submission bundled a local copy of the BERT tokenizer.
For the public project we use the maintained Hugging Face implementation while
preserving the import used by the training code.
"""
from transformers import BertTokenizer

__all__ = ["BertTokenizer"]
