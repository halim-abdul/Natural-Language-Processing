import torch
from torch import nn
from transformers import BertConfig, BertModel as HFBertModel

from utils import get_parameter_dtype

class BertPreTrainedModel(nn.Module):
    config_class = BertConfig

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config
        self.name_or_path = getattr(config, "name_or_path", "")

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        local_files_only = kwargs.pop("local_files_only", False)
        config = kwargs.pop("config", None)
        if config is None:
            config = BertConfig.from_pretrained(
                pretrained_model_name_or_path,
                local_files_only=local_files_only,
            )
        model = cls(config, *model_args, **kwargs)
        hf = HFBertModel.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
        state = hf.state_dict()

        replacements = {
            "embeddings.word_embeddings": "word_embedding",
            "embeddings.position_embeddings": "pos_embedding",
            "embeddings.token_type_embeddings": "tk_type_embedding",
            "embeddings.LayerNorm": "embed_layer_norm",
            "encoder.layer": "bert_layers",
            "pooler.dense": "pooler_dense",
            "attention.self": "self_attention",
            "attention.output.dense": "attention_dense",
            "attention.output.LayerNorm": "attention_layer_norm",
            "intermediate.dense": "interm_dense",
            "output.dense": "out_dense",
            "output.LayerNorm": "out_layer_norm",
        }

        mapped = {}
        for key, value in state.items():
            new_key = key
            if new_key.startswith("bert."):
                new_key = new_key[5:]
            for old, new in replacements.items():
                new_key = new_key.replace(old, new)
            mapped[new_key] = value

        model.load_state_dict(mapped, strict=False)
        model.eval()
        return model
