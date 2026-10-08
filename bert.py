import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        scores = torch.matmul(query, key.transpose(-1,-2))
        scores = scores / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = context.transpose(1, 2).contiguous()
        context = context.view(context.size(0), context.size(1), self.all_head_size)
        return context

    def forward(self, hidden_states, attention_mask):
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        return self.attention(key_layer, query_layer, value_layer, attention_mask)

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        projected = dense_layer(output)
        dropped = dropout(projected)
        return ln_layer(input + dropped)

    def forward(self, hidden_states, attention_mask):
        attn_output = self.self_attention(hidden_states, attention_mask)
        attn_output = self.add_norm(
            hidden_states, attn_output, self.attention_dense,
            self.attention_dropout, self.attention_layer_norm
        )
        intermediate = self.interm_dense(attn_output)
        mlp_output = self.interm_af(intermediate)
        return self.add_norm(
            attn_output, mlp_output, self.out_dense,
            self.out_dropout, self.out_layer_norm
        )

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()
        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        inputs_embeds = self.word_embedding(input_ids)
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)
        hidden_states = inputs_embeds + pos_embeds + tk_type_embeds
        hidden_states = self.embed_layer_norm(hidden_states)
        hidden_states = self.embed_dropout(hidden_states)
        return hidden_states

    def encode(self, hidden_states, attention_mask):
        extended_attention_mask = get_extended_attention_mask(attention_mask, self.dtype)
        for layer_module in self.bert_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask):
        embedding_output = self.embed(input_ids=input_ids)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        first_tk = self.pooler_af(self.pooler_dense(sequence_output[:, 0]))
        return {"last_hidden_state": sequence_output, "pooler_output": first_tk}
