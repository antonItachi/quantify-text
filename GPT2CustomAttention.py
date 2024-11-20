import torch
import torch.nn as nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class CustomGPT2FlashAttention(torch.nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, layer_idx=None):
        super().__init__()

        n_state = nx  # hidden_dim (n_embd)
        assert n_state % config.n_head == 0, "n_state must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = n_state // config.n_head
        self.scale = scale
        self.layer_idx = layer_idx
        self.config = config

        # Variables for Attention mechanism
        self.q_attn = torch.nn.Linear(nx, n_state, bias=True)
        self.k_attn = torch.nn.Linear(nx, n_state, bias=True)
        self.v_attn = torch.nn.Linear(nx, n_state, bias=True)

        self.c_proj = torch.nn.Linear(n_state, nx, bias=True)

        # Dropout
        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

        # Masking
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def split_heads(self, x):
        """
        Split heads without separate logic for key or query to ensure compatibility.
        """
        new_shape = x.size()[:-1] + (self.n_head, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # [batch, num_heads, seq_length, head_dim]

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.n_head * self.head_dim,)
        return x.view(*new_shape)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k.transpose(-1, -2))

        # Scaling
        if self.scale:
            w = w / (self.head_dim ** 0.5)
        if getattr(self.config, "scale_attn_by_inverse_layer_idx", False):
            w = w / float(self.layer_idx + 1)

        # Apply causal mask
        mask = self.bias[:, :, : w.size(-2), : w.size(-1)]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask

        outputs = (torch.matmul(w, v),)
        if output_attentions:
            outputs += (w,)
        return outputs

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False,
                output_attentions=False):
        batch, seq_len, hidden_dim = hidden_states.size()
        query = self.q_attn(hidden_states)
        key = self.k_attn(hidden_states)
        value = self.v_attn(hidden_states)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        query_length = query.shape[2]
        tgt_len = key.shape[2]

        num_heads = self.n_head
        head_dim = hidden_dim // num_heads

        query = query.reshape(batch, query_length, num_heads, head_dim)
        key = key.reshape(batch, tgt_len, num_heads, head_dim)
        value = value.reshape(batch, tgt_len, num_heads, head_dim)

        # Changing data type for Flash Attention.
        target_dtype = torch.float16

        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

        # Calculating Flash Attention
        attn_outputs = flash_attn_func(query, key, value)

        attn_weights_reshaped = attn_outputs.reshape(batch, query_length, num_heads * head_dim).to(torch.float32)
        attn_output = self.c_proj(attn_weights_reshaped)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights_reshaped,)

        return outputs


class CustomGPT2Attention(torch.nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, layer_idx=None):
        super().__init__()

        n_state = nx  # hidden_dim (n_embd)
        assert n_state % config.n_head == 0, "n_state must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = n_state // config.n_head
        self.scale = scale
        self.layer_idx = layer_idx
        self.config = config

        # Variables for Attention mechanism
        self.q_attn = torch.nn.Linear(nx, n_state, bias=True)
        self.k_attn = torch.nn.Linear(nx, n_state, bias=True)
        self.v_attn = torch.nn.Linear(nx, n_state, bias=True)

        self.c_proj = torch.nn.Linear(n_state, nx, bias=True)

        # Dropout
        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

        # Masking
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def split_heads(self, x):
        """
        Split heads without separate logic for key or query to ensure compatibility.
        """
        new_shape = x.size()[:-1] + (self.n_head, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # [batch, num_heads, seq_length, head_dim]

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.n_head * self.head_dim,)
        return x.view(*new_shape)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k.transpose(-1, -2))

        # Scaling
        if self.scale:
            w = w / (self.head_dim ** 0.5)
        if getattr(self.config, "scale_attn_by_inverse_layer_idx", False):
            w = w / float(self.layer_idx + 1)

        # Apply causal mask
        mask = self.bias[:, :, : w.size(-2), : w.size(-1)]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask

        outputs = (torch.matmul(w, v),)
        if output_attentions:
            outputs += (w,)
        return outputs

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False,
                output_attentions=False):
        query = self.q_attn(hidden_states)
        key = self.k_attn(hidden_states)
        value = self.v_attn(hidden_states)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache:
            present = (key, value)
        else:
            present = None

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)

        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = (a, present) + attn_outputs[1:]
        return outputs