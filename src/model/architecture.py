import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .attention import SlidingWindowAttention
from .moe import MoELayer
from .rope import RotaryEmbedding


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TardBotDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = SlidingWindowAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            window_size=config.attention_window_size,
            stride=config.attention_stride,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

        use_moe = layer_idx >= config.num_hidden_layers // 4

        if use_moe:
            self.mlp = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                expert_capacity_factor=config.expert_capacity_factor,
                router_aux_loss_coef=config.router_aux_loss_coef,
                expert_checkpoint_dir=config.expert_checkpoint_dir,
                max_experts_in_memory=config.max_experts_in_memory,
                active_expert=config.active_expert,
                device=None,  # Will be set during forward pass
            )
            self.is_moe = True
        else:
            from .moe import Expert
            self.mlp = Expert(config.hidden_size, config.intermediate_size)
            self.is_moe = False

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        router_aux_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe:
            # Ensure expert manager/single expert uses the correct device
            if hasattr(self.mlp, "expert_manager") and self.mlp.expert_manager is not None:
                if self.mlp.expert_manager.device != hidden_states.device:
                    self.mlp.expert_manager.device = hidden_states.device
            if hasattr(self.mlp, "single_expert") and self.mlp.single_expert is not None:
                single_expert_device = next(self.mlp.single_expert.parameters()).device
                if single_expert_device != hidden_states.device:
                    self.mlp.single_expert = self.mlp.single_expert.to(hidden_states.device)
            if hasattr(self.mlp, "dense_expert") and self.mlp.dense_expert is not None:
                dense_device = next(self.mlp.dense_expert.parameters()).device
                if dense_device != hidden_states.device:
                    self.mlp.dense_expert = self.mlp.dense_expert.to(hidden_states.device)
            hidden_states, router_aux_loss = self.mlp(hidden_states, router_aux_loss)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, router_aux_loss


class TardBotModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            TardBotDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def save_expert_checkpoints(self):
        """Save all expert checkpoints to disk."""
        for layer in self.layers:
            if hasattr(layer.mlp, 'save_expert_checkpoints'):
                layer.mlp.save_expert_checkpoints()

    def post_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values[0][0].shape[2] if past_key_values is not None else seq_length,
                dtype=torch.long,
                device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=input_ids.device if input_ids is not None else inputs_embeds.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_attn_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values
        )

        hidden_states = inputs_embeds
        router_aux_loss = torch.tensor(0.0, device=hidden_states.device)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        next_decoder_cache = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs, router_aux_loss = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    router_aux_loss,
                )
            else:
                layer_outputs, router_aux_loss = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    router_aux_loss=router_aux_loss,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
            "router_aux_loss": router_aux_loss,
        }

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
    ) -> torch.Tensor:
        batch_size, seq_length = input_shape

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0

        if attention_mask is not None and len(attention_mask.shape) == 2:
            expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length + past_length)
            inverted_mask = 1.0 - expanded_mask
            inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
            return inverted_mask

        return attention_mask

    def _gradient_checkpointing_func(self, func, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)


class TardBotForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TardBotModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if not config.tie_word_embeddings:
            self.lm_head.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.use_cache
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        return_dict = return_dict if return_dict is not None else True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            router_aux_loss = outputs.get("router_aux_loss", None)
            if router_aux_loss is not None:
                loss = loss + router_aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "router_aux_loss": outputs.get("router_aux_loss"),
        }

    def save_expert_checkpoints(self):
        """Save all expert checkpoints to disk."""
        self.model.save_expert_checkpoints()
