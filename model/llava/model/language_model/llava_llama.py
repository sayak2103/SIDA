#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""
LLAVA_LLAMA model implementation:
    Custom language model based on the LLaMA architecture, 
    extending it with multimodal capabilities under the "Llava" model type. 
    It integrates with the Hugging Face Transformers library and defines a causal 
    language model (LlavaLlamaForCausalLM) that supports multimodal inputs, 
    such as text and images.

    Key Classes:
    - LlavaConfig: Custom configuration class for Llava models, extending LlamaConfig.
    - LlavaLlamaModel: Custom model class that extends LlamaModel with Llava capabilities.
          - LlavaMetaModel: Base class for Llava models, providing multimodal support.
          - LlamaModel: Base class for LLaMA models, providing the core architecture.
    - LlavaLlamaForCausalLM: Custom causal language model class that extends LlamaForCausalLM,
    integrating LlavaLlamaModel and adding support for multimodal inputs and outputs.
    
    Key Functions:
    - __init__ (in LlavaLlamaForCausalLM)
        -Initializes the model with a multimodal backbone (LlavaLlamaModel) and
          a linear head (lm_head).
        -Calls post_init() for weight initialization.
    - forward (in LlavaLlamaForCausalLM)
        -Defines the forward pass of the model, handling multimodal inputs,
            computing logits, and optionally calculating loss.
    - prepare_inputs_for_generation (in LlavaLlamaForCausalLM)
        -Prepares inputs for the generation step, handling past key values,
            attention masks, and multimodal inputs like images.          
"""


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
Inputs:
    - input_ids:
        A tensor of shape (batch_size, sequence_length) 
        containing token IDs for the input sequence.
    - attention_mask:
        A tensor of shape (batch_size, sequence_length) 
        indicating which tokens should be attended to (1 for valid tokens, 0 for padding).
    - past_key_values:
        A list of tensors containing precomputed key and 
        value states for faster decoding during generation.
    - inputs_embeds:
        A tensor of shape (batch_size, sequence_length, embedding_dim) 
        containing precomputed embeddings for the input tokens.
    - images:
        A tensor of shape (batch_size, channels, height, width)
          representing multimodal image inputs.
Targets:
    - labels:
        A tensor of shape (batch_size, sequence_length) containing 
        the target token IDs for loss computation.
    - logits:
        A tensor of shape (batch_size, sequence_length, vocab_size) 
        containing the predicted token probabilities.
    - loss:
        A scalar tensor representing the cross-entropy loss, 
        computed if labels are provided.
    - hidden_states:
        A tensor of shape (batch_size, sequence_length, hidden_size) 
        containing the hidden states of the model.
    - attentions:
        A list of tensors containing attention weights for each layer (optional).
Return:
    - loss:
        Type: torch.FloatTensor (optional)
        Description: The cross-entropy loss, computed if labels are provided.
    - logits:
        Type: torch.FloatTensor
        Shape: (batch_size, sequence_length, vocab_size)
        Description: The predicted token probabilities.
    - past_key_values:
        Type: List[torch.FloatTensor]
        Description: Contains precomputed key and value states for faster decoding.
    - hidden_states:
        Type: torch.FloatTensor (optional)
        Description: Hidden states of the model for each layer.
    - attentions:
        Type: List[torch.FloatTensor] (optional)
        Description: Attention weights for each layer.
    """

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update( 
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
