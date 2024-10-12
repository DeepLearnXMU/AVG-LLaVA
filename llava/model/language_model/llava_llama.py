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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward_single_granularity(
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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        vis_token_granularity: Optional[int] = None,
        reduction: Optional[str] = 'mean',
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                vis_token_granularity = vis_token_granularity,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction=reduction)
                
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if reduction == 'none':
                loss = loss.view(labels.size(0), labels.size(1) - 1)
                loss = loss.sum(dim = 1)
                # Divide the number of elements that are valid
                loss = loss / (labels != -100).sum(dim = 1)
            
        return loss, logits, outputs

    def rank_loss(self, logits, margin=0.1, scale=1):
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        scores = log_probs
        scores = scores * scale
        margin_loss = torch.tensor(0,device = logits.device, dtype=logits.dtype)
        
        for j in range(1, logits.shape[1]):
            pos_score = scores[:,:-j]
            neg_score = scores[:,j:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * j)
            # if torch.isnan(loss_func(pos_score,neg_score,ones)):
            #     print()
            margin_loss += loss_func(pos_score,neg_score,ones)

        return margin_loss

    def adaptive_rank_loss(self, logits, llm_loss, margin=0.1, scale=1):
        # logits: (batch, len(vis_token_granularity))
        # llm_loss: (batch, len(vis_token_granularity))
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        scores = log_probs
        scores = scores * scale
        margin_loss = torch.tensor(0,device = logits.device, dtype=logits.dtype)
        
        for j in range(1, logits.shape[1]):
            pos_score = scores[:,:-j]
            neg_score = scores[:,j:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            updated_margin = margin * j * (llm_loss[:, j:] - llm_loss[:, :-j]).contiguous().view(-1)
            # updated_margin = margin * (llm_loss[:, j:] - llm_loss[:, :-j]).contiguous().view(-1)
            # updated_margin = margin * j
            diff = neg_score - pos_score + updated_margin
            # print("margin: {}".format(updated_margin))
            margin_loss += torch.max(diff, torch.zeros_like(diff)).mean()

        return margin_loss
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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if self.training and self.config.vis_token_granularity is not None:
            if self.config.router_select == True:
                loss_accumulate = []
                for vis_token_granularity_element in self.config.vis_token_granularity:
                    with torch.no_grad():
                        loss_item, logits, outputs = self.forward_single_granularity(
                            input_ids = input_ids,
                            attention_mask = attention_mask,
                            position_ids = position_ids,
                            past_key_values = past_key_values,
                            inputs_embeds = inputs_embeds,
                            labels = labels,
                            use_cache = use_cache,
                            output_attentions = output_attentions,
                            output_hidden_states = output_hidden_states,
                            images = images,
                            image_sizes = image_sizes,
                            return_dict = return_dict,
                            vis_token_granularity= vis_token_granularity_element,
                            reduction='none'
                        )
                    loss_accumulate.append(loss_item)
                    assert len(outputs) == 1, 'len(outputs) == 1 is False'

                # Sort loss from large to small
                loss_accumulate = torch.stack(loss_accumulate, dim = 0)
                loss_accumulate = loss_accumulate.permute(1, 0)
                loss_accumulate, indices = torch.sort(loss_accumulate, descending = False)

                router_logits = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )
                # router_logits = torch.rand_like(loss_accumulate) #(batch, len(vis_token_granularity))
                # loss_fct = CrossEntropyLoss(label_smoothing=0.1)
                loss_fct = CrossEntropyLoss()
                #The index with the smallest loss is used as the label
                router_labels = indices[:, 0]
                router_labels = router_labels.to(loss_accumulate.device)
                loss_router = loss_fct(router_logits, router_labels)
                # sort router_logits according to the indices
                router_logits = router_logits.gather(1, indices)
                rank_loss = self.adaptive_rank_loss(router_logits, loss_accumulate, margin=self.config.margin, scale=1)
                loss = self.config.ce_loss_weight * loss_router + self.config.rank_loss_weight * rank_loss
                # print("Adaptive Router loss: {}, Rank loss: {}".format(rank_loss, self.rank_loss(router_logits, margin=self.config.margin, scale=1)))
                output = (router_logits,) + outputs[1:]
                return (loss,) + output
            else:
                # print("The model is in training mode.")
                loss = 0
                logits_accumulate = []
                for vis_token_granularity_element in self.config.vis_token_granularity:
                    loss_item,  logits, outputs = self.forward_single_granularity(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        position_ids = position_ids,
                        past_key_values = past_key_values,
                        inputs_embeds = inputs_embeds,
                        labels = labels,
                        use_cache = use_cache,
                        output_attentions = output_attentions,
                        output_hidden_states = output_hidden_states,
                        images = images,
                        image_sizes = image_sizes,
                        return_dict = return_dict,
                        vis_token_granularity= vis_token_granularity_element
                    )
                    loss += loss_item/len(self.config.vis_token_granularity)
                    logits_accumulate.append(logits)
                    assert len(outputs) == 1, 'len(outputs) == 1 is False'
                logits = torch.cat(logits_accumulate, dim = 1)
                        
                if not return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output if loss is not None else output
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            
        else:
            # print("The model is in evaluation mode or trained without vis_token_granularity.")
            
        
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes
                )

            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        vis_token_granularity = kwargs.pop("vis_token_granularity", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                vis_token_granularity = vis_token_granularity
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
