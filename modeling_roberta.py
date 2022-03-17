# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import math
import torch
import torch.nn as nn

from transformers import *
from torch.nn import CrossEntropyLoss, MSELoss
from modules import SCAttention, split_ques_context, TrmCoAtt
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class RobertaForQuestionAnsweringAVPool(RobertaPreTrainedModel):
    def __init__(self, config, no_answer_loss_coef: float = 1.0, ):
        super(RobertaForQuestionAnsweringAVPool, self).__init__(config)
        self.num_labels = config.num_labels
        self.no_answer_loss_coef = no_answer_loss_coef
        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 2))
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None, is_impossibles=None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        first_word = sequence_output[:, 0, :]

        has_log = self.has_ans(first_word)

        outputs = (start_logits, end_logits, has_log,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            choice_loss = loss_fct(has_log, is_impossibles)
            total_loss = (start_loss + end_loss + self.no_answer_loss_coef * choice_loss) / 3
            outputs = (total_loss,) + outputs
            # print(sum(is_impossibles==1),sum(is_impossibles==0))cd

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)