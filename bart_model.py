# Place this file here:
# /opt/python-3.7.7/lib/python3.7/site-packages/simpletransformers/classification/transformer_models/

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, BartClassificationHead

class BartForSequenceClassification(BartPretrainedModel):
    def __init__(self, config, weight=None):
        super(BartForSequenceClassification, self).__init__(config)
        self.model = BartModel(config)
        print(config)
        print(config.eos_token_id)
        print(config.classifier_dropout)
        print(config.d_model)
        print(config.num_labels)
       
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.weight = weight
        self.eos_token_id = config.eos_token_id
        self.num_labels = config.num_labels

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.eos_token_id)

        sentence_representation = hidden_states[eos_mask, :].view( hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, : ]
        logits = self.classifier(sentence_representation)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

