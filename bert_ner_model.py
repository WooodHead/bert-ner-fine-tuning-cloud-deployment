import numpy as np
import torch
from transformers import BertModel, BertConfig

dataset_source_path = "dataset/dataset.zip"
dataset_target_path = "dataset/"
model_save_path = "weights/"


class BERTForNerModel(torch.nn.Module):
    def __init__(self, num_labels, pretrained=True):
        super(BERTForNerModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased') if pretrained \
            else BertModel(BertConfig.from_pretrained('bert-base-cased'))
        self.dropout = torch.nn.Dropout(0.1)
        self.num_labels = num_labels
        self.classifier = torch.nn.Linear(768, self.num_labels)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, ids, mask=None, labels=None):
        outputs = self.bert(ids, mask)
        logits = self.classifier(self.dropout(outputs[0]))
        loss = None
        if labels is not None:
            target_labels = labels.view(-1)
            if mask is not None:
                target_labels = torch.where(mask.view(-1) == 1, labels.view(-1),
                                            torch.tensor(self.loss_function.ignore_index).type_as(labels))
            loss = self.loss_function(logits.view(-1, self.num_labels), target_labels)
        return loss, logits


def classify_tokens(model, tokenizer, idx2tag, sentence, device):
    tokenized_sentence = tokenizer.encode(sentence)
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[1].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0], skip_special_tokens=True)
    target_tokens, target_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            target_tokens[-1] = target_tokens[-1] + token[2:]
        else:
            target_labels.append(idx2tag[label_idx])
            target_tokens.append(token)
    return list(zip(target_tokens, target_labels))
