import zipfile
import os.path
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from pathlib import Path
import bert_ner_model
from bert_ner_model import BERTForNerModel, dataset_target_path, dataset_source_path, model_save_path


def prepare_dataset(tokenizer):
    # https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
    if not os.path.isfile(dataset_source_path + 'ner.csv'):
        with zipfile.ZipFile(dataset_source_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_target_path)
    df = pd.read_csv(dataset_target_path + 'ner.csv', encoding="ISO-8859-1", error_bad_lines=False)
    dataset = df[['sentence_idx', 'word', 'tag']]
    aggregate_function = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
    grouped_sentences = [s for s in dataset.groupby("sentence_idx").apply(aggregate_function)]
    tags = list(set(dataset["tag"].values))
    # Converters
    tag2idx, idx2tag = {t: i for i, t in enumerate(tags)}, {i: t for i, t in enumerate(tags)}
    sentences = [' '.join([s[0] for s in sent]) for sent in grouped_sentences]
    labels = [[tag2idx.get(s[1]) for s in sent] for sent in grouped_sentences]
    max_len = 200

    class NERDataset(Dataset):
        def __init__(self, sentences, labels):
            self.len = len(sentences)
            self.sentences = sentences
            self.labels = labels

        def __getitem__(self, index):
            inputs = tokenizer.encode_plus(str(self.sentences[index]),
                                           None,
                                           add_special_tokens=True,
                                           max_length=max_len,
                                           padding='max_length',
                                           return_token_type_ids=True)
            label = self.labels[index]
            label.extend([tag2idx['O']] * max_len)
            return {
                'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
                'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
                'tags': torch.tensor(label[:max_len], dtype=torch.long)
            }

        def __len__(self):
            return self.len

    return NERDataset(sentences, labels), tag2idx, idx2tag


def fine_tune_bert_for_ner():
    epochs = 5
    batch_size = 32
    learning_rate = 2e-05
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    dataset, tag2idx, idx2tag = prepare_dataset(tokenizer)
    model = BERTForNerModel(len(tag2idx)).to(device)
    training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        for _, data in enumerate(training_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['tags'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            model(ids, mask, labels=targets)[0].backward()
            optimizer.step()
        print("Epoch {} finished".format(epoch + 1))

    # Test model
    test_sentence = "My name is James and I live in London, England."
    # [('My', 'O'), ('name', 'O'), ('is', 'O'), ('James', 'B-per'), ('and', 'O'), ('I', 'O'), ('live', 'O'),
    # ('in', 'O'), ('London', 'B-geo'), (',', 'O'), ('England', 'B-geo'), ('.', 'O')]
    print(bert_ner_model.classify_tokens(model, tokenizer, idx2tag, test_sentence, device))

    # Save model and idx2tag map
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path + "bert_ner.pth")
    with open(model_save_path + "idx2tag_map.pkl", 'wb') as output_file:
        pickle.dump(idx2tag, output_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    fine_tune_bert_for_ner()
