import os
from argparse import ArgumentParser
from genericpath import exists
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import BertWordPieceTokenizer
from transformers.optimization import AdamW

from bert_models import BertPretrain


class MLMDataset(Dataset):
    def __init__(self, mlm_txt, tokenizer, max_seq_length=512):
        super().__init__()

        with open(mlm_txt) as f:
            self.data = f.readlines()

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.as_tensor(self.tokenizer.encode(self.data[index]).ids)[: self.max_seq_length]


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10%
    original.
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability
    # args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # special_tokens_mask = [
    #   # for val in labels.tolist()
    #   tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
    # ]
    probability_matrix.masked_fill_(torch.eq(labels, 0).cpu(), value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.token_to_id("[MASK]")

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def main(args):
    bert = BertPretrain(args.model_name_or_path).cuda()
    optimizer = AdamW(bert.parameters(), lr=5e-5, eps=1e-8)

    # Configure tokenizer
    token_vocab_path = "bert-base-uncased-vocab.txt"
    tokenizer = BertWordPieceTokenizer(token_vocab_path, lowercase=True)

    dataset = MLMDataset(args.mlm_data_txt, tokenizer, 50)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        collate_fn=lambda samples: pad_sequence([s for s in samples], batch_first=True),
    )

    bert.train()
    for epoch in range(1, 1 + args.num_epochs):
        losses = []
        for batch in tqdm(dataloader, desc=f"Epoch {epoch:2d}", ncols=120):
            bert.zero_grad()

            inputs, labels = mask_tokens(batch, tokenizer)
            loss = bert(inputs.cuda(), labels.cuda())
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(bert.parameters())

            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch:2d} - loss: {np.mean(losses)}")

    os.makedirs(args.output_dir, exist_ok=True)
    bert.bert_model.save_pretrained(args.output_dir)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--mlm_data_txt")
    parser.add_argument("--output_dir")

    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
