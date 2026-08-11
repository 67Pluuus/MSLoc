from transformers import PreTrainedTokenizer
import re
import torch
import torch.nn as nn


class ScoreTower(nn.Module):

    def __init__(self, tokenizer, pretrain_tokenizer=None, hidden_dim=None, pretrain_embedding=None):
        super().__init__()
        self.tokenizer = tokenizer

        assert hidden_dim is not None or pretrain_embedding is not None, \
            'can not set both pretrain embedding and hidden dim to None'
        if pretrain_embedding is not None:
            hidden_dim = pretrain_embedding.shape[1]
        self.embed_tokens = nn.Embedding(len(self.tokenizer.get_vocab()), hidden_dim)

    def _get_pretrain_tokenizer_ids(self, pretrain_tokenizer):
        pretrain_token_ids = {k: pretrain_tokenizer(k).input_ids for k in self.tokenizer.get_vocab()}
        for k, v in pretrain_token_ids.items():
            if v[0] == pretrain_tokenizer.bos_token_id:
                pretrain_token_ids[k] = v[1:]
        return pretrain_token_ids

    def encode(self, scores):
        def insert_separator(X, sep):
            return [self.tokenizer(ele).input_ids for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        scores = [format(s, '0>3.1f') for s in scores]

        input_ids = []
        for ids in insert_separator(scores, '<sep>'):
            input_ids.extend(ids)
        input_ids.extend(self.tokenizer('<sync>').input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class ScoreTokenizer(PreTrainedTokenizer):
    def __init__(self, *args, **kwargs):
        self.vocab = {'<sync>': 0, '<sep>': 1}
        for i in range(10):
            self.vocab[f'{i}'] = i + 2
        self.vocab['.'] = 10 + 2

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        super().__init__(*args, **kwargs)

    def _tokenize(self, text, *args, **kwargs):
        pattern = '|'.join(map(re.escape, self.vocab.keys()))
        tokens = re.findall(pattern, text)
        return [token for token in tokens if token]

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)


if __name__ == "__main__":
    tokenizer = ScoreTokenizer()
    score_tower = ScoreTower(tokenizer, hidden_dim=10)
    print(score_tower([5.0]))
