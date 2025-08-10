import torch
import transformers
from torch import nn
from transformers import GPT2Tokenizer


class GPT2(nn.Module):

    def __init__(self, num_classes=2):
        super(GPT2, self).__init__()

        model_name = '../gpt2-medium'

        self.in_planes = 1024
        self.max_length = 64
        # 读取模型对应的tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # 载入模型
        self.model = transformers.GPT2Model.from_pretrained(model_name)

        self.linear = nn.Linear(self.in_planes, num_classes)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def tokenize(self, input_text):
        # padd or truncate to 64
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True, truncation=True, padding='max_length',
                                          max_length=self.max_length)
        # print(input_ids, len(input_ids))
        assert len(input_ids) == self.max_length
        return input_ids

    def forward(self, input_ids):

        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        out = self.model(input_ids, attention_mask=attention_mask)[0]  # Models outputs are now tuples
        out = torch.mean(out, 1)

        out = self.linear(out)

        return out


class GPT2_S(nn.Module):

    def __init__(self, num_classes=2):
        super(GPT2_S, self).__init__()

        model_name = '../gpt2'

        self.in_planes = 768
        self.max_length = 64
        # 读取模型对应的tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # 载入模型
        self.model = transformers.GPT2Model.from_pretrained(model_name)

        self.linear = nn.Linear(self.in_planes, num_classes)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def tokenize(self, input_text):
        # padd or truncate to 64
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=True, truncation=True, padding='max_length',
                                          max_length=self.max_length)
        # print(input_ids, len(input_ids))
        assert len(input_ids) == self.max_length
        return input_ids

    def forward(self, input_ids):

        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        out = self.model(input_ids, attention_mask=attention_mask)[0]  # Models outputs are now tuples
        out = torch.mean(out, 1)

        out = self.linear(out)

        return out