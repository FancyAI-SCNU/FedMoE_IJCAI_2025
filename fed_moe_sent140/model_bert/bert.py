import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer


class BertNet(nn.Module):

    def __init__(self, num_classes=2):
        super(BertNet, self).__init__()

        model_name = 'bert-base-uncased'

        self.in_planes = 768
        self.max_length = 64
        # 读取模型对应的tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # 载入模型
        self.model = BertModel.from_pretrained(model_name)

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
        # input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
        # input_ids = torch.tensor([input_ids]).to(device)
        out = self.model(input_ids)[0]  # Models outputs are now tuples
        out = torch.mean(out, 1)

        out = self.linear(out)
        prob = self.softmax(out)

        return out, prob


if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    input_text = "Here is some text to encode"
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]

    print(last_hidden_states.size())
