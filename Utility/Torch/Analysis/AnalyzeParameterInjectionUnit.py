import pandas
import torch
import torchmetrics
from torch import nn

import Utility.Torch.Learnables.Attention
from Utility.Torch.Learnables import ContextTools
import seaborn as sns
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

def construct_mock_data(total_samples, subsamples):
    sublabels = torch.arange(0, subsamples)
    labels = torch.multinomial(sublabels.type(torch.float32), total_samples, replacement=True).type(torch.long)
    inputs = torch.randn([total_samples,1, 512])

    return inputs, labels




class PIU_Model(nn.Module):
    def __init__(self, defaults, dropout, total_samples):
        super().__init__()

        embed_width = defaults['embedding_width']

        self.key = []
        self.value = []
        self.accuracy = []
        self.loss = []

        self.norm = nn.LayerNorm(embed_width)
        self.prediction = Utility.Torch.Learnables.Attention.PIMU(**defaults)
        self.dropout = nn.Dropout(dropout)
        self.final = nn.Linear(embed_width, total_samples)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, tensor: torch.Tensor, labels: torch.Tensor):

        #Perform processing
        tensor = self.norm(tensor)
        tensor = self.dropout(self.prediction(tensor)) + tensor
        tensor = self.final(tensor)
        tensor = tensor.mean(dim=-2)

        #Perform loss generation and metrics
        loss = self.loss_func(tensor, labels)

        self.loss.append(loss.clone().detach())
        self.key.append(self.prediction.Key.clone().detach())
        self.accuracy.append(torchmetrics.functional.accuracy(tensor, labels))
        #return loss

        return loss


def trainer(data, model: PIU_Model, batches, batch_size, options):
        inputs, labels = data
        optim = torch.optim.Adam(model.parameters())
        for batch in range(batches):
            optim.zero_grad()

            # Select batch

            batch_choice = torch.randint(0, options, [batch_size])
            batch_labels = labels[batch_choice]
            batch_inputs = inputs[batch_choice, :, :]

            #Learn
            loss = model(batch_inputs, batch_labels)


            loss.backward()
            optim.step()
        return model


class Analyze:
    def accuracy(self):
        frame_data = {}
        frame_data['accuracy'] = torch.stack(self.model.accuracy)
        return pandas.DataFrame(frame_data)
    def key_norm(self):
        key_data = torch.stack(self.model.key)
        key_data = torch.linalg.vector_norm(key_data, dim=-1, ord=2)
        frame_data = {}
        for head in range(key_data.shape[-2]):
            for channel in range(key_data.shape[-1]):
                msg = "key magnitudes: " + "head " + str(head) + ", channel " + str(channel)
                frame_data[msg] = key_data[..., head, channel]
        return pandas.DataFrame(frame_data)
    def loss(self):
        frame_data = {}
        frame_data['loss'] = torch.stack(self.model.loss)
        return pandas.DataFrame(frame_data)
    def __init__(self, model: PIU_Model):
        self.model = model




batch_size = 64
batches = 800
dropout = 0.4
setup = {}
options = 10000
suboptions = 400

setup['embedding_width'] = 512
setup['mem_width'] = 20
setup['heads'] = 128
setup['mode'] = "softmax"

model = PIU_Model(setup, dropout, options)
data = construct_mock_data(options, suboptions)
model = trainer(data, model, batches, batch_size, options)
analyzer = Analyze(model)

print(analyzer.loss().head())
print(analyzer.loss().tail())

sns.lineplot(data=analyzer.accuracy())
plt.show()

sns.lineplot(data=analyzer.loss())
plt.show()