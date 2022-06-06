import pandas
import torch
import torchmetrics
from torch import nn
from Utility.Torch.Learnables import ContextTools
import seaborn as sns
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

def construct_mock_data():
    inputs = torch.randn([4000,1, 512])
    labels = torch.arange(0, 512, 4000)
    return inputs, labels




class PIU_Model(nn.Module):
    def __init__(self, defaults, dropout):
        super().__init__()

        embed_width = defaults['embedding_width']

        self.key = []
        self.value = []
        self.accuracy = []
        self.loss = []

        self.norm = nn.LayerNorm(embed_width)
        self.prediction = ContextTools.ParameterInjectionUnit(**defaults, dropout=dropout)
        self.final = nn.Linear(embed_width, 4000)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, tensor: torch.Tensor, labels: torch.Tensor):

        #Perform processing
        tensor = self.norm(tensor)
        tensor = self.prediction(tensor) + tensor
        tensor = self.final(tensor)
        tensor = tensor.mean(dim=-2)

        #Perform loss generation and metrics
        #(batch, embedding), (prediction, )
        loss = self.loss_func(tensor, labels)
        accuracy = torchmetrics.functional.accuracy(tensor, labels)

        self.accuracy.append(accuracy.clone().detach())
        self.loss.append(loss.clone().detach())
        self.key.append(self.prediction.Key.clone().detach())

        #return loss

        return loss


def trainer(data, model: PIU_Model, batches, batch_size):
        inputs, labels = data
        optim = torch.optim.Adam(model.parameters())
        for batch in range(batches):
            optim.zero_grad()

            # Select batch

            batch_labels = torch.randint(0, 4000, [batch_size])
            batch_inputs = inputs[batch_labels, :, :]

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
dropout = 0.8
setup = {}
setup['embedding_width'] = 512
setup['mem_width'] = 10
setup['heads'] = 128
setup['mode'] = "softmax"

model = PIU_Model(setup, dropout)
data = construct_mock_data()
model = trainer(data, model, batches, batch_size)
analyzer = Analyze(model)

print(analyzer.loss().head())
print(analyzer.loss().tail())
sns.lineplot(data=analyzer.accuracy())
plt.show()


sns.lineplot(data=analyzer.loss())
plt.show()