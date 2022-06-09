from typing import Optional, List

from torch import nn

from Utility.Torch.Models.SupertransformerOld.StreamTools import StreamTensor


class AbstractResStartup(nn.Module):
    """


    """
    def __init__(self):
        super().__init__()
    def forward(self,
                res_stream: Optional[List[StreamTensor]],
                auxiliary_stream: Optional[List[StreamTensor]]
                ) -> List[StreamTensor]:
        raise NotImplementedError()