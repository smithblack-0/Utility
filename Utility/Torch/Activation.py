import torch


def vl_relu(tensor, alpha =0.01):
    """

    The Virtual Leaky ReLU acts like a ReLU
    on the forward pass. However, on the backward pass,
    it instead acts as a leaky relu, passing through
    gradients multiplied by the leak constant.

    This should hopefully prevent stuck nodes, while
    dealing with the principle problem of leaky
    relu's - sensitivity to unusual data.
    """

    #Define a variable. It exists only to serve as an indirect reference
    indirect = []

    #Define the pre and post functions. These serve
    #to capture and restore grads
    def pre(grad):
        """ Capture gradients before running through the relu."""
        indirect.append(grad)
    def post(dead_grads):
        """ Revive gradients crushed by the relu, pretending it is a leaky relu"""

        intercepted_grads = indirect.pop()
        final_grads = torch.where(dead_grads == 0, alpha*intercepted_grads, dead_grads)
        return final_grads
    #Attach pre and post functions to the appropriate places. Perform relu. Return
    if tensor.requires_grad:
        tensor.register_hook(post)
        tensor = torch.nn.functional.relu(tensor)
        tensor.register_hook(pre)
    else:
        tensor = torch.nn.functional.relu(tensor)

    return tensor
