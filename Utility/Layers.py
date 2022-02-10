

# perform imports
import numpy as np
import torch


#####
# DEFINE VIEW TOOL
####

class View():
    """
    The View Class. The view layer contains code to extend
    the standard torch view mode. It is capable of reshaping the
    last dimension of a tensors shape from one shape to another without
    changing any prior dimensions, so long as the number of tensor entries
    match.

    The View Class has two modes. In functional mode, one may provide
    a tensor, input_shape, and output_shape to the method .functional. This
    will then execute the reshape logic. Alternatively, one can instance
    the class with input_shape and output_shape, in which case it will behave
    as a reshaping layer and consistently apply the same reshape.

    For example, if one wanted to reshape tensor t=(4, 3, 4,2), to become (4,3,8),
    one could either initialize a layer with View((4,2), 8), then call it,
    or apply View.functional(t, (4,2), 8)
    """

    ## Define the helper functions which functionally perform the view
    @classmethod
    def _view(cls, tensor, input_shape, output_shape):
        """
        A function to change a tensor with last dimensions
        input shape to last dimensions output shape.

        For instance, given (4, 3, 4,2), reshape (4,2) to (8)
        resulting in (4, 3, 8). Expects tensor to be tensorflow vectors.



        :param tensor: The tensor to reshape
        :param input_shape: The shape of the input dimensions which we wish to handle. Can be an integer, or a list
          of integers.
        :param output_shape: The shape of the output dimensions we wish to end up in
        :return: The reshaped tensor.
        """

        # Perform reshape. Apply FAFP error handling.
        try:
            slice_length = len(input_shape)  # Find out how many dimensions will be dynamic
            static_shape, replacement_shape = tensor.shape[:-(slice_length)], tensor.shape[-(
                slice_length):]  # Slice apart the input shape.
            reshape = (*static_shape, *output_shape)  # replace the output shape
            return tensor.view(reshape)  # reshape and return
        except Exception as err:

            # Perform input verification, and figure out where the failure happened
            msg = "While attempting to reshape, I found an error: %s. \n" % err
            msg = msg + "Holmes adds that: 'Dear Watson, %s'" % cls._autopsy(tensor, input_shape, output_shape)
            raise Exception(msg)

    @classmethod
    def _autopsy(cls, tensor, input_shape, output_shape):
        """
        A helper function to analyze what went wrong after an exception is raised

        :param tensor: The input tensor
        :param input_shape: The input shape
        :param output_shape: The output shape
        :return: A string representing anything I found that is wrong.
        """

        if not torch.is_tensor(tensor):
            return "tensor input was not a torch tensor"

        try:
            input_shape = np.array(input_shape, dtype=np.int32)
            output_shape = np.array(output_shape, dtype=np.int32)
        except:
            return "Either input or output shape were unclean. They did not cast to int array"

        if len(input_shape.shape) > 1:
            return "Input shape had more dimensions than one. Should be an int, or list of ints"
        if len(output_shape.shape) > 1:
            return "Output shape had more dimensions than one. Should be an int, or list of ints"

        if np.prod(input_shape) != np.prod(output_shape):
            return "The number of input tensor units was %s, and out tensor units was %s. These are not compatible" \
                   % (np.prod(input_shape), np.prod(output_shape))

        try:
            tensor + torch.zeros([*input_shape])
        except:
            "Input shape and tensor shape are different"

        return "Could not find any additional information"

    # functional connector
    @classmethod
    def functional(cls, tensor, input_shape, output_shape):
        """
        A function to change a tensor with last dimensions
        input shape to last dimensions output shape.

        For instance, given (4, 3, 4,2), reshape (4,2) to (8)
        resulting in (4, 3, 8). Expects tensor to be tensorflow vectors.



        :param tensor: The tensor to reshape
        :param input_shape: The shape of the input dimensions which we wish to handle. Can be an integer, or a list
          of integers.
        :param output_shape: The shape of the output dimensions we wish to end up in
        :return: The reshaped tensor.
        """

        return cls._view(tensor, input_shape, output_shape)

    ### Develop the stateful logic for that version of access
    def __init__(self, input_shape, output_shape):
        self._input_shape = input_shape
        self._output_shape = output_shape

    def __call__(self, tensor):
        return self.functional(tensor, self._input_shape, self._output_shape)