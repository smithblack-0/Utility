"""
A method to perform reshaping.

torch's native view is very powerful, but has a significant
limitation. One can not set a starting shape, target shape, and have
it reshape one to the other while leaving other dimensions alone. Being able to do this has applications
in, for example, transformers, and a variety of other situations.

This section rectifies this by providing additional view methods. FAFP is used to keep performance penalty minimal.
"""

#perform imports
import numpy as np
import torch





def _Autopsy(tensor, input_shape, output_shape):
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
    tensor + torch.zeros(input_shape)
  except:
    "Input shape and tensor shape are different"

  return "Could not find any additional information"


def _View(tensor, input_shape, output_shape):
  """
  A helper function to perform a view reshape. Does not contain
  error handling

  :param tensor: the tensor to reshape.
  :param input_shape: the shape to turn it into.
  :param output_shape: the shape to transform it from.
  :return:
  """

  slice_length = len(input_shape) #Find out how many dimensions will be dynamic
  static_shape, replacement_shape = tensor.shape[:-(slice_length)], tensor.shape[-(slice_length):] #Slice apart the input shape.
  reshape = (*static_shape, *output_shape) #replace the output shape
  return tensor.view(reshape) #reshape and return


#Define program interface
def View(tensor, input_shape, output_shape):
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

  #Turn ints into lists for further usage

  if not isinstance(input_shape, (list, tuple)):
    input_shape = [input_shape]
  if not isinstance(output_shape, (list, tuple)):
    output_shape = [output_shape]

  #Perform reshape. Apply FAFP error handling.
  try:
    return _View(tensor, input_shape, output_shape)
  except Exception as err:

      #Perform input verification, and figure out where the failure happened
      msg = "While attempting to reshape, I found an error: %s. \n" % err
      msg = msg + "Holmes adds that: 'Dear Watson, %s'" % _Autopsy(tensor, input_shape, output_shape)
      raise Exception(msg)