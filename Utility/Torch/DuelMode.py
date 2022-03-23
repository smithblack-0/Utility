class View():
    """
    Description:

    The view class contains code to extend
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

    --- attributes (in layer mode) ---
    input_shape
    output_shape

    --- methods ---

    functional(tensor, input_shape, output_shape):
        Returns an implicit view from input_shape to output_shape
    __call__ (in layer mode):
        In layer mode, returns a view.

    """

    @staticmethod
    def functional(tensor, input_shape, output_shape):
        """

        An improved form of pytorch's view. This will, when passed an input shape and output shape,
        fill in any missing dimensions from tensor's shape argument, then attempt to turn the last
        dimensions from input shape to output shape.


        """

        # Convertion

        if isinstance(input_shape, numbers.Number):
            input_shape = [input_shape]
        if isinstance(output_shape, numbers.Number):
            output_shape = [output_shape]

        # Basic sanity testing
        assert np.prod(input_shape) == np.prod(output_shape), "Input shape and output shape were not compatible"

        slice_length = len(input_shape)
        assert np.array_equal(input_shape, tensor.shape[-slice_length:]), "Input shape and tensor shape not compatible"

        # Construct view resize

        new_view = [*tensor.shape[:-slice_length], *output_shape]

        # View. Return

        return tensor.view(new_view)

    def __init__(self, input_shape, output_shape):
        assert np.prod(input_shape) == np.prod(output_shape), "Input  shape and output shape are incompatible"

        self.input_shape = input_shape
        self.output_shape = output_shape

    def __call__(self, tensor):
        return self.functional(tensor, self.input_shape, self.output_shape)


class Local():
    """

    Description:

    The Local class performs a strided, memory efficient,
    extract of information surrounding a particular index, much as
    you might see in preparation for convolution. It has two modes, and
    it will add dimensions.

    It has a functional mode, and a layer mode.

    Standard Convolutional termonagy such as kernel, stride_rate, and dilation rate is used to
    help keep the class familiar.

    ---- (attributes, layer mode), (functional, function mode)---

    kernel_width : how wide the kernel is
    stride_rate : how far forward the kernel moves for each local construction
    dilation_rate : from the start index of the kernel, how far each subsequent item in the kernel is
    pad : bool. D
    """

    @staticmethod
    def functional(tensor, kernel_width, stride_rate, dilation_rate, pad=False):
        """

        A function to produce local views of the last dimension of a tensor. These are
        views, indexed along the second to last dimension, with content along the last
        dimension, which are the precursors to convolution, possessing a kernel width,
        stride rate, and dilation rate as defined in the local view class.

        Enabling padding prevents information loss due to striding.

        """

        # Construct shape. Take into account the kernel_width, dilation rate, and stride rate.

        # The kernel width, and dilation rate, together modifies how far off the end of the
        # data buffer a naive implimentation would go, in an additive manner. Striding, meanwhile
        # is a multiplictive factor

        compensation = (kernel_width - 1) * dilation_rate  # calculate dilation-kernel correction
        final_index_shape = tensor.shape[-1] - compensation  # apply
        assert final_index_shape > 0, "Configuration is not possible - final kernel exceeds available tensors"
        final_index_shape = final_index_shape // stride_rate  # Perform striding correction.
        final_shape = (*tensor.shape[:-1], final_index_shape, kernel_width)  # Final shape

        # Construct the stride. The main worry here is to ensure that the dilation striding, and primary
        # striding, now occurs at the correct rate. This is done by taking the current one, multiplying,
        # and putting this in the appropriate location.

        final_stride = (*tensor.stride()[:-1], stride_rate * tensor.stride()[-1], dilation_rate * tensor.stride()[-1])

        # Pad tensor, if requested, to prevent loss of information. All padding goes on the end

        if pad:
            tensor = F.pad(tensor, (0, compensation))

            # Finish by striding and returning

        return tensor.as_strided(final_shape, final_stride)

