class DKernel():
    """

    Dynamically growable kernel. The DKernel
    is defined at setup to possess a number
    of dimensions, and an initialization
    function.

    It then becomes an item which can be queried
    by slicing and will return an appropriately shaped
    slice, with values which have never been accessed
    being initialized by the function.

    Attempting to perform
    -- methods ---

    add

    """

