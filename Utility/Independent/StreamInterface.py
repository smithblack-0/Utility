class StreamInterface():
    """
    A class to transparently allow editing items in a complex complex data stream
    as though they are in a list.

    This class traverses the data structure, identifies using the provided conditional
    leafs which are of interest, and provides a clean interface to edit each of them,
    in sequence, as though it were a list.

    By default the class can take apart and edit lists, dictionaries, and tuples.
    However, proper configuration will allow it to handle much more.

    ------ Usage Methods -----

    __init__: Creates an instance. Must be provided a stream, a leaf conditional,
               and a leaf copy function.
    __getitem__ : Allows accessing by index as in StreamInterface[:].
                  Shows off only valid leafs. Slicing supported.
    __setitem__ : Allows setting by index as in StreamInterface[0] = 5.
                  Shows off only valid leafs. Python slicing supported
    __len__ : Shows the length of the known leafs
    __call__ : Returns the stream in it's original format.

    clone : Creates a copy of the current stream.

    ------ Overriddable methods -----

    The three overriddable methods are parse, access, and set. These are
    responsible for, respectively, figuring out for a given data object what
    the access keys are, performing the access using the key, and performing
    a set using the key and a value.

    All three must be implimented for a data type to be supported. Overriding them,
    particularly when retaining a super call for standard support, allows easy
    extension to additional data types.

    parse(self, data_structure) :
            The return here should be a key, value pair as in a dictionary
            if the value is supported. If not, nothing or None should be returned
    access(self, data_structure, key) :
            The return should access the data structure entry at key then return
             the accessed value
    set(self, data_structure, key, value):
            The structure here should go and set the item at value key
            in data_structure to value. Whatever that means. Then it
            should return the modified data structure.

    ### Example ###
    If you wanted to support getting data from a tree stream, it would be
    possible. One could go and do something decent, with node type NODE,
    and children stored in a list at node.children, something like as follows:

    def parse(self, data):
      super().parse(data)
      if isinstance(data, Node):
        indices = range(len(data.children))
        return indices, data.children
    def access(self, data, key):
      super().parse(data)
      if isinstance(data, Node):
        return data.children[key]
    def set(self, data, key, value)
        data.remove_child(key)
        data.insert_child(key, value)
        return data



    """

    # User implimented methods
    def parse(self, data_structure):
        """
        Parse. Takes a stream item, representing some sort of mutable data
        structure, and returns key-value pairs for each entry of the structure.

        The keys are used by access to access the data.
        """
        if isinstance(data_structure, dict):
            return data_structure.keys(), data_structure.values()
        if isinstance(data_structure, list):
            return list(range(len(data_structure))), data_structure
        if isinstance(data_structure, tuple):
            return list(range(len(data_structure))), data_structure

    def access(self, data_structure, key):
        """
        This is the method responsible for accessing an item based on the
        key value returned. Can be modified to traverse trees, etc.
    
        The keys are defined by parse, and used to access the data
        """
        if isinstance(data_structure, dict):
            return data_structure[key]
        if isinstance(data_structure, list):
            return data_structure[key]
        if isinstance(data_structure, tuple):
            return data_structure[key]

    def set(self, data_structure, key, value):
        """
        This method is responsible for setting
        a new value with a given key.

        It should be overridden for interfacing with
        objects
        """
        if isinstance(data_structure, dict):
            data_structure[key] = value
            return data_structure
        if isinstance(data_structure, list):
            data_structure[key] = value
            return data_structure
        if isinstance(data_structure, tuple):
            # Tuples are immutable. Convert to list, perform set, convert back.
            data_structure = list(data_structure)
            data_structure[key] = value
            data_structure = tuple(data_structure)
            return data_structure

    # Internal Logic
    class Address():
        """
        Tiny data class. Mainly exists so I know what an address is vs a
        list of addresses
        """

        def __init__(self, address):
            self.address = address

        def __call__(self):
            return self.address.copy()

    def make_addresses(self, stream, leaf_condition, stack=[]):
        processed_structure = self.parse(stream)
        if processed_structure is None:
            # No match was found. End recursion.
            if leaf_condition(stream):
                # A tensor was matched. Return address
                return [self.Address(stack.copy())]
            else:
                # No tensor. Return empty address - no need to worry about this.
                return []

        # Structure was matched, and unwound.
        addresses = []
        keys, values = processed_structure
        for key, value in zip(keys, values):
            # Increase the stack, to accomodate for the new entry
            stack.append(key)

            # Get the output. Append it to the addresses
            new_addresses = self.make_addresses(value, stack)
            addresses = addresses + new_addresses

            # Pop off the new entry
            stack.pop()
        return addresses

    def access_address(self, address):
        """
        Access an address in a stream
        """
        # Copy address so we don't pop off things
        # from the master
        address = address()

        # Define recursive access
        def access(stream, address):
            if len(address) == 0:
                return stream
            key = address.pop(0)
            return access(self.access(stream, key), address)

        # Perform access
        return access(self._stream, address)

    def set_address(self, address, value):
        # Get list copy
        address = address()

        # Define recursive access
        def access(stream, address):
            # At the end
            if len(address) == 0:
                return True

            # Recursion not over
            key = address.pop(0)
            output = access(stream[key], address)
            if output is True:
                stream = self.set(stream, key, value)
            return stream

        self._stream = access(self._stream, address)

    def __init__(self, stream, leaf_condition, leaf_copy):

        # Make addresses, which tell in sequence where mutable tensor objects are found.
        self._stream = stream
        self._copy = leaf_copy
        self._addresses = self.make_addresses(stream, leaf_condition)

    def __len__(self):
        """
        Return the length indicating the number of accessable tensors
        """
        return len(self._addresses)

    def __getitem__(self, key):
        """
        Gets one of the modifiable tensors.
        """
        # Get appropriate addresses. Then use the addresses to get the values
        addresses = self._addresses[key]
        if not isinstance(addresses, list):
            return self.access_address(addresses)

        output = [self.access_address(address) for address in addresses]
        return output

    def __setitem__(self, key, values):
        """
        Sets a tensor in the series to something
        """
        addresses = self._addresses[key]
        if not isinstance(addresses, list):
            addresses = [addresses]
        if not isinstance(values, list):
            values = [values]
        for address, value in zip(addresses, values):
            self.set_address(address, value)

        return self

    def __call__(self):
        return self._stream

    ### User Methods ###
    def clone(self):
        """
        Creates an independent copy of the current stream.
        """

        # Make new instance
        instance = copy.deepcopy(self)

        # Copy tensors into new instance to preserve any dynamic memory
        # links.
        for index in range(len(instance)):
            instance[index] = self[index].clone()
