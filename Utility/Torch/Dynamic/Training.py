"""

The holder for  trainers

"""

import torch
import asyncio

class Manager():
    """

    The base manager class

    * Sits above a collection of trainers, called a population
    *

    """



class Trainer():
    """

    The base trainer class.

    * Sits above the model
    * Responsible for building and returning an identical copy of itself when requested.
    * Responsible for managing stateful provisioning logic.
    * Responsible for gradient descent, loss, and intermediary boosting.
    * Responsible for interfacing with the model.

    * An observer of PolicyUpdate.
    * Pushes reward to PolicyUpdate.
    * Updates policies based on callback from PolicyUpdate

    """
    def __init__(self, model, policy_manager, data_loader):
        """
        The initialization process sets up a group of parallel population entities
        using the async functions if provided.

        :param model: The DModule model we are constructing from. This should have a working
            save and load method.
        """

        self.data = data_loader
        self.model = model

