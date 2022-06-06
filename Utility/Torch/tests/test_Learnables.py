import asyncio
import unittest
import torch
import torchmetrics
import seaborn
import pandas

from matplotlib import pyplot as plt
from torch import nn
from Utility.Torch.Learnables import Layers
from Utility.Torch.Learnables import ContextTools

class testLinear(unittest.TestCase):
    """
    This is the test feature for the linear layer.
    """
    def test_Regular(self):
        """ Tests if the standard pytorch linear layer is reproduced"""

        tensor = torch.rand([33, 2, 5])
        tester = Layers.Linear(5, 10)
        test = tester(tensor)
        self.assertTrue(test.shape[-1] == 10, "Regular pytorch layer not reproduced")
    def test_Reshapes(self):
        """ Tests whether the reshape functionality is working in isolation """
        #Define test tensor
        tensor = torch.rand([30, 20, 15])

        #Define test layers
        test_expansion = Layers.Linear(15, [5, 3])
        test_collapse = Layers.Linear([20, 15], 300)
        test_both = Layers.Linear([20, 15], [10, 30])

        #Perform tests

        test_expansion_result = test_expansion(tensor)
        test_collapse_result = test_collapse(tensor)
        test_both_result = test_both(tensor)

        expansion_bool = [*test_expansion_result.shape] == [30, 20, 5, 3]
        collapse_bool = [*test_collapse_result.shape] == [30, 300]
        both_bool = [*test_both_result.shape] == [30, 10, 30]

        #Assert results
        self.assertTrue(expansion_bool, "Reshape: Expansion failed")
        self.assertTrue(collapse_bool, "Reshape: collapse failed")
        self.assertTrue(both_bool, "Reshape: Compound failed")
    def test_Heading(self):
        """ Tests whether the head kernels and bias are implemented such that calling works"""

        tensor = torch.randn([10, 30, 20, 10])

        #Create test layers

        test_single = Layers.Linear(10, 20, 20)
        test_multiple = Layers.Linear(10, 20, (30, 20))

        #Run tests

        test_single_result = test_single(tensor)
        test_multiple_result = test_multiple(tensor)

    def test_Head_Independence(self):
        """ Tests whether each head is completely independent"""
        
        #Create tensors
        tensor_a = torch.stack([torch.zeros([20]), torch.zeros([20])])
        tensor_b = torch.stack([torch.zeros([20]), torch.ones([20])])
        
        #create tester
        
        test_head_independence = Layers.Linear(20, 20, 2)
        
        #Run tests
        
        test_result_a = test_head_independence(tensor_a)
        test_result_b = test_head_independence(tensor_b)
        
        #Analyze and assert result
        result_bool = torch.all(test_result_a[0] == test_result_b[0])
        self.assertTrue(result_bool, "Heads were found to be interacting")
    def test_gradients(self):
        """Test whether or not gradients are propogating properly"""
        test_tensor = torch.randn([20, 10])
        
        #Develop test layer
        test_grad = Layers.Linear([20, 10], 1)

        #Develop optim
        test_optim = torch.optim.SGD(test_grad.parameters(), lr=0.01)

        #perform test
        test_result = test_grad(test_tensor)
        test_result.backward()
        
        test_optim.step()
    def test_jit_basic(self):
        """ Test whether or not the module is scriptable when instanced"""
        # Develop test layer
        test_tensor = torch.randn([30, 20, 20])
        test_script = Layers.Linear(20, 10, 1)

        #Perform test
        scripted = torch.jit.script(test_script)
        scripted(test_tensor)

class testBandedAttn(unittest.TestCase):
    def testBasic(self):
        """ Test whether the defaults are working """
        query = torch.arange(96).view(1, 4, 24).type(torch.float32)
        key = query.clone()
        value = query.clone()

        tester = Layers.BandedMultiheadedAttention(24, 3)
        tester(query, key, value)
    def testCompressive(self):
        """ Test whether or not compression and expansion abilities are functioning"""
        query = torch.randn([2,3,80,80])
        key1 = torch.randn([2, 3, 40, 80])
        key2 = torch.randn([2, 3, 160, 80])
        value1 = key1.detach()
        value2 = key2.detach()

        tester_decompress = Layers.BandedMultiheadedAttention(80, 20, compression_ratio=(1, 2))
        tester_compress = Layers.BandedMultiheadedAttention(80, 20, compression_ratio=(2, 1))

        test_decompress = tester_decompress(query, key2, value2)
        test_compress = tester_compress(query, key1, value1)
    def testGradients(self):
        """ Tests that gradients are updating when a layer is run """
        query = torch.arange(96).view(1, 4, 24).type(torch.float32)
        key = query.clone()
        value = query.clone()

        tester = Layers.BandedMultiheadedAttention(24, 3)
        test_optim = torch.optim.Adam(tester.parameters())
        params = [item.clone().detach() for item in tester.parameters()]

        for _ in range(10):
            test_optim.zero_grad()
            test_result = tester(query, key, value)
            loss = test_result.sum()**2

            loss.backward()
            test_optim.step()

        for initial, final in zip(params, tester.parameters()):
            test = torch.not_equal(initial, final)
            test = torch.any(test)
            self.assertTrue(test, "Parameters not updating")
    def test_Saveable(self):
        """ Tests whether or not saving, then loading, works properly"""

        query = torch.arange(96).view(1, 4, 24).type(torch.float32)
        key = query.clone()
        value = query.clone()

        tester = Layers.BandedMultiheadedAttention(24, 3)
        tester2 = Layers.BandedMultiheadedAttention(24, 3)

        state_dict = tester.state_dict()
        tester2.load_state_dict(state_dict)

        resulta = tester(query, key, value)
        resultb = tester2(query, key, value)
        self.assertTrue(torch.equal(resulta, resultb))

    def test_jit(self):
        """Tests whether or not the layer can be jit compiled. """
        query = torch.arange(96).view(1, 4, 24).type(torch.float32)
        key = query.clone()
        value = query.clone()

        tester = Layers.BandedMultiheadedAttention(24, 3)
        tester = torch.jit.script(tester)
        tester(query, key, value)

class test_NIFCU(unittest.TestCase):
    """
    This is the test case for the
    NIFCU layer.

    """
    show_graphs = False
    def setUp(self):
        defaults = {}
        defaults['embedding_width'] = 512
        defaults['memory_width'] = 6
        defaults['heads'] = 4
        defaults['integration_lr'] = 0.001
        defaults['decay_rate'] = 0.999
        defaults['dropout'] = 0.95
        self.defaults = defaults
    def test_constructor(self):
        layer = ContextTools.AutoCalibrationInjector(**self.defaults)
    def test_basic_forward(self):
        tensor = torch.randn([10, 30, 512])
        layer = ContextTools.AutoCalibrationInjector(**self.defaults)
        output = layer(tensor)
    def test_torchscript_forward(self):
        tensor = torch.randn([10, 30, 512])
        layer = ContextTools.AutoCalibrationInjector(**self.defaults)
        layer = torch.jit.script(layer)
        output = layer(tensor)

    def test_dev(self):
        layernorm = nn.LayerNorm(512)

        inputs = layernorm(torch.randn([512, 1, 512])).detach()
        all_labels = torch.arange(0, 512, dtype=torch.long)
        layer = ContextTools.AutoCalibrationInjector(**self.defaults)
        final = nn.Linear(512, 512)

        batches = 1600
        batch_size = 64
        optim = torch.optim.Adam(layer.parameters(), lr=0.001)
        loss_func = nn.CrossEntropyLoss()

        #Starting loss

        prediction = layer(inputs)
        prediction = final(prediction).squeeze()
        loss = loss_func(prediction, all_labels)

        #Train
        calibration = []
        activity = []
        accuracy = []
        key = []
        for i in range(batches):
            optim.zero_grad()
            pre_calibration = layer.CalibrationBuffer.clone().detach()
            pre_activity = layer.activity.clone().detach()

            choice = torch.randint(0, 512, [batch_size])
            labels = all_labels[choice]
            batch = inputs[choice, :, :]

            prediction = layer(batch)
            prediction = final(prediction).squeeze()
            loss = loss_func(prediction, labels)

            loss.backward()
            optim.step()


            calibration.append(layer.CalibrationBuffer.abs().mean(dim=-1).clone().detach())
            activity.append(layer.curiosity.abs().clone().detach())
            key.append(layer.KeyParameters.abs().mean(dim=-1).clone().detach())
            accuracy.append(torchmetrics.functional.accuracy(prediction, labels))

        for i in range(batches):

            choice = torch.randint(0, 512, [batch_size])
            labels = all_labels[choice]
            batch = inputs[choice, :, :]

            prediction = layer(batch)
            prediction = final(prediction).squeeze()
            loss = loss_func(prediction, labels)

            calibration.append(layer.CalibrationBuffer.abs().mean(dim=-1).clone().detach())
            activity.append(layer.curiosity.abs().clone().detach())
            key.append(layer.KeyParameters.abs().mean(dim=-1).clone().detach())
            accuracy.append(torchmetrics.functional.accuracy(prediction, labels))

        #Eval

        prediction = layer(inputs)
        prediction = final(prediction).squeeze()
        loss = loss_func(prediction, all_labels)

        print(loss)
        print(torchmetrics.functional.accuracy(prediction, all_labels))

        calibration = torch.stack(calibration).flatten(1, 2)
        activity = torch.stack(activity).flatten(1,2)
        key = torch.stack(key).flatten(1, 2)
        cal_to_key_ratio = (calibration/key)


        calibration_names = ["calibration channel " + str(item) for item in range(calibration.shape[-1])]
        calibration = dict(zip(calibration_names, calibration.transpose(-1, -2)))

        activity_name = ["activity_channel " + str(item) for item in range(activity.shape[-1])]
        activity = dict(zip(activity_name, activity.transpose(-1, -2)))

        key_names = ["key channel " + str(item) for item in range(key.shape[-1])]
        key = dict(zip(key_names, key.transpose(-1, -2)))

        cal_key_ratio_names = ["cal key " + str(item) for item in range(cal_to_key_ratio.shape[-1])]
        cal_to_key_ratio = dict(zip(cal_key_ratio_names, cal_to_key_ratio.transpose(-1, -2)))


        frame = pandas.DataFrame(calibration)
        plot = seaborn.lineplot(data=frame)
        plt.show()

        frame = pandas.DataFrame(activity)
        plot = seaborn.lineplot(data=frame)
        plt.show()

        frame = pandas.DataFrame(key)
        plot = seaborn.lineplot(data=frame)
        plt.show()

        frame = pandas.DataFrame(cal_to_key_ratio)
        plot = seaborn.lineplot(data=frame)
        plt.show()

        frame = pandas.DataFrame({"accuracy" : torch.stack(accuracy)})
        plot = seaborn.lineplot(data=frame)
        plt.show()

if __name__ == "__main__":
    unittest.main()