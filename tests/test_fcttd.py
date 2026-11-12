import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fcttd.layers import TuckerKruskalLinear, tucker_kruskal_parameter_breakdown

class FCTTDTest(unittest.TestCase):

    def test_parameter_counts_and_forward_without_reconstruction(self):
        self.assertEqual(tucker_kruskal_parameter_breakdown((16, 32, 49), (16, 16, 16), (8,) * 6, 90, False)['total_params'], 5480)
        self.assertEqual(tucker_kruskal_parameter_breakdown((16, 32, 49), (16, 16, 16), (8,) * 6, 400, False)['total_params'], 20360)
        layer = TuckerKruskalLinear((16, 32, 49), (16, 16, 16), (8,) * 6, 4)
        layer.reconstruct_weight = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('dense reconstruction used'))
        self.assertEqual(tuple(layer(torch.randn(2, 25088)).shape), (2, 4096))

class ContractionTest(unittest.TestCase):
    def test_factorized_forward_and_gradients_match_dense(self):
        torch.manual_seed(43)
        layer = TuckerKruskalLinear((2, 3, 2), (2, 2, 2), (2,) * 6, 3, dtype=torch.float64)
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.normal_(0, 0.3)
        x = torch.randn(4, 12, dtype=torch.float64, requires_grad=True)
        factorized = layer(x)
        dense = torch.nn.functional.linear(x, layer.reconstruct_weight(), layer.bias)
        torch.testing.assert_close(factorized, dense, rtol=1e-10, atol=1e-12)
        variables = (x,) + tuple(layer.parameters())
        factor_gradients = torch.autograd.grad(factorized.square().sum(), variables, retain_graph=True)
        dense_gradients = torch.autograd.grad(dense.square().sum(), variables)
        for first, second in zip(factor_gradients, dense_gradients):
            torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-12)
