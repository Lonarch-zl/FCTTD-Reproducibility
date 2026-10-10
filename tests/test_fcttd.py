import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from fcttd.layers import TuckerKruskalLinear, tucker_kruskal_parameter_breakdown

class FCTTDTest(unittest.TestCase):
  def test_parameter_counts_and_forward_without_reconstruction(self):
    self.assertEqual(tucker_kruskal_parameter_breakdown((16,32,49),(16,16,16),(8,)*6,90,False)['total_params'], 5480)
    self.assertEqual(tucker_kruskal_parameter_breakdown((16,32,49),(16,16,16),(8,)*6,400,False)['total_params'], 20360)
    layer=TuckerKruskalLinear((16,32,49),(16,16,16),(8,)*6,4)
    layer.reconstruct_weight=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('dense reconstruction used'))
    self.assertEqual(tuple(layer(torch.randn(2,25088)).shape), (2,4096))
