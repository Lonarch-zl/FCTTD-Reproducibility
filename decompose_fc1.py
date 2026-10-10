#!/usr/bin/env python3
"""Replace VGG-16 FC1 by an HOSVD/Tucker + CP/Kruskal layer."""
import argparse, json
from pathlib import Path
import torch
from torchvision.models import vgg16
from fcttd.layers import TuckerKruskalLinear, tucker_kruskal_parameter_breakdown

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dense-checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--in-shape', nargs=3, type=int, default=[16,32,49])
    p.add_argument('--out-shape', nargs=3, type=int, default=[16,16,16])
    p.add_argument('--tucker-ranks', nargs=6, type=int, default=[8]*6)
    p.add_argument('--cp-rank', type=int, default=400)
    p.add_argument('--tucker-iters', type=int, default=10)
    p.add_argument('--cp-iters', type=int, default=200)
    args = p.parse_args()
    torch.manual_seed(0)
    model = vgg16(weights=None); model.classifier[6] = torch.nn.Linear(4096, 10)
    state = torch.load(args.dense_checkpoint, map_location='cpu', weights_only=False)
    state = state.get('model_state', state.get('state_dict', state))
    model.load_state_dict(state)
    layer, stats = TuckerKruskalLinear.from_linear(
        model.classifier[0], args.in_shape, args.out_shape, args.tucker_ranks,
        args.cp_rank, decompose_device='cpu', n_iter_tucker=args.tucker_iters,
        n_iter_cp=args.cp_iters, return_stats=True)
    model.classifier[0] = layer
    meta = {'method':'FCTTD', 'cp_rank':args.cp_rank, 'in_shape':args.in_shape,
            'out_shape':args.out_shape, 'tucker_ranks':args.tucker_ranks,
            'parameter_breakdown': tucker_kruskal_parameter_breakdown(
                args.in_shape,args.out_shape,args.tucker_ranks,args.cp_rank,bias=True),
            'decomposition_stats':stats}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'metadata': meta}, args.output)
    print(json.dumps(meta, indent=2))

if __name__ == '__main__': main()
