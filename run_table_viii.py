#!/usr/bin/env python3
"""Print (or execute) the exact Table VIII rank/seed grid."""
import argparse, json, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); p.add_argument('--data-root',default='./data'); p.add_argument('--output-root',default='./runs/table_viii')
    a=p.parse_args(); cfg=json.load(open(Path(__file__).parent/'configs/table_viii.json'))
    for seed in cfg['seeds']:
      for rank in cfg['kruskal_ranks']:
        cmd=[sys.executable,'train_vgg16_cifar10.py','--model-source','torchvision','--data-root',a.data_root,'--seed',str(seed),'--fc1-out-shape',*map(str,cfg['tensorization']['out_shape']),'--fc1-in-shape',*map(str,cfg['tensorization']['in_shape']),'--fc1-tucker-ranks',*map(str,cfg['tucker_ranks']),'--fc1-cp-rank',str(rank),'--output-dir',str(Path(a.output_root)/f'r{rank}_seed{seed}')]
        print(' '.join(cmd))
        if a.execute: subprocess.run(cmd,check=True)
if __name__=='__main__': main()
