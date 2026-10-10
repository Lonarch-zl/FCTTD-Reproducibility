#!/usr/bin/env python3
import argparse, json, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg16
from fcttd.layers import TuckerKruskalLinear

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',default='./data'); p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--fcttd',action='store_true'); p.add_argument('--cp-rank',type=int,default=400)
    a=p.parse_args(); device=torch.device(a.device)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); meta=ck.get('metadata',{})
    compressed=a.fcttd or meta.get('method')=='FCTTD'
    m=vgg16(weights=None); m.classifier[6]=torch.nn.Linear(4096,10)
    if compressed:
        m.classifier[0]=TuckerKruskalLinear(meta.get('in_shape',[16,32,49]),meta.get('out_shape',[16,16,16]),meta.get('tucker_ranks',[8]*6),meta.get('cp_rank',a.cp_rank))
    m.load_state_dict(ck.get('model_state',ck.get('state_dict',ck))); m.to(device).eval()
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    ds=datasets.CIFAR10(a.data_root,train=False,download=True,transform=tf); dl=DataLoader(ds,a.batch_size,shuffle=False,num_workers=2)
    n=correct=0
    with torch.no_grad():
        for x,y in dl:
            z=m(x.to(device)); correct += (z.argmax(1).cpu()==y).sum().item(); n += y.numel()
    print(json.dumps({'top1_accuracy':correct/n,'n':n,'checkpoint':a.checkpoint},indent=2))
if __name__=='__main__': main()
