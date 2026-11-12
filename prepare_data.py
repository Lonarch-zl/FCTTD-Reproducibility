import hashlib, io, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
root = Path(__file__).resolve().parent / 'data'
root.mkdir(exist_ok=True)
files = {'train': ('8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883', 50000), 'test': ('841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de', 10000)}

def fetch(split):
    sha, n = files[split]
    path = root / (split + '.parquet')
    if not path.exists():
        url = f'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet'
        with urllib.request.urlopen(url, timeout=120) as r, path.open('wb') as out:
            while (data := r.read(1024 * 1024)):
                out.write(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    table = pq.read_table(path).to_pydict()
    labels = np.asarray(table['label'], dtype=np.int64)
    images = np.stack([np.asarray(Image.open(io.BytesIO(x['bytes'])).convert('RGB')) for x in table['img']])
    assert images.shape == (n, 32, 32, 3) and labels.shape == (n,)
    assert np.array_equal(np.bincount(labels), np.full(10, n // 10))
    np.savez(root / (split + '.npz'), images=images, labels=labels)
    print(split, images.shape, sha, flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(fetch, files))
(root / 'mirror_provenance.json').write_text(json.dumps({'source': 'https://huggingface.co/datasets/uoft-cs/cifar10', 'parquet_sha256': {k: v[0] for k, v in files.items()}, 'conversion': 'RGB uint8 images in original row order; original integer labels; no filtering'}, indent=2))
