# graph-encoder-llm (Full Repo Files)

> This document contains the full, copy-paste-ready contents for each file in the repository `graph-encoder-project/`.
> You can copy each file's content into the corresponding path in your local repo and push to GitHub.

---

== FILE: README.md ==

````markdown
# Graph Encoder → Adapter → LLM (Repo)

**Target configuration:** PyTorch + PyTorch Geometric (PyG); LLM via API (remote inference); Python scripts; minimal runnable demo included.

## Overview
This repository demonstrates a pipeline where a Graph Neural Network encodes PyG-format graphs into embeddings, a lightweight adapter (for local small LLM testing) is available, and a remote LLM API is called with graph-augmented prompts for downstream tasks (retrieval, QA, explanation).

## Quickstart
1. Create virtualenv and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

2. Prepare example data (edge list already included):

```bash
python src/build_graphs.py --input data/example/edge_list.csv --out data/example/graph.pt
```

3. Train a small graph encoder (toy demo):

```bash
python src/graph_encoder/train.py --data-path data/example/graph.pt --epochs 3 --out outputs/encoder.pth
```

4. Train adapter locally on a small HF model (optional):

```bash
python src/adapters/train_adapter.py --encoder-checkpoint outputs/encoder.pth --epochs 1 --out outputs/adapter.pth
```

5. Run inference (requires `OPENAI_API_KEY` environment variable):

```bash
export OPENAI_API_KEY="sk-..."
python src/inference/run_inference.py --graph data/example/graph.pt --adapter-checkpoint outputs/adapter.pth
```

## Project structure

See `src/` for all implementation code. Key modules:

* `src/build_graphs.py`: build PyG graphs from CSV edgelist
* `src/graph_encoder/`: GNN model, dataset wrapper, trainer
* `src/adapters/`: adapter modules and training script
* `src/llm/`: prompt builder & API client
* `src/inference/`: end-to-end inference script

## Extensions

* Add `src/gflownet/` for GFlowNet sampler
* Add `src/flow_matching/` for embedding-space flow pretraining

## License

MIT

```

---

== FILE: requirements.txt ==
```

torch>=2.0
torchvision
transformers>=4.30
torch-geometric
scipy
pandas
pyyaml
tqdm
requests
wandb

```

---

== FILE: data/example/edge_list.csv ==
```

src,dst
0,1
1,2
2,3
3,0
1,3

````

---

== FILE: src/build_graphs.py ==
```python
# src/build_graphs.py
import argparse
import torch
from torch_geometric.data import Data
import pandas as pd


def build_from_edgelist(edgelist_path, out_path):
    df = pd.read_csv(edgelist_path)
    src = torch.tensor(df['src'].values, dtype=torch.long)
    dst = torch.tensor(df['dst'].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    # Create simple node features: one-hot over max node id + 1 (toy)
    num_nodes = int(max(df['src'].max(), df['dst'].max()) + 1)
    x = torch.eye(num_nodes)
    data = Data(x=x, edge_index=edge_index)
    torch.save(data, out_path)
    print(f"Saved PyG Data to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    build_from_edgelist(args.input, args.out)
````

---

== FILE: src/graph_encoder/models.py ==

```python
# src/graph_encoder/models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class BaseEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # global pooling: use a single-graph pooling with dummy batch
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = global_mean_pool(x, batch)
        return g


class GATEncoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        g = global_mean_pool(x, batch)
        return g


# Flow-matching encoder placeholder: in this repo we implement a simple wrapper
class FlowMatchingEncoder(BaseEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # flow-matching specific params would go here

    def forward(self, data):
        # same forward for demo
        return super().forward(data)
```

---

== FILE: src/graph_encoder/dataset.py ==

```python
# src/graph_encoder/dataset.py
import torch
from torch_geometric.data import InMemoryDataset


class SingleGraphDataset(InMemoryDataset):
    """Wrap a single PyG Data object as a dataset with one element."""

    def __init__(self, data_path, transform=None):
        super().__init__('.', transform)
        self.data_path = data_path
        self.data = torch.load(data_path)
        self.data_list = [self.data]

    def len(self):
        return 1

    def get(self, idx):
        return self.data_list[idx]
```

---

== FILE: src/graph_encoder/train.py ==

```python
# src/graph_encoder/train.py
import argparse
import torch
from torch.utils.data import DataLoader
from src.graph_encoder.models import BaseEncoder
from src.graph_encoder.dataset import SingleGraphDataset


def train(data_path, epochs, lr, out_path, device='cpu'):
    dataset = SingleGraphDataset(data_path)
    loader = DataLoader(dataset, batch_size=1)
    data = dataset.get(0)
    in_dim = data.x.size(1)
    model = BaseEncoder(in_dim=in_dim)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            emb = model(batch)
            # toy self-supervised loss: make embedding norm close to 1
            loss = ((emb.norm(dim=1) - 1.0) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1}/{epochs} loss={loss.item():.4f}")
    torch.save(model.state_dict(), out_path)
    print(f"Saved encoder checkpoint to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', dest='out', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    train(args.data_path, args.epochs, args.lr, args.out, device=args.device)
```

---

== FILE: src/adapters/adapter_layers.py ==

```python
# src/adapters/adapter_layers.py
import torch
import torch.nn as nn


class CrossAttentionAdapter(nn.Module):
    def __init__(self, graph_dim, model_dim, num_heads=4):
        super().__init__()
        self.key_proj = nn.Linear(graph_dim, model_dim)
        self.to_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)

    def forward(self, model_tokens, graph_embeddings):
        # model_tokens: (B, T, D)
        # graph_embeddings: (B, Dg)
        keys = self.key_proj(graph_embeddings).unsqueeze(1)  # (B, 1, D)
        # use keys as K and V, queries are model_tokens
        attn_out, attn_weights = self.to_attn(query=model_tokens, key=keys, value=keys)
        return attn_out
```

---

== FILE: src/adapters/train_adapter.py ==

```python
# src/adapters/train_adapter.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.adapters.adapter_layers import CrossAttentionAdapter
from src.graph_encoder.models import BaseEncoder
from src.graph_encoder.dataset import SingleGraphDataset


def train_adapter(data_path, encoder_ckpt, hf_model_name, epochs, out_path, device='cpu'):
    # load graph encoder
    data = torch.load(data_path)
    in_dim = data.x.size(1)
    encoder = BaseEncoder(in_dim=in_dim)
    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    encoder.to(device)
    encoder.eval()
    # load small HF model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    model.to(device)
    # adapter
    graph_dim = 128
    model_dim = model.config.hidden_size
    adapter = CrossAttentionAdapter(graph_dim=graph_dim, model_dim=model_dim)
    adapter.to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    # build simple dataset prompt from graph
    prompt = f"Graph: nodes={data.x.size(0)} edges={data.edge_index.size(1)}
" \
             f"Question: What is the node count?"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    for epoch in range(epochs):
        with torch.no_grad():
            graph_emb = encoder(data.to(device))
        # expand tokens to batch
        tokens = inputs['input_ids']
        emb = model.transformer.wte(tokens)  # (B, T, D)
        out = adapter(emb, graph_emb)
        # project back and compute toy loss to encourage attention output magnitude
        loss = ((out.norm(dim=-1) - 1.0) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Epoch {epoch+1}/{epochs} adapter loss={loss.item():.4f}")
    torch.save(adapter.state_dict(), out_path)
    print(f"Saved adapter to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--encoder-checkpoint', required=True)
    parser.add_argument('--hf-model', default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    train_adapter(args.data_path, args.encoder_checkpoint, args.hf_model, args.epochs, args.out, device=args.device)
```

---

== FILE: src/llm/api_client.py ==

```python
# src/llm/api_client.py
import os
import requests

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API = 'https://api.openai.com/v1/chat/completions'


def call_openai_chat(messages, model='gpt-4o-mini'):
    if OPENAI_API_KEY is None:
        raise RuntimeError('OPENAI_API_KEY not set in environment')
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
    data = {
        'model': model,
        'messages': messages,
        'temperature': 0.0,
        'max_tokens': 512
    }
    resp = requests.post(OPENAI_API, json=data, headers=headers)
    resp.raise_for_status()
    return resp.json()
```

---

== FILE: src/llm/prompt_builder.py ==

```python
# src/llm/prompt_builder.py

def graph_to_prompt(graph_data, top_k_nodes=None):
    nodes = [f"N{i}" for i in range(graph_data.x.size(0))]
    src, dst = graph_data.edge_index
    edges = [f"N{s}->N{t}" for s, t in zip(src.tolist(), dst.tolist())]
    prompt = '<Graph>
Nodes: ' + ','.join(nodes) + '
Edges: ' + ','.join(edges) + '
</Graph>
'
    prompt += 'Task: Given the graph, answer the question truthfully and reference nodes as needed.'
    return prompt
```

---

== FILE: src/inference/run_inference.py ==

```python
# src/inference/run_inference.py
import argparse
import torch
from src.llm.prompt_builder import graph_to_prompt
from src.llm.api_client import call_openai_chat


def run(graph_path, model='gpt-4o-mini'):
    data = torch.load(graph_path)
    prompt = graph_to_prompt(data)
    messages = [{'role':'system','content':'You are a helpful assistant.'},
                {'role':'user','content':prompt + '
Question: How many nodes are in the graph?'}]
    resp = call_openai_chat(messages, model=model)
    print('LLM response:')
    try:
        print(resp['choices'][0]['message']['content'])
    except Exception:
        print(resp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True)
    parser.add_argument('--model', default='gpt-4o-mini')
    args = parser.parse_args()
    run(args.graph, model=args.model)
```

---

== FILE: scripts/train_flow_matching.sh ==

```
#!/bin/bash
python src/graph_encoder/train.py --data-path data/example/graph.pt --epochs 3 --out outputs/encoder.pth
```

---

== FILE: scripts/train_cgm.sh ==

```
#!/bin/bash
python src/adapters/train_adapter.py --data-path data/example/graph.pt --encoder-checkpoint outputs/encoder.pth --hf-model gpt2 --epochs 1 --out outputs/adapter.pth
```

---

== FILE: scripts/run_inference.sh ==

```
#!/bin/bash
export OPENAI_API_KEY="sk-..."
python src/inference/run_inference.py --graph data/example/graph.pt --model gpt-4o-mini
```

---

== FILE: experiments/exp1_flow_matching.md ==

```markdown
# Exp 1: Flow-matching encoder (toy)
- dataset: data/example/graph.pt
- model: BaseEncoder (toy)
- epochs: 3
- metrics: toy embedding-norm loss
```

---

== FILE: experiments/exp2_gflownet.md ==

```markdown
# Exp 2: GFlowNet prototype (placeholder)
- NOTE: Add GFlowNet code into src/gflownet/ if needed.
```

---

== FILE: experiments/exp3_cgm.md ==

```markdown
# Exp 3: CGM-style adapter integration
- dataset: data/example/graph.pt
- adapter: CrossAttentionAdapter
- hf-model used in demo: gpt2
```

---

== FILE: .gitignore ==

```
__pycache__/
venv/
outputs/
*.pth
*.pt
.env
```

---

End of files.

---

## Next steps I can take (pick one):

1. Generate `Dockerfile` + `environment.yml` for reproducibility.
2. Add a minimal GFlowNet prototype (`src/gflownet/`) with toy MDP and training script.
3. Implement a small unit test and GitHub Actions CI to run the toy training on push.

Tell me which one and I will add it into the repository textdoc.
