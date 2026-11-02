"""Single-GPU BF16 model engine; FP32 CE and CPU gradient accumulation."""
import os, json, random, gc
from pathlib import Path
R=Path(os.environ.get('GRPO_RUN_DIR','run')).expanduser().resolve()
for k,v in {'HF_HOME':'cache/hf','XDG_CACHE_HOME':'cache','TORCH_HOME':'cache/torch','TRITON_CACHE_DIR':'cache/triton','TMPDIR':'tmp','CUDA_CACHE_PATH':'cache/cuda'}.items():
 os.environ[k]=str(R/v);(R/v).mkdir(parents=True,exist_ok=True)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from PIL import Image

def status(**kw):
 print(json.dumps(kw),flush=True)

def write(path,obj):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def cleanup():
 gc.collect();torch.cuda.empty_cache()

def reduce2(a,b):
 # FP64 diagnostics only, fixed tensor order and chunk size.
 ans=0.
 for x,y in zip(a.reshape(-1).split(1048576),b.reshape(-1).split(1048576)):
  ans+=float(torch.dot(x.double(),y.double()))
 return ans

def wrap_checkpoint(module):
 orig=module.forward
 def forward(*args,**kwargs):
  if torch.is_grad_enabled():return checkpoint(orig,*args,use_reentrant=False,**kwargs)
  return orig(*args,**kwargs)
 module.forward=forward

class Engine:

    def __init__(self, precision):
        self.arm = os.environ.get('QA_ARM', 'image_full')
        self.precision = precision
        self.dtype = torch.float32 if precision == 'fp32' else torch.bfloat16
        torch.set_num_threads(8)
        torch.manual_seed(20261022)
        random.seed(20261022)
        np.random.seed(20261022)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.processor = AutoProcessor.from_pretrained(R / 'model', local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(R / 'model', dtype=self.dtype, device_map='cuda', attn_implementation='sdpa', local_files_only=True).eval()
        self.model.requires_grad_(True)
        self.params = list(self.model.named_parameters())
        self.dest = None
        self.eos = self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        for module in [*self.model.model.visual.blocks, *self.model.model.language_model.layers]:
            wrap_checkpoint(module)
        for name, p in self.params:

            def hook(param, name=name):
                if self.dest is None:
                    raise RuntimeError('Untracked gradient')
                grad = param.grad.detach().to(device='cpu', dtype=torch.float32)
                self.dest[name].add_(grad, alpha=self.scale)
                param.grad = None
            p.register_post_accumulate_grad_hook(hook)
        self.inputs_cache = {}
        if self.arm != 'image_full':
            for n, p in self.params:
                if n.startswith('model.visual.'):
                    p.requires_grad_(False)
            self.params = [(n, p) for n, p in self.params if p.requires_grad]

    def inputs(self, row):
        if row['id'] not in self.inputs_cache:
            if not row.get('image'):
                messages = [{'role': 'user', 'content': [{'type': 'text', 'text': row['prompt']}]}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                self.inputs_cache[row['id']] = self.processor(text=[text], return_tensors='pt')
                return self.inputs_cache[row['id']].to('cuda')
            im = Image.open(row['image']).convert('RGB')
            im.thumbnail((448, 448))
            messages = [{'role': 'user', 'content': [{'type': 'image', 'image': im}, {'type': 'text', 'text': row['prompt']}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.inputs_cache[row['id']] = self.processor(text=[text], images=[im], min_pixels=65536, max_pixels=200704, return_tensors='pt')
        return self.inputs_cache[row['id']].to('cuda')

    def lp(self, row, tokens):
        feed = dict(self.inputs(row))
        ids = torch.tensor([tokens], device='cuda')
        feed['input_ids'] = torch.cat([feed['input_ids'], ids], 1)
        feed['attention_mask'] = torch.cat([feed['attention_mask'], torch.ones_like(ids)], 1)
        if 'mm_token_type_ids' in feed:
            feed['mm_token_type_ids'] = torch.cat([feed['mm_token_type_ids'], torch.zeros_like(ids)], 1)
        self.model.model.rope_deltas = None
        logits = self.model(**feed, use_cache=False, logits_to_keep=len(tokens) + 1).logits[:, :-1].float()
        return -torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), reduction='none')

    def zeros(self):
        return {n: torch.zeros(p.shape, dtype=torch.float32) for n, p in self.params}

    def probe(self, rows, gradient=False):
        losses = []
        g = self.zeros() if gradient else None
        self.dest = g
        self.scale = 1 / len(rows)
        for qi, row in enumerate(rows):
            tokens = self.processor.tokenizer.encode(row['target'], add_special_tokens=False) + [self.eos]
            with torch.set_grad_enabled(gradient):
                loss = -self.lp(row, tokens).mean()
            losses.append(float(loss.detach()))
            if gradient:
                loss.backward()
            if gradient and (qi + 1) % 8 == 0:
                status(stage='probe_backward', block_completed=qi + 1, block_total=len(rows))
        self.dest = None
        return (losses, g)

    def grpo(self, row, rollout):
        g = self.zeros()
        self.dest = g
        self.scale = 1 / 8
        for sample in rollout['samples']:
            advantage = sample['advantage']
            if advantage == 0:
                continue
            lp = self.lp(row, sample['tokens'])
            loss = -(torch.exp(lp - lp.detach()) * advantage).mean()
            loss.backward()
        self.dest = None
        return g

    @torch.no_grad()
    def sync(self, master):
        for n, p in self.params:
            p.copy_(master[n])

    def master_from_model(self):
        return {n: p.detach().cpu().float().clone() for n, p in self.params}

def clip_global_gradient(g,max_norm=1.0):
 # Match torch.nn.utils.clip_grad_norm_(..., norm_type=2, foreach=False).
 total=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(t,2) for t in g.values()]),2)
 if not torch.isfinite(total):raise RuntimeError('Non-finite global gradient norm')
 factor=torch.clamp(max_norm/(total+1e-6),max=1.0)
 for t in g.values():t.mul_(factor)
 return float(total),float(factor)
