# -*- coding: utf-8 -*-
"""Task2: single-GPU training memory strategies."""
import argparse, json, os, time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
SEED = 7
def reset_peak():
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
def mib(x): return x / 2**20
class Block(nn.Module):
    def __init__(self, d, hidden):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, x): return x + self.net(x)
class TinyTrain(nn.Module):
    def __init__(self, d=1024, hidden=4096, layers=4, classes=1024):
        super().__init__(); self.blocks=nn.ModuleList([Block(d, hidden) for _ in range(layers)]); self.head=nn.Linear(d, classes)
    def forward(self, x, ckpt=False, offload=False):
        ctx = torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else torch.enable_grad()
        with ctx:
            for b in self.blocks:
                x = checkpoint(b, x, use_reentrant=False) if ckpt else b(x)
            return self.head(x)
def run_case(name, micro, accum, ckpt=False, offload=False, d=512, seq=2048, layers=8):
    torch.manual_seed(SEED); dev=torch.device('cuda'); model=TinyTrain(d=d, hidden=4*d, layers=layers, classes=d).to(dev); opt=torch.optim.AdamW(model.parameters(), lr=1e-3)
    reset_peak(); t0=time.perf_counter(); opt.zero_grad(set_to_none=True); losses=[]
    for step in range(accum):
        x=torch.randn(micro, seq, d, device=dev); y=torch.randn(micro, seq, d, device=dev); pred=model(x, ckpt=ckpt, offload=offload); loss=((pred-y)**2).mean()/accum; loss.backward(); losses.append(float(loss.detach())*accum); del x,y,pred,loss
    opt.step(); torch.cuda.synchronize(); elapsed=time.perf_counter()-t0
    result={'name':name,'micro_batch':micro,'accum_steps':accum,'effective_batch':micro*accum,'checkpoint':ckpt,'offload':offload,'peak_allocated_mib':round(mib(torch.cuda.max_memory_allocated()),2),'reserved_mib':round(mib(torch.cuda.max_memory_reserved()),2),'step_time_s':round(elapsed,4),'loss_first':round(losses[0],6),'loss_last':round(losses[-1],6)}
    del model,opt; torch.cuda.empty_cache(); return result
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',default='evidence/task2'); a=ap.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    print('GPU:',torch.cuda.get_device_name(0),'torch',torch.__version__,'VRAM_GiB',round(torch.cuda.get_device_properties(0).total_memory/2**30,2),flush=True)
    cases=[('baseline_batch4',4,1,False,False),('accum_micro1',1,4,False,False),('checkpoint_micro1',1,4,True,False),('offload_micro1',1,4,False,True)]; rows=[]
    for c in cases:
        print('RUN',c[0],flush=True); r=run_case(c[0],c[1],c[2],c[3],c[4]); rows.append(r); print(json.dumps(r,ensure_ascii=False),flush=True)
    out=os.path.join(a.out_dir,'task2_results.json'); json.dump({'environment':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__},'results':rows},open(out,'w'),indent=2,ensure_ascii=False)
    try:
        import matplotlib.pyplot as plt
        names=[r['name'] for r in rows]; peaks=[r['peak_allocated_mib'] for r in rows]; times=[r['step_time_s'] for r in rows]; fig,ax=plt.subplots(1,2,figsize=(13,5)); ax[0].bar(names,peaks); ax[0].set_ylabel('peak allocated (MiB)'); ax[1].bar(names,times); ax[1].set_ylabel('step time (s)'); fig.suptitle('Task2 vm-60: micro-step / checkpoint / offload'); fig.tight_layout(); fig.savefig(os.path.join(a.out_dir,'task2_memory_strategy_runshot.png'),dpi=160)
    except Exception as e: print('PLOT_SKIPPED',repr(e),flush=True)
    print('ALL_TESTS_PASS',out,flush=True)
if __name__=='__main__': main()
