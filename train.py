"""Probe-gradient candidate selection, followed by sequential GRPO training."""
import os
os.environ['QA_ARM']='image_full'
from pathlib import Path
import argparse, fcntl, hashlib, json, math, re, time, traceback
from engine import Engine, torch, np, write, cleanup, clip_global_gradient
from easy_loss import compute_policy_loss, compute_grpo_outcome_advantage
R=Path(os.environ.get('GRPO_RUN_DIR','run')).expanduser().resolve()
CONTEXT={}

def status(stage, **kw):
    obj=dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),stage=stage,**CONTEXT,**kw)
    write(R/'status.json',obj);print(json.dumps(obj),flush=True)

def load(name): return json.loads((R/'data'/f'{name}.json').read_text())

def norm(t): return re.sub(r'\s+',' ',t.strip().lower()).strip(' .,!?:;\"\'')

def save_torch(obj,p):
    p=Path(p);tmp=p.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(p)

def rollout(e,row,seed,path):
    if path.exists():return json.loads(path.read_text())
    torch.manual_seed(seed);feed=e.inputs(row);e.model.model.rope_deltas=None
    with torch.no_grad():
        outs=e.model.generate(**feed,do_sample=True,temperature=1.,top_p=1.,top_k=0,
            num_return_sequences=8,max_new_tokens=32,use_cache=True,eos_token_id=e.eos,
            pad_token_id=e.processor.tokenizer.pad_token_id)
    samples=[]
    for out in outs:
        tok=out[feed['input_ids'].shape[1]:].tolist()
        if e.eos in tok:tok=tok[:tok.index(e.eos)+1]
        text=e.processor.tokenizer.decode(tok,skip_special_tokens=True).strip()
        samples.append(dict(tokens=tok,text=text,reward=float(norm(text)==norm(row['target'])),truncated=e.eos not in tok))
    rewards=torch.tensor([[s['reward']] for s in samples],dtype=torch.float32)
    with torch.no_grad():
        adv,_=compute_grpo_outcome_advantage(rewards,torch.ones_like(rewards),[0]*8)
    for s,a in zip(samples,adv[:,0]):s['advantage']=float(a)
    obj=dict(id=row['id'],seed=seed,samples=samples,zero_advantage=bool((adv==0).all()))
    write(path,obj);return obj

def grpo(e,row,roll):
    g=e.zeros();e.dest=g;e.scale=1/8
    for s in roll['samples']:
        if s['advantage']==0:continue
        lp=e.lp(row,s['tokens']).unsqueeze(0)
        loss,_=compute_policy_loss(lp.detach(),lp,torch.full_like(lp,s['advantage']),
            torch.ones_like(lp),0.2,0.2,3.0,'default','seq')
        assert torch.isfinite(loss)
        loss.backward()
    e.dest=None
    gn,clip=clip_global_gradient(g,1.)
    return g,gn,clip

def one_param(b,m,g,copy=True):
    p=torch.nn.Parameter(b.clone() if copy else b)
    opt=torch.optim.AdamW([p],lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    if m is not None:opt.state[p]={k:(v.clone() if copy else v) for k,v in m.items()}
    p.grad=g;opt.step();p.grad=None
    return p.detach(),{k:v.detach() for k,v in opt.state[p].items()}

@torch.no_grad()
def preview(base,mom,g,h=None):
    score=0.;changed=0;d2=0.
    for n,b in base.items():
        p,state=one_param(b,mom.get(n),g[n])
        assert torch.isfinite(p).all(), n
        q0=b.bfloat16().reshape(-1);q1=p.bfloat16().reshape(-1)
        ix=torch.nonzero(q0!=q1).reshape(-1);changed+=ix.numel()
        if ix.numel():
            delta=q1[ix].double()-q0[ix].double();d2+=float(torch.dot(delta,delta))
            if h is not None:score-=float(torch.dot(h[n].reshape(-1)[ix].double(),delta))
        del p,state,q0,q1,ix
    return dict(score=score,changed_elements=changed,update_norm=math.sqrt(d2))

@torch.no_grad()
def commit(base,mom,g,h=None):
    score=0.;changed=0
    for n in list(base):
        old=base[n].bfloat16().reshape(-1)
        new,state=one_param(base[n],mom.get(n),g[n],copy=False)
        assert torch.isfinite(new).all(), n
        q1=new.bfloat16().reshape(-1);ix=torch.nonzero(old!=q1).reshape(-1);changed+=ix.numel()
        if h is not None and ix.numel():
            score-=float(torch.dot(h[n].reshape(-1)[ix].double(),q1[ix].double()-old[ix].double()))
        base[n]=new;mom[n]=state
    return dict(score=score,changed_elements=changed)

def self_test():
    torch.set_num_threads(8)
    lp=torch.tensor([[-1.,-2.,-3.]],requires_grad=True)
    loss,_=compute_policy_loss(lp.detach(),lp,torch.full_like(lp,0.7),torch.ones_like(lp),.2,.2,3.,'default','seq')
    loss.backward();assert torch.allclose(lp.grad,torch.full_like(lp,-.7/3),atol=1e-7)
    b={'p':torch.tensor([1.00390625,-.001,0.0])};mom={};g={'p':torch.tensor([.1,.2,.3])};h={'p':torch.tensor([.4,.5,.6])}
    prior=b['p'].clone();a=preview(b,mom,g,h)
    assert torch.equal(prior,b['p']) and not mom
    c=commit(b,mom,g,h);assert a['score']==c['score'] and a['changed_elements']==c['changed_elements']
    # Stateful AdamW preview must preserve moments and agree with commit too.
    prior=b['p'].clone();oldmom={k:v.clone() for k,v in mom['p'].items()};a=preview(b,mom,g,h)
    assert torch.equal(prior,b['p']) and all(torch.equal(oldmom[k],v) for k,v in mom['p'].items())
    c=commit(b,mom,g,h);assert a['score']==c['score'] and a['changed_elements']==c['changed_elements']
    write(R/'results/self_test.json',dict(passed=True,preview_commit_exact=True,easy_loss_gradient=True))

def probe_gradient(e,rows):
    g=e.zeros();e.dest=g;e.scale=1/len(rows);losses=[]
    for i,row in enumerate(rows):
        tokens=e.processor.tokenizer.encode(row['target'],add_special_tokens=False)+[e.eos]
        loss=-e.lp(row,tokens).mean();assert torch.isfinite(loss)
        losses.append(float(loss.detach()));loss.backward()
        if (i+1)%8==0:status('probe_gradient',completed=i+1,total=len(rows))
        e.inputs_cache.clear()
    e.dest=None
    assert all(bool(torch.isfinite(v).all()) for v in g.values())
    return losses,g

def evaluate(e,rows,label,accuracy):
    losses=[];correct=[];answers=[]
    for i,row in enumerate(rows):
        v,_=e.probe([row],False);assert math.isfinite(v[0]);losses+=v
        if accuracy:
            feed=e.inputs(row);e.model.model.rope_deltas=None
            with torch.no_grad():
                out=e.model.generate(**feed,do_sample=False,num_return_sequences=1,max_new_tokens=32,
                    use_cache=True,eos_token_id=e.eos,pad_token_id=e.processor.tokenizer.pad_token_id)
            answer=e.processor.tokenizer.decode(out[0,feed['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            answers.append(answer);correct.append(norm(answer)==norm(row['target']))
        e.inputs_cache.clear()
        if (i+1)%16==0:status('evaluation',subset=label,completed=i+1,total=len(rows))
    return dict(ids=[x['id'] for x in rows],losses=losses,mean_ce=float(np.mean(losses)),
        accuracy=float(np.mean(correct)) if accuracy else None,correct=correct,answers=answers)

def evaluate_sets(e,step,root,baseline):
    p=root/f'eval_{step:02d}.json'
    if p.exists():return json.loads(p.read_text())
    a=evaluate(e,load('probe'),'probe128',step in [0,20]);b=evaluate(e,load('validation'),'heldout128',step in [0,20])
    out=dict(step=step,probe128=a,heldout128=b,probe32_ce=float(np.mean(a['losses'][:32])))
    if baseline:
        out['improvement']={k:baseline[k]['mean_ce']-out[k]['mean_ce'] for k in ['probe128','heldout128']}
        out['improvement']['probe32']=baseline['probe32_ce']-out['probe32_ce']
        if step==20:out['accuracy_change']={k:out[k]['accuracy']-baseline[k]['accuracy'] for k in ['probe128','heldout128']}
    write(p,out);return out

def train_arm(e,n,cs,baseline):
    CONTEXT.update(arm=f'probe{n}');root=R/f'probe{n}'
    cpfile=root/'latest.pt';cp=torch.load(cpfile if cpfile.exists() else R/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
    base,mom=cp['parameters'],cp['moments'];history=cp.get('history',[]);startstep=cp.get('completed_steps',0)
    e.sync(base);write(root/'history.json',history)
    if startstep>0 and startstep%5==0:evaluate_sets(e,startstep,root,baseline)
    for step in range(startstep+1,21):
        CONTEXT.update(step=step);t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
        status('step_start',total_steps=20)
        before,h=probe_gradient(e,load('probe')[:n]);bestg=None;best=None;records=[]
        group=cs[(step-1)*4:step*4]
        for retry in range(5):
            for ci,row in enumerate(group):
                status('candidate_rollout',id=row['id'],candidate=ci+1,retry=retry)
                seed=20261101+step*100+retry*10+ci
                rr=rollout(e,row,seed,R/'rollouts'/f'arm{n}_step{step:02d}_r{retry}_{row["id"]}.json')
                rec=dict(id=row['id'],retry=retry,rewards=[s['reward'] for s in rr['samples']],eligible=False)
                if not rr['zero_advantage']:
                    status('candidate_gradient',id=row['id'],candidate=ci+1)
                    g,gn,clip=grpo(e,row,rr);pr=preview(base,mom,g,h)
                    rec.update(gradient_norm=gn,clip=clip,**pr,eligible=gn>0 and pr['changed_elements']>0)
                    if rec['eligible'] and (best is None or rec['score']>best['score']):
                        bestg=g;best=rec
                    del g
                records.append(rec);write(root/f'candidates_step{step:02d}.json',records)
                e.inputs_cache.clear();cleanup()
            if best is not None:break
        if best is None:raise RuntimeError(f'No effective GRPO update in fixed group at arm {n} step {step}; no silent skip')
        status('commit_selected_update',id=best['id'],predicted=best['score'])
        actual=commit(base,mom,bestg,h)
        assert actual['changed_elements']==best['changed_elements']
        assert math.isclose(actual['score'],best['score'],rel_tol=1e-10,abs_tol=1e-12)
        del bestg,h;e.sync(base);cleanup()
        history.append(dict(step=step,selected_id=best['id'],predicted_improvement=best['score'],
            probe_before=float(np.mean(before)),changed_elements=actual['changed_elements'],
            eligible_count=sum(r['eligible'] for r in records),attempted_questions=len(records),
            peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,seconds_before_checkpoint=time.monotonic()-t0))
        if step%10==0:
            status('saving_checkpoint',checkpoint_interval=10)
            stepfile=root/f'step_{step:03d}.pt'
            save_torch(dict(parameters=base,moments=mom,history=history,completed_steps=step,arm=n),stepfile)
            alias=root/'latest.link.tmp'
            if alias.exists():alias.unlink()
            os.link(stepfile,alias);alias.replace(cpfile)
        write(root/'history.json',history)
        if step%5==0:evaluate_sets(e,step,root,baseline)
        status('step_complete',completed=step,total=20,selected_id=best['id'])
    write(root/'COMPLETED.json',dict(steps=20,probe=n,final_evaluation=str(root/'eval_20.json')))
    del cp,base,mom;cleanup();CONTEXT.clear()

def main():
    lock=open(R/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((R/'manifest.json').read_text())
    for name,expected in manifest['input_hashes'].items():
        assert hashlib.sha256((R/'data'/name).read_bytes()).hexdigest()==expected, f'Input modified: {name}'
    cs=load('candidates80')
    assert len(cs)==80 and len(load('probe'))==128 and len(load('validation'))==128
    self_test();status('loading_model');e=Engine('bf16')
    cp=torch.load(R/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
    assert set(cp['parameters'])=={n for n,p in e.params}
    assert all(p.dtype==torch.float32 for p in cp['parameters'].values()), 'Checkpoint requires FP32 master parameters'
    e.sync(cp['parameters']);del cp;cleanup()
    baseline=evaluate_sets(e,0,R/'results',None)
    for n in [32,128]:train_arm(e,n,cs,baseline)
    final={str(n):json.loads((R/f'probe{n}/eval_20.json').read_text()) for n in [32,128]}
    write(R/'results/comparison.json',dict(baseline=baseline,arms=final,selection_used_validation=False))
    write(R/'results/COMPLETED.json',dict(arms=[32,128],steps_per_arm=20,candidates=80))
    status('complete',arms=2,steps_per_arm=20)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    (R/'results').mkdir(parents=True,exist_ok=True)
    try:
        if a.self_test:self_test();print('SELF_TEST_PASSED')
        else:main()
    except BaseException as ex:
        write(R/'results/ERROR.json',dict(error=repr(ex),traceback=traceback.format_exc(),context=CONTEXT))
        status('failed',error=repr(ex));raise
