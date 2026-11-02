"""Create a run from local, user-supplied model/checkpoint and disjoint QA splits."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def read_rows(path, count):
    path=Path(path).expanduser().resolve()
    rows=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows,list) or len(rows)!=count:
        raise ValueError(f'{path}: expected {count} rows')
    for row in rows:
        for key in ['id','image_id','image','prompt','target']:
            if key not in row:raise ValueError(f'Missing {key}')
        row['id']=str(row['id']);row['image_id']=str(row['image_id'])
        image=Path(row['image']).expanduser()
        if not image.is_absolute():image=path.parent/image
        image=image.resolve()
        if not image.is_file():raise ValueError(f'Missing image: {image}')
        row['image']=str(image)
    if len({r['id'] for r in rows})!=count or len({r['image_id'] for r in rows})!=count:
        raise ValueError('Duplicate question/image ID within split')
    return rows


def validate_splits(candidates,probe,validation):
    for field in ['id','image_id']:
        sets=[{r[field] for r in rows} for rows in [candidates,probe,validation]]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
            raise ValueError(f'Overlapping {field} across splits')


def prepare(args):
    model=Path(args.model).expanduser().resolve()
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    if not model.is_dir() or not checkpoint.is_file():raise ValueError('Local model/checkpoint missing')
    cs=read_rows(args.candidates,80);ps=read_rows(args.probe,128);vs=read_rows(args.validation,128)
    validate_splits(cs,ps,vs)
    random.Random(args.seed).shuffle(cs)
    root=Path(args.run_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
    for d in ['data','results','rollouts','cache','tmp','probe32','probe128']:(root/d).mkdir()
    (root/'model').symlink_to(model,target_is_directory=True)
    (root/'checkpoint.pt').symlink_to(checkpoint)
    for name,rows in [('candidates80',cs),('probe',ps),('validation',vs)]:
        (root/'data'/f'{name}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    manifest=dict(seed=args.seed,model=str(model),checkpoint=str(checkpoint),
        arms=[32,128],steps=20,group_size=4,selection_count=1,
        checkpoint_steps=[10,20],eval_steps=[0,5,10,15,20],accuracy_steps=[0,20],
        input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data').glob('*.json')})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(root)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['run-dir','model','checkpoint','candidates','probe','validation']:p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=20261101)
    prepare(p.parse_args())
