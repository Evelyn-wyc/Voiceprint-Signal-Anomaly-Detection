"""Small, reproducible theory-alignment experiment. See THEORY.md."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from datasets import SCENARIOS, real_cases, synthetic_case
from experiments import ROOT, HERE, array_digest, code_hashes, digest, git, input_record, resolve_path, write_json


def validate_config(c):
    if c['experiment']!='qpad_theory_probe_v1':raise ValueError('Unknown experiment.')
    if not c['solvers'] or len(set(c['solvers']))!=len(c['solvers']) or not set(c['solvers'])<={'palm','amsgrad'}:raise ValueError('Invalid solvers.')
    s=c['simulation'];o=c['optimization'];m=c['model'];prox=c['proximal']
    if not s['scenarios'] or not set(s['scenarios'])<=set(SCENARIOS) or len(set(s['scenarios']))!=len(s['scenarios']):raise ValueError('Invalid scenarios.')
    if not s['seeds'] or len(set(s['seeds']))!=len(s['seeds']) or any(type(v)!=int or v<0 for v in s['seeds']):raise ValueError('Invalid seeds.')
    if type(s['length'])!=int or type(s['period'])!=int or s['period']<8 or s['length']<4*s['period']:raise ValueError('Invalid simulation dimensions.')
    if not 0<m['boundary_fraction']<0.5 or type(m['alignment_points'])!=int or m['alignment_points']<4:raise ValueError('Invalid envelope or alignment grid.')
    if m['anomaly_domain'] not in ('signed','nonnegative'):raise ValueError('Invalid anomaly domain.')
    if len(m['duration_bounds_in_scale'])!=2 or not np.isfinite(m['duration_bounds_in_scale']).all() or not 0<m['duration_bounds_in_scale'][0]<m['duration_bounds_in_scale'][1]:raise ValueError('Invalid duration bounds.')
    for k in ['amplitude_bound_factor','anomaly_bound_factor']:
        if not np.isfinite(m[k]) or m[k]<1:raise ValueError('Invalid amplitude bounds.')
    for k in ['check_every','consecutive_checks','max_backtracks']:
        if type(o[k])!=int or o[k]<1:raise ValueError(f'Invalid {k}.')
    if not 0.5<o['amsgrad_decay_power']<=1:raise ValueError('Decay power must be in (0.5,1].')
    if not 0<o['descent_fraction']<1 or not 0<o['backtrack_factor']<1:raise ValueError('Invalid line search.')
    for k in ['stationarity_tolerance','fixed_point_tolerance','amsgrad_learning_rate','amsgrad_decay_scale','step_growth','step_min','step_max','descent_tolerance']:
        if not np.isfinite(o[k]) or o[k]<=0:raise ValueError(f'Invalid {k}.')
    if o['step_growth']<1 or o['step_max']<=o['step_min']:raise ValueError('Invalid step limits.')
    for kind in ['simulation','real']:
        q=c['qpad'][kind]
        if type(q['K'])!=int or q['K']<3 or type(q['epochs'])!=int or q['epochs']<1:raise ValueError('Invalid K/iteration budget.')
        if not np.isfinite(q['initial_duration']) or q['initial_duration']<=0:raise ValueError('Invalid initial duration.')
        if set(q['parameters'])!={'lambda1','lambda2','eta1','eta2','eta3','psi'} or any(not np.isfinite(v) or v<0 for v in q['parameters'].values()):raise ValueError('Invalid regularization.')
    for k in ['eps_abs','eps_rel','residual_multiplier']:
        if not np.isfinite(prox[k]) or prox[k]<=0:raise ValueError('Invalid QP tolerance.')
    if type(prox['max_iter'])!=int or prox['max_iter']<1:raise ValueError('Invalid QP budget.')
    init=c['initialization']
    if len(set(init['seeds']))!=len(init['seeds']) or any(type(v)!=int or v<0 for v in init['seeds']):raise ValueError('Invalid initialization seeds.')
    if len(set(init['modes']))!=len(init['modes']) or not set(init['modes'])<={'local','feasible_random'}:raise ValueError('Invalid initialization modes.')
    ev=c['evaluation']
    if ev['primary_multiplier'] not in ev['multipliers'] or len(set(ev['multipliers']))!=len(ev['multipliers']) or any(not np.isfinite(v) or v<=0 for v in ev['multipliers']):raise ValueError('Invalid thresholds.')
    if type(c['cpu_threads'])!=int or c['cpu_threads']<1:raise ValueError('Invalid thread count.')
    if c['real']['enabled'] and (not c['real']['recording_indices'] or len(set(c['real']['recording_indices']))!=len(c['real']['recording_indices'])):raise ValueError('Invalid recording selection.')


def prepare(config,args):
    s=config['simulation'];cases=[]
    for scenario in s['scenarios']:
        for seed in s['seeds']:
            cases.append({'id':f'sim_{scenario}_s{seed}','dataset':'simulation','scenario':scenario,'seed':seed,
                          'group':f'seed_{seed}','source_paths':[],**synthetic_case(scenario,seed,s['length'],s['period'])})
    if config['real']['enabled']:
        r=config['real'];all_real=real_cases(resolve_path(args.data_root),r['expected_count'],r['length'],args.real_signal_dir,args.real_label_dir)
        groups=sorted({c['group'] for c in all_real})
        selected=[groups[i] for i in r['recording_indices']]
        if len(set(selected))!=len(selected):raise ValueError('Duplicate selected recordings.')
        for group in selected:
            case=min((c for c in all_real if c['group']==group),key=lambda c:c['id'])
            case['id']='real_'+case['id'];cases.append(case)
    tasks=[]
    for case in cases:
        starts=[('default',0)]
        if case['id']==config['initialization']['signal']:
            starts += [(mode,seed) for mode in config['initialization']['modes'] for seed in config['initialization']['seeds']]
        for mode,seed in starts:
            for solver in config['solvers']:
                tasks.append({'id':f'{case["id"]}__{solver}__{mode}_{seed}','case_id':case['id'],
                              'solver':solver,'initialization':mode,'initialization_seed':seed})
    if config['initialization']['modes'] and config['initialization']['seeds'] and config['initialization']['signal'] not in {c['id'] for c in cases}:raise ValueError('Initialization input missing.')
    return cases,tasks


def environment(device):
    import torch,osqp,matplotlib
    import scipy,sklearn,threadpoolctl
    if device=='auto':device='cuda:0' if torch.cuda.is_available() else 'cpu'
    if device!='cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):raise ValueError('Use cpu, auto, or cuda:N.')
    target=torch.device(device)
    if target.type=='cuda':torch.empty(1,device=target)
    return target,{'python':sys.version,'device':str(target),
                   'device_name':torch.cuda.get_device_name(target) if target.type=='cuda' else 'CPU',
                   'torch_cuda_build':torch.version.cuda,
                   'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','osqp','matplotlib','threadpoolctl']}}


def references(cases,args):
    """Attach only integrity-checked, matching default archived QPAD results."""
    candidates=[resolve_path(args.reference_output)] if args.reference_output else [ROOT/'rebuttal/outputs/necessary_v1',ROOT/'results/rebuttal/server/necessary_v1']
    root=next((p for p in candidates if (p/'manifest.json').is_file()),None)
    if args.reference_output and root is None:raise FileNotFoundError('Reference manifest missing.')
    found={}
    if root is None:return found
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('experiment')!='necessary_rebuttal_v1':raise ValueError('Unexpected reference experiment.')
    records={v['id']:v for v in manifest['inputs']}
    for case in cases:
        key=case['id'];jp=root/'fits'/f'{key}__QPAD__default.json';ip=root/'inputs'/f'{key}.npz'
        if not jp.is_file() or not ip.is_file():continue
        saved=json.loads(jp.read_text());fp=jp.with_suffix('.npz')
        if saved.get('status')!='completed':continue
        if digest(fp)!=saved['artifact_sha256']:raise ValueError(f'Reference artifact corrupted: {fp}')
        with np.load(ip,allow_pickle=False) as arr:
            if key not in records or any(array_digest(arr[k])!=v for k,v in records[key]['arrays'].items()):raise ValueError('Reference input hash mismatch.')
            for k,v in case.items():
                if isinstance(v,np.ndarray):
                    if k not in arr or arr[k].shape!=v.shape:raise ValueError('Reference input shape mismatch.')
                    matches=np.array_equal(arr[k],v) if v.dtype==bool else np.allclose(arr[k],v,rtol=0,atol=1e-12*max(1.,float(np.max(np.abs(v)))))
                    if not matches:raise ValueError(f'Reference input differs: {key}/{k}')
                    case[k]=arr[k].copy()  # Fit exactly the archived arrays when available.
        found[key]={'metadata':saved,'npz':str(fp),'input_npz':str(ip),'metadata_sha256':digest(jp),'input_sha256':digest(ip)}
    return found


def validate_result(output,task,record,config):
    if record.get('status')!='completed' or any(record.get(k)!=v for k,v in task.items()):raise ValueError('Task record differs or is incomplete.')
    p=output/'fits'/f'{task["id"]}.npz'
    if not p.is_file() or digest(p)!=record['artifact_sha256']:raise ValueError('Result file/hash mismatch.')
    with np.load(p,allow_pickle=False) as a:
        for key in ['P','A','objective_history','trace','block_audit']:
            if key not in a or not np.isfinite(a[key]).all():raise ValueError(f'Invalid {key}.')
        n=len(a['P']);k=config['qpad'][record['dataset']]['K']
        if a['P'].shape!=(n,) or a['A'].shape!=(n,):raise ValueError('Component shape mismatch.')
        with np.load(output/'inputs'/f'{task["case_id"]}.npz',allow_pickle=False) as inp:
            if inp['X'].shape!=(n,):raise ValueError('Input length differs.')
        for b in ['M','t','T','A']:
            for stage in ['initial','final']:
                v=a[f'{stage}_{b}'];bounds=record['metadata']['anomaly_bounds'] if b=='A' else record['metadata']['bounds'][b]
                if v.shape!=((n,) if b=='A' else (k,)) or not np.isfinite(v).all():raise ValueError('Parameter shape/values invalid.')
                if np.any(v<np.asarray(bounds[0])-1e-10) or np.any(v>np.asarray(bounds[1])+1e-10):raise ValueError('Parameter bounds violated.')
                if b=='t' and np.any(np.diff(v)<-1e-10):raise ValueError('Onset order violated.')
        if len(a['objective_history'])!=record['metadata']['iterations']+1:raise ValueError('Objective trace incomplete.')
        if task['solver']=='palm':
            audit=a['block_audit']
            if audit.shape!=(record['metadata']['iterations']*4,11):raise ValueError('Missing block descent checks.')
            if np.any(audit[:,3]>audit[:,2]-audit[:,8]+audit[:,7]+1e-12):raise ValueError('Sufficient decrease check failed.')
        if record['metadata']['termination']=='stationarity_tolerance':
            o=config['optimization'];count=o['consecutive_checks'];cols=list(a['trace_columns']);tail=a['trace'][-count:]
            if len(tail)<count or np.any(tail[:,cols.index('iteration')]==0):raise ValueError('Too few stationarity checks.')
            if np.any(tail[:,cols.index('mapping_inf')]>o['stationarity_tolerance']) or np.any(tail[:,cols.index('relative_fixed_point_residual')]>o['fixed_point_tolerance']) or np.any(tail[:,cols.index('constraint_violation')]>1e-10):raise ValueError('Stationarity label invalid.')
    return record


def primary(metrics,config):
    return next(r for r in metrics if r['direction']=='positive' and r['rule']=='robust' and r['multiplier']==config['evaluation']['primary_multiplier'])


def summarize(output):
    from evaluation import evaluate
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text());c=manifest['config'];rows=[];failures=[];complete=[]
    for item in manifest['inputs']:
        with np.load(output/'inputs'/f'{item["id"]}.npz',allow_pickle=False) as a:
            if any(array_digest(a[k])!=v for k,v in item['arrays'].items()):raise ValueError('Stored input changed.')
    for task in manifest['tasks']:
        try:
            saved=json.loads((output/'fits'/f'{task["id"]}.json').read_text())
            if saved.get('status')=='failed':raise ValueError(saved.get('error','Failed fit.'))
            rec=validate_result(output,task,saved,c)
            complete.append(rec);m=primary(rec['metrics'],c);d=rec['metadata']['final_diagnostics']
            rows.append({'case':task['case_id'],'solver':task['solver'],'initialization':task['initialization'],'seed':task['initialization_seed'],
                         'termination':rec['metadata']['termination'],'iterations':rec['metadata']['iterations'],
                         'objective':d['objective'],'mapping_inf':d['mapping_inf'],'fixed_point_residual':d['relative_fixed_point_residual'],
                         'constraint_violation':d['constraint_violation'],'objective_increases':rec['metadata']['objective_increases'],
                         'F1':m['f1'],'AP':m['ap'],'FPR':m['fpr'],'P_RMSE':rec['p_rmse'],'A_RMSE':rec['a_rmse'],'seconds':rec['seconds']})
        except (OSError,ValueError,KeyError,TypeError,StopIteration) as exc:failures.append({'id':task['id'],'error':str(exc)})
    analysis=output/'analysis';analysis.mkdir(exist_ok=True)
    if rows:
        with (analysis/'fits.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (analysis/'failures.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['id','error']);w.writeheader();w.writerows(failures)
    summary={'planned':len(manifest['tasks']),'completed':len(complete),'failed_or_missing':len(failures),
             'stationarity_tolerance':sum(r['termination']=='stationarity_tolerance' for r in rows),
             'iteration_budget':sum(r['termination']=='iteration_budget' for r in rows)}
    write_json(analysis/'summary.json',summary)
    def fmt(v):return '—' if v is None else f'{v:.5g}' if isinstance(v,(int,float)) else str(v)
    lines=['# QPAD 模型与理论对齐：小规模验证','',
           f'计划 {summary["planned"]} 次，完成 {summary["completed"]} 次，失败/缺失 {summary["failed_or_missing"]} 次。',
           f'达到最优性残差阈值：{summary["stationarity_tolerance"]} 次；达到迭代预算：{summary["iteration_budget"]} 次。','',
           f'模型：局部半正弦（两端各 {100*c["model"]["boundary_fraction"]:g}% 平滑连接）、连续相位对齐、显式盒约束和起点顺序。',
           f'异常可行域：{c["model"]["anomaly_domain"]}。两个求解器采用相同目标、初值、约束和检测协议。',
           'PALM 的每次接受更新均检查充分下降；AMSGrad 使用递减步长及约束投影。',
           'mapping_inf 是输出变量坐标中的复合最优性残差；完成固定预算不代表满足该阈值。',
           '本轮比较求解行为与检测效果，理论结论仍需针对实际定义证明。','',
           '| 输入 | 初值 | 求解器 | 步数 | 停止原因 | 目标 | 最优性残差 | F1 | AP | A RMSE | 秒 |',
           '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join(fmt(v) for v in [r['case'],f'{r["initialization"]}/{r["seed"]}',r['solver'],r['iterations'],r['termination'],r['objective'],r['mapping_inf'],r['F1'],r['AP'],r['A_RMSE'],r['seconds']])+' |')
    lines += ['','## 数值检查','',f'- 参数约束最大违反量：{fmt(max((r["constraint_violation"] for r in rows),default=0))}。',
              f'- PALM 超出逐轮容差的目标上升次数：{sum(r["objective_increases"] for r in rows if r["solver"]=="palm")}。',
              '- 已接受更新及诊断点的近端 QP 残差、逐块下降余量和步长保存在 fits/*.npz 与 JSON。',
              '- 同一初值下比较两求解器；跨初值比较 P/A 恢复与检测，不以不同初值的目标相近作为唯一性证明。','',
              '## 归档 QPAD 对照','',
              '这些对照使用必要实验中已保存的默认初值结果；两版目标定义不同，比较检测与恢复指标。',
              '| 输入 | F1 | AP | A RMSE | 秒 |','|---|---:|---:|---:|---:|']
    refrows=[]
    for case_id,ref in manifest['references'].items():
        ip=output/'inputs'/f'{case_id}.npz';jp=output/'reference'/f'{case_id}.json';fp=output/'reference'/f'{case_id}.npz'
        if digest(jp)!=ref['copied_metadata_sha256'] or digest(fp)!=ref['artifact_sha256']:raise ValueError('Copied reference changed.')
        rec=json.loads(jp.read_text())
        with np.load(ip,allow_pickle=False) as inp,np.load(fp,allow_pickle=False) as fit:
            metrics,_=evaluate(fit['A'],inp['X'],inp['labels'],c['evaluation']['multipliers']);m=primary(metrics,c)
            rmse=float(np.sqrt(np.mean((fit['A']-inp['A_true'])**2))) if 'A_true' in inp else None
        refrows.append({'case':case_id,'F1':m['f1'],'AP':m['ap'],'A_RMSE':rmse,'seconds':rec['seconds']})
        lines.append('| '+' | '.join(fmt(v) for v in refrows[-1].values())+' |')
    if not refrows:lines.append('\n本输出未附带归档对照；可回传后与 necessary_v1 对照。')
    if refrows:
        with (analysis/'reference.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(refrows[0]));w.writeheader();w.writerows(refrows)
    if failures:lines+=['','## 失败或缺失','']+[f'- {r["id"]}: {r["error"]}' for r in failures]
    lines+=['','## 如何判断','',
            '1. 检查约束和 PALM 下降记录是否通过。',
            '2. 检查最优性残差是否下降、是否达到阈值；区分预算结束和残差达标。',
            '3. 检查 F1/AP、P/A 恢复是否保持或改善；较低目标不直接等同于较好检测。',
            '4. 比较默认、局部扰动与可行域内随机背景初值；本轮 K 和可行域固定，A 初值为零。',
            '5. 确定候选求解器后，再决定是否扩大 QPAD 验证；本轮结果是方法检查。','']
    (analysis/'report.md').write_text('\n'.join(lines))
    make_plots(output,manifest,complete)
    return len(failures)


def make_plots(output,manifest,complete):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=output/'analysis/figures';figures.mkdir(exist_ok=True)
    for case in manifest['inputs']:
        recs=[r for r in complete if r['case_id']==case['id']]
        if not recs:continue
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for r in recs:
            with np.load(output/'fits'/f'{r["id"]}.npz',allow_pickle=False) as a:
                cols=list(a['trace_columns']);t=a['trace'];label=f'{r["solver"]}/{r["initialization"]}/{r["initialization_seed"]}'
                axes[0].plot(np.arange(len(a['objective_history'])),a['objective_history'],label=label)
                axes[1].semilogy(t[:,cols.index('iteration')],np.maximum(t[:,cols.index('mapping_inf')],1e-16),label=label)
        axes[0].set(title='Objective',xlabel='Iteration');axes[1].set(title='Composite stationarity residual',xlabel='Iteration')
        axes[1].axhline(manifest['config']['optimization']['stationarity_tolerance'],ls=':',color='black')
        axes[0].legend(fontsize=6);fig.suptitle(case['id']);fig.tight_layout();fig.savefig(figures/f'{case["id"]}_optimization.png',dpi=150);plt.close(fig)
        with np.load(output/'inputs'/f'{case["id"]}.npz',allow_pickle=False) as inp:
            fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True);axes[0].plot(inp['X'],color='0.5',lw=.6,label='X')
            if 'P_true' in inp:axes[1].plot(inp['P_true'],color='black',lw=1,label='P true');axes[2].plot(inp['A_true'],color='black',lw=1,label='A true')
            for r in recs:
                if r['initialization']!='default':continue
                with np.load(output/'fits'/f'{r["id"]}.npz',allow_pickle=False) as a:
                    axes[1].plot(a['P'],lw=.8,label=r['solver']);axes[2].plot(a['A'],lw=.8,label=r['solver'])
            for ax in axes:ax.legend(fontsize=8)
            axes[0].set_title(case['id']);axes[2].set_xlabel('Feature grid point');fig.tight_layout();fig.savefig(figures/f'{case["id"]}_components.png',dpi=150);plt.close(fig)


def execute(output,manifest,cases,refs,device,resume):
    import shutil,torch
    from evaluation import evaluate
    from theory_model import fit_theory
    if output.exists():
        if not resume:raise FileExistsError('Output exists; use --resume for the same configuration.')
        old=json.loads((output/'manifest.json').read_text())
        for k in ['config','code_sha256','inputs','tasks','environment','references']:
            if old[k]!=manifest[k]:raise ValueError(f'Resume rejected: {k} changed.')
        for record in old['inputs']:
            with np.load(output/'inputs'/f'{record["id"]}.npz',allow_pickle=False) as a:
                if any(array_digest(a[k])!=v for k,v in record['arrays'].items()):raise ValueError('Resume input corrupted.')
    else:
        if resume:raise FileNotFoundError('No output to resume.')
        for name in ['inputs','fits','reference','live']: (output/name).mkdir(parents=True,exist_ok=True)
        write_json(output/'manifest.json',manifest);write_json(output/'config.json',manifest['config'])
        for case in cases:np.savez_compressed(output/'inputs'/f'{case["id"]}.npz',**{k:v for k,v in case.items() if isinstance(v,np.ndarray)})
        for key,r in refs.items():
            shutil.copyfile(r['npz'],output/'reference'/f'{key}.npz');write_json(output/'reference'/f'{key}.json',r['metadata'])
    by_id={c['id']:c for c in cases};config=manifest['config']
    for index,task in enumerate(manifest['tasks'],1):
        jp=output/'fits'/f'{task["id"]}.json'
        if resume and jp.exists():
            saved=json.loads(jp.read_text())
            if saved.get('status')=='completed':
                validate_result(output,task,saved,config);print(f'[{index}/{len(manifest["tasks"])}] reuse {task["id"]}',flush=True);continue
        case=by_id[task['case_id']];print(f'[{index}/{len(manifest["tasks"])}] {task["id"]}',flush=True)
        def progress(row):
            write_json(output/'live'/f'{task["id"]}.json',row)
            if row['iteration']%100==0:print(f'  iter={row["iteration"]} F={row["objective"]:.6g} residual={row["mapping_inf"]:.3g}',flush=True)
        try:
            if device.type=='cuda':torch.cuda.synchronize(device)
            started=time.perf_counter()
            p,a,meta,extra=fit_theory(case['X'],case['dataset'],config,device,task['solver'],task['initialization'],task['initialization_seed'],progress)
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.perf_counter()-started
            metrics,z=evaluate(a,case['X'],case['labels'],config['evaluation']['multipliers'])
            fp=jp.with_suffix('.npz');tmp=fp.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,P=p,A=a,standardized=z,**extra)
            tmp.replace(fp)
            rec={**task,'status':'completed','dataset':case['dataset'],'metadata':meta,'seconds':seconds,'metrics':metrics,
                 'p_rmse':float(np.sqrt(np.mean((p-case['P_true'])**2))) if 'P_true' in case else None,
                 'a_rmse':float(np.sqrt(np.mean((a-case['A_true'])**2))) if 'A_true' in case else None,'artifact_sha256':digest(fp)}
            validate_result(output,task,rec,config)
            print(f'  {meta["termination"]}, {seconds:.1f}s, F1={primary(metrics,config)["f1"]:.4f}',flush=True)
        except Exception as exc:
            rec={**task,'status':'failed','error':str(exc),'traceback':traceback.format_exc()};print(rec['traceback'],flush=True)
        write_json(jp,rec)
        with (output/'events.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'task':task['id'],'status':rec['status']})+'\n')
        write_json(output/'progress.json',{'position':index,'planned':len(manifest['tasks']),'task':task['id'],'status':rec['status']})
    return 1 if summarize(output) else 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run','summarize'])
    parser.add_argument('--config',type=Path,default=HERE/'configs/theory_probe.json')
    parser.add_argument('--data-root',default=str(ROOT));parser.add_argument('--real-signal-dir');parser.add_argument('--real-label-dir')
    parser.add_argument('--device',default='auto');parser.add_argument('--output',default='rebuttal/outputs/theory_probe_v1')
    parser.add_argument('--reference-output');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--sim-only',action='store_true');parser.add_argument('--solver',choices=['both','amsgrad','palm'],default='both')
    parser.add_argument('--max-iterations',type=int)
    args=parser.parse_args(argv)
    try:
        output=resolve_path(args.output)
        if args.command=='summarize':return 1 if summarize(output) else 0
        config=json.loads(args.config.expanduser().resolve().read_text())
        if args.sim_only:config['real']['enabled']=False
        if args.solver!='both':config['solvers']=[args.solver]
        if args.max_iterations is not None:
            for kind in ['simulation','real']:config['qpad'][kind]['epochs']=args.max_iterations
        validate_config(config)
        device,env=environment(args.device)
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(config['cpu_threads'])
        cases,tasks=prepare(config,args);refs=references(cases,args)
        import hashlib
        refs_manifest={k:{'source_npz':r['npz'],'source_input':r['input_npz'],'input_sha256':r['input_sha256'],
                          'artifact_sha256':r['metadata']['artifact_sha256'],
                          'copied_metadata_sha256':hashlib.sha256((json.dumps(r['metadata'],ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()} for k,r in refs.items()}
        manifest={'experiment':config['experiment'],'created_utc':datetime.now(timezone.utc).isoformat(),'git_commit':git('rev-parse','HEAD'),
                  'rebuttal_git_status':git('status','--porcelain','--untracked-files=all','--','rebuttal'),
                  'config':config,'code_sha256':code_hashes(),'environment':env,'inputs':[input_record(c) for c in cases],
                  'tasks':tasks,'references':refs_manifest,
                  'timing_scope':'setup, optimization, stationarity checks and live progress writes; CUDA synchronized; excludes final evaluation/plots'}
        if args.command=='check':
            print(json.dumps({'status':'ready','device':str(device),'inputs':[c['id'] for c in cases],
                              'fits':len(tasks),'solvers':config['solvers'],'archived_references':len(refs),'output':str(output)},ensure_ascii=False,indent=2));return 0
        with threadpool_limits(limits=config['cpu_threads']):return execute(output,manifest,cases,refs,device,args.resume)
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
