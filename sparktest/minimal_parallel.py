"""Fixed five ECGs: equal-workload independent training tasks at selected concurrency."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from phase1_ecg_robustness.src.models import build_model


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def seed(value):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def worker(args):
    output = args.output
    started = time.perf_counter()
    result = {'status': 'failed', 'seed': args.seed, 'model': args.model, 'steps': args.steps}
    try:
        seed(args.seed)
        assert torch.cuda.is_available(), 'CUDA required'
        checkpoint = torch.load(args.inputs / f'{args.model}.pt', map_location='cpu', weights_only=False)
        with np.load(args.inputs / 'fixed5.npz') as packed:
            x = torch.from_numpy(packed['x'].copy()).cuda() / float(checkpoint['scale_mv'])
            y = torch.from_numpy(packed['y'].copy()).cuda()
        assert x.shape == (5, 12, 1000) and y.shape == (5, 5)
        model = build_model(args.model, **checkpoint['model_kwargs']).cuda().train()
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        def step():
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            return loss
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        model.load_state_dict(initial)
        del initial
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        result['preparation_seconds'] = time.perf_counter() - started
        (output / 'ready').touch()
        deadline = time.monotonic() + 120
        while not args.gate.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Parent start gate unavailable')
            time.sleep(.002)
        torch.cuda.synchronize()
        clock = time.perf_counter()
        first_loss = None
        for index in range(args.steps):
            loss = step()
            if index == 0:
                first_loss = float(loss.detach().cpu())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - clock
        if not torch.isfinite(loss) or any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite training state')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite/missing gradient')
        model.eval()
        with torch.inference_mode():
            logits = model(x).cpu().numpy()
        if logits.shape != (5, 5) or not np.isfinite(logits).all():
            raise FloatingPointError('Invalid logits')
        np.save(output / 'logits.npy', logits)
        torch.save({'model_name': args.model, 'model_state': model.state_dict(), 'seed': args.seed,
                    'steps': args.steps, 'benchmark_only': True}, output / 'checkpoint.pt')
        result.update(status='passed', train_seconds=seconds, samples_per_second=5*args.steps/seconds,
                      first_loss=first_loss, final_loss=float(loss.detach().cpu()),
                      logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                      checkpoint_sha256=digest(output/'checkpoint.pt'),
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
    except Exception:
        result['error'] = traceback.format_exc()
    result['worker_wall_seconds'] = time.perf_counter() - started
    save(output / 'result.json', result)
    return 0 if result['status'] == 'passed' else 1


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    from sparktest.train_benchmark import MemoryMonitor
    with np.load(args.inputs / 'test_full.npz') as source:
        fixed = {key: source[key][:5].copy() for key in ('y', 'ecg_id', 'patient_id')}
        fixed['x'] = source['clean'][:5].copy()
    np.savez(args.output / 'fixed5.npz', **fixed)
    import shutil
    for name in ('resnet', 'tcn'):
        shutil.copyfile(args.inputs / f'{name}.pt', args.output / f'{name}.pt')
    summary = {'status': 'running', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'records': 5, 'ecg_ids': fixed['ecg_id'].tolist(), 'steps_per_task': args.steps,
               'seeds': [17, 29, 43, 101, 202, 307, 401, 503][:args.tasks],
               'task_count': args.tasks, 'concurrencies': args.concurrencies,
               'precision': 'deterministic FP32, AMP/TF32 off',
               'torch': torch.__version__, 'cuda': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(0), 'input_sha256': digest(args.output/'fixed5.npz'),
               'source_sha256': digest(Path(__file__)), 'groups': [], 'comparisons': [],
               'monitor_source_sha256': digest(ROOT/'sparktest/train_benchmark.py'),
               'scope': 'Five fixed clean ECGs repeated for timing; not learning-quality evidence or full-cohort scaling proof. Model/optimizer warmup excluded from measured gate-to-completion wave time, but included in group wall. Existing ComfyUI left untouched.'}
    save(args.output / 'summary.json', summary)
    monitor = None
    try:
        for name in ('resnet', 'tcn'):
            for concurrency in args.concurrencies:
                group_dir = args.output / f'{name}_p{concurrency}'
                group_dir.mkdir()
                monitor = MemoryMonitor(group_dir)
                monitor.start()
                group_start = time.perf_counter()
                group = {'model': name, 'concurrency': concurrency, 'jobs': [], 'waves': []}
                for offset in range(0, args.tasks, concurrency):
                    processes, logs = [], []
                    gate = group_dir / f'gate_{offset}'
                    try:
                        for number in summary['seeds'][offset:offset+concurrency]:
                            folder = group_dir / f'seed_{number}'
                            folder.mkdir()
                            log = (folder/'stdout.log').open('w', encoding='utf-8')
                            logs.append(log)
                            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--inputs', str(args.output.resolve()),
                                       '--output', str(folder.resolve()), '--model', name, '--seed', str(number),
                                       '--steps', str(args.steps), '--gate', str(gate.resolve())]
                            processes.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), folder))
                        deadline = time.monotonic() + 120
                        while not all((folder/'ready').exists() for _,folder in processes):
                            if any(process.poll() is not None for process,_ in processes):
                                raise RuntimeError('Worker failed before start gate; inspect per-worker result/log')
                            if time.monotonic() > deadline:
                                raise TimeoutError('Worker preparation timed out')
                            time.sleep(.005)
                        wave_start = time.perf_counter()
                        gate.touch()
                        for process,folder in processes:
                            if process.wait(timeout=180) != 0:
                                raise RuntimeError(f'Worker failed: {folder}')
                        wave_seconds = time.perf_counter() - wave_start
                        jobs = [json.loads((folder/'result.json').read_text()) for _,folder in processes]
                        group['jobs'].extend(jobs)
                        group['waves'].append({'gate_to_exit_seconds': wave_seconds,
                                               'max_worker_train_seconds': max(j['train_seconds'] for j in jobs)})
                    finally:
                        for process,_ in processes:
                            if process.poll() is None:
                                process.terminate()
                                process.wait(timeout=10)
                        for log in logs:
                            log.close()
                group['wall_seconds'] = time.perf_counter() - group_start
                group['memory'] = monitor.finish()
                monitor = None
                group['steady_train_makespan_seconds'] = sum(w['max_worker_train_seconds'] for w in group['waves'])
                group['gate_to_exit_seconds'] = sum(w['gate_to_exit_seconds'] for w in group['waves'])
                group['steady_samples_per_second'] = args.tasks*5*args.steps/group['steady_train_makespan_seconds']
                summary['groups'].append(group)
                save(args.output/'summary.json',summary)
                print(json.dumps({k:group[k] for k in ('model','concurrency','wall_seconds','steady_train_makespan_seconds','steady_samples_per_second')}),flush=True)
            reference = next(g for g in summary['groups'] if g['model']==name and g['concurrency']==1)
            for candidate in [g for g in summary['groups'] if g['model']==name and g['concurrency']>1]:
                identical = all(j['logits_sha256']==next(r for r in reference['jobs'] if r['seed']==j['seed'])['logits_sha256'] for j in candidate['jobs'])
                speedup=reference['steady_train_makespan_seconds']/candidate['steady_train_makespan_seconds']
                summary['comparisons'].append({'model':name,'concurrency':candidate['concurrency'],
                    'steady_speedup':speedup,'efficiency':speedup/candidate['concurrency'],
                    'including_startup_speedup':reference['wall_seconds']/candidate['wall_seconds'],
                    'prediction_hashes_identical':identical})
                if not identical:
                    raise RuntimeError('Concurrent numerical results differ from sequential')
        summary['status']='passed'
    except Exception:
        if monitor is not None:
            monitor.finish()
        summary.update(status='failed',error=traceback.format_exc())
    summary['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save(args.output/'summary.json',summary)
    return 0 if summary['status']=='passed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--tasks',type=int,choices=(4,8),default=4)
    parser.add_argument('--concurrencies',type=int,nargs='+',choices=(1,2,4,8),default=[1,2,4])
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--model',choices=('resnet','tcn'))
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--gate',type=Path)
    args=parser.parse_args()
    if args.steps<1: parser.error('--steps must be positive')
    if args.concurrencies != sorted(set(args.concurrencies)) or args.concurrencies[0] != 1:
        parser.error('--concurrencies must be sorted, unique, and include serial baseline1')
    if max(args.concurrencies)>args.tasks:
        parser.error('Concurrency cannot exceed task count')
    raise SystemExit(worker(args) if args.worker else run(args))
