"""Independent signed-control numerical and publication acceptance (no re-inference)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from .common import (METRICS, CONTRASTS, array_sha256, check_info, file_info,
                     legacy_paths, load_config, load_manifest, load_reference,
                     read_json, require_freeze, resolve_path, save_json, stage_paths)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context, atol=2e-12):
    require(np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True), context)


def verify_matrices(cfg, freeze):
    matrices = read_json(check_info(freeze['matrices']))
    controls = read_json(check_info(freeze['sign_controls']))
    a = np.asarray(matrices['standard']['matrix'], dtype=np.float64)
    covariance = a @ a.T
    u, singular, _ = np.linalg.svd(a, full_matrices=True)
    rank = int(np.linalg.matrix_rank(a))
    require(rank == 8, 'Standard rank must be eight')
    p, q = np.asarray(matrices['projection']['P']), np.asarray(matrices['projection']['Q'])
    close(p, u[:, :rank] @ u[:, :rank].T, 'Independent SVD projector')
    close(q, u[:, rank:] @ u[:, rank:].T, 'Independent SVD null projector')
    close(p @ p, p, 'Projector idempotence')
    close(p.T, p, 'Projector symmetry')
    close(q @ a, np.zeros_like(a), 'Standard null projection')
    rng = np.random.Generator(np.random.PCG64(cfg['selection']['seed']))
    accepted = []
    for candidate in controls['candidate_history']:
        signs = np.r_[1, rng.choice(np.array([-1, 1]), size=11)]
        require(np.array_equal(signs, candidate['sign_vector']), 'Frozen PCG64 sequence replay')
        changed = np.linalg.norm(signs[:, None] * covariance * signs[None, :] - covariance) / np.linalg.norm(covariance)
        eligible = (4 <= (signs < 0).sum() <= 7 and tuple(signs) not in accepted
                    and changed > cfg['selection']['covariance_change_min'])
        require(eligible == candidate['accepted'], 'First-eligible selection replay')
        if eligible:
            accepted.append(tuple(signs))
    require(len(accepted) == 5, 'Exactly five accepted fixed controls')
    for control, matrix, signs in zip(controls['controls'], matrices['controls'], accepted):
        require(tuple(control['sign_vector']) == signs, 'Control acceptance order')
        signed = np.asarray(signs)[:, None] * a
        close(signed, matrix['matrix'], 'Signed matrix bytes', atol=0)
        require(np.linalg.matrix_rank(signed) == rank, 'Signed matrix rank')
        close(np.linalg.svd(signed, compute_uv=False), singular, 'Every singular value')
        close(np.linalg.eigvalsh(signed @ signed.T), np.linalg.eigvalsh(covariance), 'Every covariance eigenvalue')
        close(np.diag(signed @ signed.T), np.diag(covariance), 'Every lead variance', atol=0)
        require(not np.array_equal(signed @ signed.T, covariance), 'Changed covariance sign layout')
        require(array_sha256(signed) == matrix['array_sha256'], 'Signed matrix array hash')
    return u[:, rank:].T, {'rank': rank, 'controls': 5, 'candidate_indices': [v['candidate_index'] for v in controls['controls']],
                         'independent_method': 'Full SVD left-null basis; direct PCG64 candidate replay'}


def verify_inputs(cfg, stage, freeze, manifest, null_basis):
    paths = stage_paths(cfg, stage)
    reference = load_reference(cfg, stage)
    n = len(reference['ids'])
    audit = pd.read_csv(check_info(manifest['audit']), keep_default_na=False)
    diagnostic = pd.read_csv(check_info(manifest['diagnostics']['per_record']))
    summary = pd.read_csv(check_info(manifest['diagnostics']['summary']))
    require(len(audit) == 126 * n and len(diagnostic) == 253 * n and len(summary) == 253, 'Full input/q grid sizes')
    require(not audit.duplicated(['case_id', 'record_index']).any(), 'Duplicate audit record')
    require(not diagnostic.duplicated(['object', 'case_id', 'record_index']).any(), 'Duplicate diagnostic record')
    require(audit['all_pass'].astype(str).str.lower().isin(['true', '1']).all(), 'An input gate failed')
    require(audit['input_sha256'].str.fullmatch('[0-9a-f]{64}').all(), 'Invalid per-record input SHA')
    require(manifest['gates']['status'] == 'passed', 'Input manifest gate failed')
    for _, group in audit.groupby('case_id', sort=False):
        group = group.sort_values('record_index')
        require(np.array_equal(group.record_index, np.arange(n)), 'Input record grid')
        require(np.array_equal(group.ecg_id, reference['ids']) and np.array_equal(group.patient_id, reference['patient_ids']), 'Input cohort identity')
    valid_q = diagnostic.q.dropna()
    require(valid_q.between(-1e-14, 1 + 1e-14).all(), 'Subspace ratio outside [0,1]')
    zero = diagnostic.zero_norm.astype(str).str.lower().isin(['true', '1'])
    require(np.array_equal(zero, diagnostic.q.isna()), 'Null q must correspond exactly to zero norm')
    for row in summary.itertuples():
        selector = diagnostic.object.eq(row.object)
        for key in ('snr', 'noise_seed', 'mode'):
            value = getattr(row, key)
            selector &= diagnostic[key].isna() if pd.isna(value) else diagnostic[key].eq(value)
        values = diagnostic.loc[selector, 'q']
        require(len(values) == n and values.notna().sum() == row.n_valid and values.isna().sum() == row.n_null, 'q summary counts')
        finite = values.dropna().to_numpy()
        if len(finite):
            q1, median, q3 = np.quantile(finite, [.25, .5, .75])
            close([row.mean, row.sd, row.median, row.q1, row.q3, row.iqr, row.min, row.max],
                  [finite.mean(), finite.std(ddof=1), median, q1, q3, q3-q1, finite.min(), finite.max()], 'Independent q summary')
    sample = np.unique([0, 1, n // 2, n - 2, n - 1])
    clean = np.asarray(np.load(check_info(freeze['clean']), mmap_mode='r')[sample])
    bases = {}
    for case in manifest['cases']:
        key = case['noise_path']
        if key not in bases:
            bases[key] = np.asarray(np.load(resolve_path(key), mmap_mode='r')[sample])
    indexed_audit = audit.set_index(['case_id', 'record_index'])
    indexed_q = diagnostic.set_index(['object', 'case_id', 'record_index'])
    for case in manifest['cases']:
        base = bases[case['noise_path']]
        signs = np.asarray(case['sign_vector'], dtype=np.float32)
        factor = np.float32(10 ** (-float(case['snr']) / 20))
        designed = (base * signs[None, :, None]) * factor
        noisy = clean + designed
        residual = noisy.astype(np.float64) - clean.astype(np.float64)
        observed_snr = 10 * np.log10(np.sum(clean.astype(np.float64)**2, axis=(1, 2)) / np.sum(residual**2, axis=(1, 2)))
        family = 'S' if case['condition'].startswith('S_') else case['condition']
        for pos, record in enumerate(sample):
            line = indexed_audit.loc[(case['case_id'], record)]
            require(array_sha256(noisy[pos]) == line.input_sha256, 'Independent float32 final-input replay')
            close(line.achieved_snr_db, observed_snr[pos], 'Independent SNR replay', atol=1e-11)
        for kind, values in ((family + '_noise', residual), (family + '_input', noisy.astype(np.float64))):
            expected = np.sum((null_basis @ values)**2, axis=(1, 2)) / np.sum(values**2, axis=(1, 2))
            observed = [indexed_q.loc[(kind, case['case_id'], record), 'q'] for record in sample]
            close(observed, expected, 'Independent left-null-basis q replay', atol=1e-12)
        if family == 'S':
            reference_noise = base * factor
            close(np.abs(np.fft.rfft(designed.astype(np.float64), axis=-1))**2,
                  np.abs(np.fft.rfft(reference_noise.astype(np.float64), axis=-1))**2,
                  'Exact per-lead every-bin designed periodogram', atol=0)
    clean_q = np.sum((null_basis @ clean.astype(np.float64))**2, axis=(1, 2)) / np.sum(clean.astype(np.float64)**2, axis=(1, 2))
    close([indexed_q.loc[('clean', 'clean', record), 'q'] for record in sample], clean_q, 'Independent clean q')
    return {'n_validation_rows': len(audit), 'n_subspace_rows': len(diagnostic), 'q_summary_groups': len(summary),
            'all_record_producer_gates': manifest['gates'], 'independent_replay_records': sample.tolist(),
            'independent_replayed_conditions': 126, 'independent_replay_input_count': len(sample) * 126,
            'q_null_rows': int(diagnostic.q.isna().sum()),
            'scope': 'All-record producer audits and all q summaries checked; waveform/SVD/FFT independent replay is a deterministic five-record sample in every condition.'}


def summarize_independent(values):
    point = values[:, 0]
    draws = values[:, 1:].mean(axis=0)
    finite = draws[np.isfinite(draws)]
    endpoints = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
    return dict(estimate=point.mean(), seed_sd=point.std(ddof=1), patient_ci_low=endpoints[0], patient_ci_high=endpoints[1],
                n_bootstrap=len(draws), n_invalid=len(draws)-len(finite))


def verify_statistics(cfg, stage, freeze, manifest):
    paths = stage_paths(cfg, stage)
    log = read_json(paths['logs'] / 'statistics.json')
    merge = read_json(paths['logs'] / 'evaluation_merge.json')
    require(log['status'] == merge['status'] == 'completed', 'Incomplete statistics or merge')
    require(log['config_sha256'] == merge['config_sha256'] == cfg['_config_sha256'], 'Statistics/merge protocol mismatch')
    for info in log['outputs'].values():
        check_info(info)
    index = pd.read_csv(check_info(merge['outputs']['prediction_index']), keep_default_na=False)
    require(len(index) == 762 and not index.duplicated(['model', 'seed', 'case_id']).any(), '762 unique prediction units')
    require(index.condition.str.startswith('S_').sum() == 540 and index.condition.isin(['E', 'I']).sum() == 216 and index.condition.eq('clean').sum() == 6, 'Signed/reuse grid ownership')
    require(merge['dgx_overlap_seconds'] > 0, 'Two real concurrent DGX workers')
    distributions = read_json(check_info(log['outputs']['distribution_manifest']))
    cache = {(item['model'], item['seed'], item['case_id']): np.load(check_info(item['distribution']), allow_pickle=False)
             for item in distributions['metric_distributions']}
    paired = {(item['table'], item['model'], str(item.get('snr', '')), item['metric'], item.get('contrast', ''), item.get('condition', ''), item.get('mode', '')):
              np.load(check_info(item['distribution']), allow_pickle=False) for item in distributions['paired_distributions']}
    require(len(cache) == 762 and len(paired) == 474, 'Complete metric/paired distributions')
    reference = load_reference(cfg, stage)
    draws = np.load(check_info(freeze['draws']), allow_pickle=False)
    require(draws.dtype == np.int32 and draws.shape == (cfg['stages'][stage]['bootstrap_replicates'], freeze['n_patients']),
            'Frozen patient multiplicity matrix shape and dtype')
    require((draws >= 0).all() and np.all(draws.sum(axis=1) == freeze['n_patients']), 'Patient multiplicities, not record resampling')
    positive = np.zeros((freeze['n_patients'], 5), dtype=np.int64)
    np.add.at(positive, reference['patient_inverse'], reference['y'])
    negative = np.bincount(reference['patient_inverse'], minlength=freeze['n_patients'])[:, None] - positive
    invalid = np.r_[False, np.any((draws @ positive == 0) | (draws @ negative == 0), axis=1)]
    for values in cache.values():
        require(values.shape == (len(draws)+1, 3) and values.dtype == np.float64, 'Metric distribution shape and dtype')
        require(np.array_equal(np.isnan(values[:, 0]), invalid) and np.isfinite(values[:, 1:]).all(),
                'Common missing-class draws retained as NaN; finite F1/ECE')
    frames = {name: pd.read_csv(check_info(info), keep_default_na=False) for name, info in log['outputs'].items() if name != 'distribution_manifest'}
    lookup = {(c['snr'], c['noise_seed'], c['condition']): c['case_id'] for c in manifest['cases']}
    seeds, noises, modes = cfg['phase1_training_seeds'], cfg['phase1_noise_seeds'], cfg['conditions'][2:]
    checked = 0
    def check_row(table, selection, values, extra=None):
        nonlocal checked
        frame = frames[table]
        chosen = np.ones(len(frame), dtype=bool)
        for key, value in selection.items():
            chosen &= frame[key].astype(str).to_numpy() == str(value)
        require(chosen.sum() == 1, 'Unique statistics row ' + str(selection))
        row = frame.loc[chosen].iloc[0]
        for key, value in summarize_independent(values).items():
            close(float(row[key]), value, 'Independent aggregate ' + key)
        for key, value in (extra or {}).items():
            close(float(row[key]), value, 'Independent descriptive dispersion ' + key)
        key = (table, selection['model'], str(selection.get('snr', '')), selection['metric'], selection.get('contrast', ''), selection.get('condition', ''), selection.get('mode', ''))
        close(paired[key], values, 'Independent complete paired-draw array')
        checked += 1
    def check_points(table, metadata, expected):
        frame = frames[table]
        selected = np.ones(len(frame), dtype=bool)
        for key, value in metadata.items():
            selected &= frame[key].astype(str).to_numpy() == str(value)
        require(selected.sum() == 1, 'Unique detail row')
        close(float(frame.loc[selected, 'estimate'].iloc[0]), expected, 'Independent detail estimate')
    for model in cfg['models']:
        clean = np.stack([cache[(model, seed, 'clean')] for seed in seeds])
        for j, metric in enumerate(METRICS):
            check_row('absolute_metrics', dict(model=model, snr='', metric=metric, condition='clean', mode=''), clean[:, :, j])
        for snr in cfg['snrs']:
            condition = {c: np.stack([[cache[(model, seed, lookup[snr, noise, c])] for noise in noises] for seed in seeds]) for c in cfg['conditions']}
            signed = np.stack([condition[mode] for mode in modes], axis=2)
            contrast_arrays = {'E-S': condition['E'][:, :, None] - signed,
                               'S-I': signed - condition['I'][:, :, None],
                               'E-I': np.repeat((condition['E']-condition['I'])[:, :, None], 5, axis=2)}
            for contrast, raw in contrast_arrays.items():
                averaged = raw.mean(axis=2).mean(axis=1)
                for j, metric in enumerate(METRICS):
                    base = dict(model=model, snr=snr, metric=metric, contrast=contrast)
                    check_row('signed_control_summary', base, averaged[:, :, j], dict(
                        noise_sd=raw[:, :, :, 0, j].mean(axis=2).mean(axis=0).std(ddof=1),
                        signed_mode_sd=raw[:, :, :, 0, j].mean(axis=1).mean(axis=0).std(ddof=1)))
                    for s, seed in enumerate(seeds):
                        check_points('signed_control_seed_effects', dict(**base, seed=seed), averaged[s, 0, j])
                        for k, noise in enumerate(noises):
                            check_points('signed_control_noise_effects', dict(**base, seed=seed, noise_seed=noise), raw[s, k, :, 0, j].mean())
                        for k, mode in enumerate(modes):
                            check_points('signed_mode_effects', dict(**base, seed=seed, mode=mode), raw[s, :, k, 0, j].mean())
                    for k, mode in enumerate(modes):
                        check_row('signed_mode_summary', dict(**base, mode=mode), raw[:, :, k, :, j].mean(axis=1),
                                  dict(noise_sd=raw[:, :, k, 0, j].mean(axis=0).std(ddof=1)))
            condition['S'] = signed.mean(axis=2)
            for name, values in condition.items():
                for j, metric in enumerate(METRICS):
                    check_row('absolute_metrics', dict(model=model, snr=snr, metric=metric, condition=name, mode=name if name in modes else ''),
                              values[:, :, :, j].mean(axis=1), dict(noise_sd=values[:, :, 0, j].mean(axis=0).std(ddof=1),
                              signed_mode_sd=signed[:, :, :, 0, j].mean(axis=1).mean(axis=0).std(ddof=1) if name == 'S' else 0.0))
    old = pd.read_csv(legacy_paths(cfg, stage)['tables'] / 'new_summary.csv')
    old = old[(old.analysis == 'snr') & (old.outcome == 'structure_effect') & (old.source_id == 'standard')]
    for row in frames['signed_control_summary'].query("contrast == 'E-I'").itertuples():
        baseline = old[(old.model == row.model) & (old.snr == int(row.snr)) & (old.metric == row.metric)]
        require(len(baseline) == 1, 'Unique original E-I endpoint')
        for key in ('estimate', 'seed_sd', 'patient_ci_low', 'patient_ci_high', 'n_invalid'):
            close(float(getattr(row, key)), float(baseline.iloc[0][key]), 'Unchanged original E-I endpoint')
    # A separate consumer evaluates explicitly replicated patient records, not the weighted bootstrap implementation.
    from phase1_ecg_robustness.src.evaluate import classification_metrics
    replayed = 0
    for model in cfg['models']:
        for condition in ('E', 'I', 'S_00'):
            case = lookup[(0, noises[0], condition)]
            row = index[(index.model == model) & (index.seed == seeds[0]) & (index.case_id == case)].iloc[0]
            with np.load(resolve_path(row.prediction_path), allow_pickle=False) as archive:
                p, thresholds = archive['p'], archive['thresholds']
            for d in (0, len(draws)//2, len(draws)-1):
                counts = draws[d]
                records = np.repeat(np.arange(len(reference['y'])), counts[reference['patient_inverse']])
                y = reference['y'][records]
                if np.any(y.sum(axis=0) == 0) or np.any(y.sum(axis=0) == len(y)):
                    require(np.isnan(cache[(model, seeds[0], case)][d+1, 0]), 'Missing-class AUROC remains NaN')
                    continue
                measured, _ = classification_metrics(y, p[records], thresholds)
                close([measured[m] for m in METRICS], cache[(model, seeds[0], case)][d+1], 'Explicit record-replication bootstrap consumer', atol=2e-7)
                replayed += 1
    return {'predictions': 762, 'new_signed': 540, 'reused_E_I': 216, 'reused_clean': 6,
            'all_paired_arrays_verified': checked, 'all_detail_rows_verified': 1944, 'original_E_I_endpoints_verified': 18,
            'explicit_patient_draw_consumer_checks': replayed, 'n_invalid': log['n_invalid'],
            'cache_counts': log['cache_counts'], 'dgx_overlap_seconds': merge['dgx_overlap_seconds']}


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.numbers, self.links, self.ids = {}, [], set()
        self.active = None
    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            require(attrs['id'] not in self.ids, 'Duplicate HTML id')
            self.ids.add(attrs['id'])
        if tag == 'span' and attrs.get('id', '').startswith('number-'):
            self.active = attrs['id']
            self.numbers[self.active] = ''
        if tag in ('a', 'img'):
            self.links.append(attrs.get('href' if tag == 'a' else 'src', ''))
    def handle_endtag(self, tag):
        if tag == 'span':
            self.active = None
    def handle_data(self, data):
        if self.active:
            self.numbers[self.active] += data


def verify_publication(cfg, paths):
    report = read_json(paths['reports'] / 'manifest.json')
    require(report['status'] == 'completed' and report['config_sha256'] == cfg['_config_sha256'], 'Report manifest identity')
    html_path, markdown_path = check_info(report['html']), check_info(report['markdown'])
    document = Document()
    document.feed(html_path.read_text(encoding='utf-8'))
    markdown = markdown_path.read_text(encoding='utf-8')
    sources = {name: pd.read_csv(check_info(info)) for name, info in report['sources'].items() if str(info['path']).endswith('.csv')}
    for info in report['sources'].values():
        check_info(info)
    for cell in report['numeric_cells']:
        value = sources[cell['source']].iloc[int(cell['row'])][cell['column']]
        expected = ('null' if pd.isna(value) else format(float(value)*cell['scale'], ('+' if cell['signed'] else '') + '.' + str(cell['digits']) + cell.get('notation', 'f')))
        require(cell['text'] == expected and document.numbers[cell['id']] == expected, 'HTML numeric cell does not round-trip to CSV')
    require(len(document.numbers) == len(report['numeric_cells']), 'Untracked HTML numeric cells')
    for table in report['markdown_tables']:
        match = re.search(r'<!-- table:' + re.escape(table['key']) + r' -->\s*(.*?)\s*<!-- /table:' + re.escape(table['key']) + r' -->', markdown, re.S)
        require(match is not None, 'Missing Markdown table')
        lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
        rows = [[cell.strip().replace('\\|', '|').replace('\\\\', '\\').replace('<br>', '\n')
                 for cell in re.split(r'(?<!\\)\|', line.strip('|'))] for line in lines]
        require(rows[0] == table['headers'] and rows[2:] == table['rows'], 'Markdown table diverged from shared numeric cells')
    from urllib.parse import unquote
    for link in document.links:
        if not link or link.startswith(('https:', 'http:', 'mailto:')):
            continue
        if link.startswith('#'):
            require(link[1:] in document.ids, 'Broken report anchor')
        else:
            require((html_path.parent / unquote(link.split('#')[0])).is_file(), 'Broken local report link: ' + link)
    figures = read_json(paths['figures'] / 'manifest.json')
    required = {'signed_control_structure_effects', 'signed_control_snr_curves', 'signed_control_matrices',
                'subspace_q_distributions', 'subspace_geometry', 'input_validation'}
    require({item['id'] for item in figures['figures']} == required and figures['figure_count'] == 6, 'All six required figures')
    from PIL import Image
    for figure in figures['figures']:
        require(set(figure['files']) == {'png', 'pdf', 'svg'}, 'Three required figure formats')
        for info in figure['files'].values():
            check_info(info)
        with Image.open(resolve_path(figure['files']['png']['path'])) as image:
            image.verify()
        for source in figure['sources']:
            check_info(source)
    return dict(numeric_cells=len(report['numeric_cells']), markdown_tables=len(report['markdown_tables']),
                figures=6, figure_files=18, local_links_checked=len(document.links),
                reports={'html': file_info(html_path), 'markdown': file_info(markdown_path)},
                browser_visual_acceptance='Recorded separately in browser_verification.json; not inferred from file checks')


def run(config=None, stage='full', publication=False):
    started = time.perf_counter()
    cfg = load_config(config)
    paths = stage_paths(cfg, stage)
    destination = paths['logs'] / ('publication_verification.json' if publication else stage + '_verification.json')
    result = dict(status='running', stage=stage, config_sha256=cfg['_config_sha256'],
                  started_at=datetime.now(timezone.utc).isoformat(), implementation=file_info(Path(__file__)))
    save_json(destination, result)
    try:
        freeze = require_freeze(cfg, stage)
        if publication:
            require(read_json(paths['logs'] / 'full_verification.json')['status'] == 'passed', 'Full numeric acceptance required')
            result['publication'] = verify_publication(cfg, paths)
        else:
            manifest = load_manifest(cfg, stage)
            for info in freeze['protected_files']:
                check_info(info)
            basis, result['matrices'] = verify_matrices(cfg, freeze)
            result['inputs'] = verify_inputs(cfg, stage, freeze, manifest, basis)
            result['statistics'] = verify_statistics(cfg, stage, freeze, manifest)
            result['smoke_authorization'] = read_json(paths['logs'] / 'smoke_verification.json') if stage == 'full' else None
            result['sources'] = {name: file_info(path) for name, path in dict(
                freeze=paths['logs']/'freeze.json', inputs=paths['inputs']/'manifest.json',
                merge=paths['logs']/'evaluation_merge.json', statistics=paths['logs']/'statistics.json').items()}
        result.update(status='passed', elapsed_seconds=time.perf_counter()-started, finished_at=datetime.now(timezone.utc).isoformat())
        save_json(destination, result)
        print(f"verification-passed stage={stage} publication={publication} seconds={result['elapsed_seconds']:.2f}", flush=True)
        return result
    except Exception as exc:
        result.update(status='failed', elapsed_seconds=time.perf_counter()-started, error=f'{type(exc).__name__}: {exc}')
        save_json(destination, result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--stage', choices=('smoke', 'full'), default='full')
    parser.add_argument('--publication', action='store_true')
    args = parser.parse_args()
    run(args.config, args.stage, args.publication)
