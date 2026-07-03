import json, os, glob

files = sorted(glob.glob('logs/expert_diag/*.json'))
for f in files:
    name = os.path.basename(f).replace('.json','')
    d = json.load(open(f))
    div = d['expert_diversity']
    agg = d['aggregated']
    
    cos_mean = div['mean_pairwise_cosine_sim']
    cos_max = div['max_pairwise_cosine_sim']
    norm_mean = div['expert_norms']['mean']
    norm_min = div['expert_norms']['min']
    norm_max = div['expert_norms']['max']
    ent_ratio = agg['entropy_ratio']['mean']
    ent_std = agg['entropy_ratio']['std']
    top1 = agg['top1_oracle_prob_mean']['mean']
    top1_std = agg['top1_oracle_prob_mean']['std']
    lcv = agg['loss_cv_mean']['mean']
    lcv_std = agg['loss_cv_mean']['std']
    lvar = agg['loss_var_mean']['mean']
    noop = agg['noop_is_best_ratio']['mean'] * 100
    verdict = d['verdict']
    
    print(f'=== {name} ===')
    print(f'  Cosine sim:      mean={cos_mean:.4f}  max={cos_max:.4f}')
    print(f'  Expert norm:     mean={norm_mean:.4f}  [{norm_min:.4f}, {norm_max:.4f}]')
    print(f'  Entropy ratio:   {ent_ratio:.4f} +/- {ent_std:.4f}  (1.0=uniform)')
    print(f'  Top-1 oracle p:  {top1:.4f} +/- {top1_std:.4f}')
    print(f'  Loss CV:         {lcv:.6f} +/- {lcv_std:.6f}')
    print(f'  Loss variance:   {lvar:.8f}')
    print(f'  Noop best:       {noop:.1f}%')
    print(f'  VERDICT:         {verdict}')
    print()
