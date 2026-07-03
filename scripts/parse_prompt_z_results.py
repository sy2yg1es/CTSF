import json, os

log_dir = 'logs/prompt_z'
results = {}

for f in sorted(os.listdir(log_dir)):
    if f.endswith('.json') and not f.startswith('train_'):
        name = f.replace('.json', '')
        d = json.load(open(os.path.join(log_dir, f)))
        results[name] = d

# Summary table
header = f"{'Experiment':<35} {'MSE':>10} {'MAE':>10} {'n':>8}"
print(header)
print('-' * len(header))
for name in sorted(results):
    d = results[name]
    mse = d.get('MSE', float('nan'))
    mae = d.get('MAE', float('nan'))
    n = d.get('n_aligned', 0)
    print(f"{name:<35} {mse:>10.6f} {mae:>10.6f} {n:>8}")

# Training logs
print("\n=== Training Logs ===")
for f in sorted(os.listdir(log_dir)):
    if f.endswith('.json') and f.startswith('train_'):
        name = f.replace('.json', '')
        d = json.load(open(os.path.join(log_dir, f)))
        loss = d.get("loss", 0)
        fl = d.get("forecast_loss", 0)
        gm = d.get("gamma_mean", 0)
        dh = d.get("effective_delta_ratio", d.get("raw_delta_to_hidden_ratio", 0))
        print(f"  {name}: loss={loss:.6f} forecast={fl:.6f} gamma={gm:.4f} eff_d/h={dh:.6f}")

# Diagnostics
print("\n=== Prompt-Z Diagnostics ===")
for name in ['ECL_H96_pz_mode0', 'Traffic_H1_pz_mode0', 'ETTh1_H24_pz_mode0',
             'ECL_H96_pz_mode1', 'Traffic_H1_pz_mode1', 'ETTh1_H24_pz_mode1']:
    if name in results:
        d = results[name]
        print(f"\n  {name}:")
        for k in ['gamma_mean_mean', 'gamma_mean_std', 'gamma_p10', 'gamma_p50', 'gamma_p90',
                   'mask_ratio_mean', 'mask_mean_mean', 'raw_delta_to_hidden_ratio_mean',
                   'effective_delta_ratio_mean', 'raw_delta_norm_mean', 'hidden_norm_mean']:
            if k in d:
                print(f"    {k}: {d[k]:.6f}")
