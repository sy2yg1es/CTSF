import json
import sys

tags = [
    'ECL_H384_GA8_noadapter',
    'ECL_H384_GA8_adapter32',
    'ECL_H384_GA8_adapter64',
    'Traffic_H384_noadapter',
    'Traffic_H384_adapter32',
]

for tag in tags:
    try:
        with open(f'logs/monitor/monitor_{tag}.json') as f:
            data = json.load(f)
    except Exception as e:
        print(f"=== {tag}: {e} ===\n")
        continue

    print(f"=== {tag} ({len(data)} updates) ===")

    # First 3 and last 3
    for r in data[:3] + data[-3:]:
        u = r['update']
        print(f"  u={u:4d} loss={r['loss_task']:.4f} var50={r['loss_var_50']:.6f} "
              f"gP={r['grad_norm_prompt']:.4f} gA={r['grad_norm_adapter']:.4f} "
              f"dP={r['param_delta_prompt']:.6f} dA={r['param_delta_adapter']:.6f}")

    # Averages
    n = len(data)
    avg = lambda k: sum(r[k] for r in n and data) / n

    avg_gP = sum(r['grad_norm_prompt'] for r in data) / n
    avg_gA = sum(r['grad_norm_adapter'] for r in data) / n
    avg_dP = sum(r['param_delta_prompt'] for r in data) / n
    avg_dA = sum(r['param_delta_adapter'] for r in data) / n
    avg_var = sum(r['loss_var_50'] for r in data) / n
    avg_loss = sum(r['loss_task'] for r in data) / n

    print(f"  AVG: loss={avg_loss:.4f} var50={avg_var:.6f}")
    print(f"  AVG: grad_prompt={avg_gP:.4f} grad_adapter={avg_gA:.4f}")
    print(f"  AVG: delta_prompt={avg_dP:.6f} delta_adapter={avg_dA:.6f}")
    print()
