"""Full-benchmark plumbing for VLM accuracy evaluation (N-041).

Modules:
  pope_full       — full POPE adversarial split loader + F1/accuracy scorer
  naturalbench_full — full NaturalBench loader + Group-Acc scorer
  hallusionbench  — HallusionBench loader + aAcc/qAcc/fAcc scorer
  amber           — AMBER discriminative loader + F1 scorer
  run_bench       — single CLI runner (with optional L0 intervention)
"""
