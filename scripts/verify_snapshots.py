"""
Verify all three prediction snapshots are valid.
Usage: python scripts/verify_snapshots.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SNAPSHOTS = {
    "pre_tournament":   {"expect_alive": 48, "expect_completed": 0},
    "post_group_stage": {"expect_alive": (16, 48), "expect_completed": 72},
    "post_r32":         {"expect_alive": 16, "expect_completed": 88},
}

all_ok = True

for stage, rules in SNAPSHOTS.items():
    path = ROOT / "data" / "predictions" / "snapshots" / f"{stage}.json"
    if not path.exists():
        print(f"MISSING  {stage}.json")
        all_ok = False
        continue

    with open(path) as f:
        d = json.load(f)

    meta = d.get("snapshot_metadata", {})
    tp   = d.get("team_probabilities", {})
    ar   = d.get("actual_results", {})

    champ_sum  = sum(v.get("p_champion", 0) for v in tp.values())
    alive      = sum(1 for v in tp.values() if v.get("p_champion", 0) > 0)
    completed  = len(ar)
    reconstructed = d.get("metadata", {}).get("reconstructed", False)

    # Check alive count
    exp_alive = rules["expect_alive"]
    if isinstance(exp_alive, tuple):
        alive_ok = exp_alive[0] <= alive <= exp_alive[1]
    else:
        alive_ok = alive == exp_alive

    sum_ok = abs(champ_sum - 1.0) < 0.02
    comp_ok = completed >= rules["expect_completed"]

    status = "OK " if (alive_ok and sum_ok and comp_ok) else "FAIL"
    if status == "FAIL":
        all_ok = False

    rec_tag = " [reconstructed]" if reconstructed else ""
    print(
        f"{status}  {stage:<22}{rec_tag}\n"
        f"       sum(p_champion)={champ_sum:.4f}  "
        f"alive={alive}  "
        f"completed_matches={completed}"
    )
    if not alive_ok:
        print(f"       FAIL: expected alive={exp_alive}, got {alive}")
    if not sum_ok:
        print(f"       FAIL: sum(p_champion)={champ_sum:.4f}, expected ~1.0")
    if not comp_ok:
        print(f"       FAIL: expected >={rules['expect_completed']} completed, got {completed}")
    print()

if all_ok:
    print("All snapshots valid.")
else:
    print("One or more snapshots failed verification.")
    sys.exit(1)