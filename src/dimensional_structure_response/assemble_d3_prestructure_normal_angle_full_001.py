from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = HERE / "d3_normal_angle_full_parts"
OUTPUT = HERE / "run_d3_prestructure_normal_angle_repro_001.py"

names = [f"run_d3_prestructure_normal_angle_repro_001.py.part{i:02d}" for i in range(5)]
missing = [name for name in names if not (PARTS / name).exists()]
if missing:
    raise FileNotFoundError(f"missing source parts: {missing}")

OUTPUT.write_bytes(b"".join((PARTS / name).read_bytes() for name in names))
print(OUTPUT)
