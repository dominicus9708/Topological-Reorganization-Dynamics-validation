Run from repository root on Windows PowerShell:

python src\dimensional_structure_response\run_d3_prestructure_exchange_repro_001.py `
  --input-config data\derived\dimensional_structure_response\input\d3_prestructure_exchange\d3_prestructure_exchange_scenarios_001.csv `
  --output results\dimensional_structure_response\output\d3_prestructure_exchange\20260806_132400 `
  --write-grid

Required Python packages:
- numpy
- pandas

Model convention:
- base universe dimension remains D3
- channels 4, 5, and 6 are exchange pathways, not additional realized dimensions
- default direct particle-domain overlap is zero
