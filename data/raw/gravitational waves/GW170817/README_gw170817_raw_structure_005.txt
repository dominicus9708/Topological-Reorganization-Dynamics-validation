GW170817 raw-data folder structure (005)

metadata/
- Event metadata JSON obtained from the official GWOSC event detail API.
- GRB 170817A timing note from NASA GCN.
- Strain listing debug dumps.

strain/
- H1, L1, V1 strain files downloaded from the confirmed GWOSC listing.
- Default target in this version: 4096 s, 4 kHz, HDF.

registry/
- Snapshot of the internal source registry used by the pipeline.

manifest/
- Download status, file paths, sizes, and hashes.
