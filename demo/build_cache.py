"""Build the demo's vendored cache from the primary sources.

Run this only to regenerate `demo/data/`. The demo itself reads the cache, so it needs
neither of these downloads nor a single third-party package.

Sources
-------
Norman et al. 2019, "Exploring genetic interaction manifolds constructed from rich
single-cell phenotypes", Science 365(6455) -- GEO accession **GSE133344**. Three files:
`GSE133344_filtered_matrix.mtx.gz` (33,694 genes x 111,668 cells, 362M nonzero entries),
`GSE133344_filtered_genes.tsv.gz`, `GSE133344_filtered_barcodes.tsv.gz`, and
`GSE133344_filtered_cell_identities.csv.gz`.

CollecTRI (Muller-Dott et al. 2023), the transcription-factor regulon collection, as a
directed interaction table with `source_genesymbol`, `target_genesymbol`, `is_directed`,
and the constituent-resource `sources` column.

What is kept
------------
Only the genes the demo names, and only cells whose guide identity resolves to a single
perturbed gene or to a non-targeting control. Doubles are dropped: a two-gene arm is a
different query shape and the demo does not ask it. The matrix is streamed once, so this
costs one pass over the 362M entries and no memory beyond the kept genes.

Usage
-----
    python3 demo/build_cache.py --norman-dir DIR --collectri PATH
"""

from __future__ import annotations

import argparse
import csv
import gzip
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"

GENES = (
    # The erythroid / myeloid programme the demo authors a graph over.
    "KLF1", "GATA1", "TAL1", "ZFPM1", "NFE2", "GATA2", "LMO2", "EPOR", "HBG1", "HBB",
    "ALAS2", "SLC4A1", "AHSP", "BCL11A", "MYB", "SPI1", "CEBPA", "RUNX1", "LYL1", "STAT5A",
    # Proliferation markers, kept because they are the covariates every single-cell
    # pipeline adjusts for and every one of them is measured *after* the perturbation.
    "MKI67", "CCNB1", "PCNA",
)


def _resolve_arm(guide_identity: str) -> str | None:
    """The perturbed gene, `NTC` for a pure control, or None for a two-gene arm."""
    perturbed = [
        gene
        for gene in guide_identity.split("__")[0].split("_")
        if not gene.startswith("NegCtrl")
    ]
    if not perturbed:
        return "NTC"
    return perturbed[0] if len(perturbed) == 1 else None


def build_cells(norman_dir: pathlib.Path, out: pathlib.Path) -> int:
    rows_wanted = {}
    with gzip.open(norman_dir / "GSE133344_filtered_genes.tsv.gz", "rt") as handle:
        for number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1] in GENES:
                rows_wanted[number] = parts[1]
    missing = set(GENES) - set(rows_wanted.values())
    if missing:
        raise SystemExit(f"genes absent from the matrix: {sorted(missing)}")

    counts: dict[int, dict[str, int]] = defaultdict(dict)
    totals: dict[int, int] = defaultdict(int)
    with gzip.open(norman_dir / "GSE133344_filtered_matrix.mtx.gz", "rt") as handle:
        for index, line in enumerate(handle):
            if index < 3:  # MatrixMarket banner, comment, dimensions
                continue
            gene_s, cell_s, value_s = line.split()
            cell = int(cell_s)
            totals[cell] += int(value_s)
            gene = int(gene_s)
            if gene in rows_wanted:
                counts[cell][rows_wanted[gene]] = int(value_s)

    barcodes = [
        line.strip()
        for line in gzip.open(norman_dir / "GSE133344_filtered_barcodes.tsv.gz", "rt")
    ]
    meta: dict[str, tuple[str, str]] = {}
    with gzip.open(norman_dir / "GSE133344_filtered_cell_identities.csv.gz", "rt") as handle:
        for record in csv.DictReader(handle):
            if record["good_coverage"] != "True":
                continue
            arm = _resolve_arm(record["guide_identity"])
            if arm is not None:
                meta[record["cell_barcode"]] = (arm, record["gemgroup"])

    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["arm", "gemgroup", "total_umi", *GENES])
        for column, barcode in enumerate(barcodes, start=1):
            entry = meta.get(barcode)
            total = totals.get(column, 0)
            if entry is None or total == 0:
                continue
            arm, gemgroup = entry
            cell = counts.get(column, {})
            writer.writerow([arm, gemgroup, total, *(cell.get(g, 0) for g in GENES)])
            written += 1
    return written


def build_edges(collectri: pathlib.Path, out: pathlib.Path) -> int:
    """Directed edges among the demo's genes, carrying the constituent-resource tags.

    The resource tags are kept because the demo's cycle finding rests on them: cycles that
    survive restriction to TRRUST alone, or to DoRothEA-A alone, are not an artefact of
    aggregating databases with different context coverage.
    """
    kept = []
    with open(collectri, newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            if record["is_directed"] != "True":
                continue
            source, target = record["source_genesymbol"], record["target_genesymbol"]
            if source in GENES and target in GENES and source != target:
                kept.append((source, target, record["sources"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "target", "resources"])
        writer.writerows(sorted(set(kept)))
    return len(set(kept))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norman-dir", type=pathlib.Path, required=True)
    parser.add_argument("--collectri", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    cells = build_cells(args.norman_dir, DATA / "norman_cells.tsv.gz")
    print(f"wrote {DATA / 'norman_cells.tsv.gz'}: {cells:,} cells x {len(GENES)} genes")
    edges = build_edges(args.collectri, DATA / "collectri_edges.tsv")
    print(f"wrote {DATA / 'collectri_edges.tsv'}: {edges} directed edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
