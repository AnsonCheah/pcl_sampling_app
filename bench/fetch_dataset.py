"""Download benchmark part meshes into ``mesh_raw/``.

Run::

    python bench/fetch_dataset.py --dataset tless
    python bench/fetch_dataset.py --dataset tless --list

Why T-LESS rather than Sileane
    Sileane is the better-motivated bin-picking dataset on paper -- it was built around
    symmetry groups -- but its meshes sit inside 7-zip archives in a *Google Drive folder*
    with no direct file URLs, which needs ``gdown`` plus ``py7zr`` (neither installed) and
    still hits Drive's interstitial on large files.  T-LESS is one plain 33 MB zip over
    HTTPS, is **CC BY 4.0** rather than non-commercial, and carries 30 industrial objects
    against Sileane's ~10 -- which matters directly, because the breadth pass wants many
    parts more than it wants many scenes per part.

    Sileane is still worth adding later for its published symmetry-group formalism; it just
    is not the thing to block the first sweep on.

What arrives
    ``mesh_raw/<dataset>/obj_*.ply`` plus ``models_info.json``.  The pipeline reads PLY
    directly (``ImportMeshStage`` calls ``o3d.io.read_triangle_mesh``) and detects the
    millimetre units BOP uses via ``geometry.mesh_repair``'s ``unit_scale``, so no
    conversion step is needed.

``models_info.json`` is the quiet prize here: it carries per-object ``symmetries_discrete``
and ``symmetries_continuous``, which is published ground truth for symmetry.  That gives the
benchmark a symmetry-aware metric without hand-rolling one, and gives
``geometry/ambiguity.py`` its first external validation set -- until now it has only been
checked against primitives whose answers we wrote ourselves.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MESH_ROOT = os.path.join(_ROOT, "mesh_raw")

DATASETS = {
    "tless": {
        "url": "https://huggingface.co/datasets/bop-benchmark/tless/resolve/main/tless_models.zip",
        "license": "CC BY 4.0",
        "attribution": (
            "T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects.\n"
            "Hodan, Haluza, Obdrzalek, Matas, Lourakis, Zabulis. WACV 2017.\n"
            "https://cmp.felk.cvut.cz/t-less/  --  Licensed CC BY 4.0.\n"
            "Obtained via the BOP benchmark: https://bop.felk.cvut.cz/datasets/\n"
        ),
        # BOP ships several model variants; the reconstructed ones are scans and the
        # CAD ones are the manually authored meshes. Prefer CAD: they are watertight, which
        # is what VHACD and the MuJoCo collision model need.
        "prefer": "models_cad",
    },
}


def _download(url: str, dest: str, chunk: int = 1 << 20) -> None:
    if os.path.exists(dest):
        print(f"  already downloaded: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return
    print(f"  GET {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest + ".part", "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                done += len(block)
                if total:
                    pct = 100.0 * done / total
                    print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)",
                          end="", flush=True)
        print()
    os.replace(dest + ".part", dest)


def fetch(name: str, force: bool = False) -> str:
    spec = DATASETS[name]
    out_dir = os.path.join(MESH_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    archive = os.path.join(out_dir, os.path.basename(spec["url"]))
    _download(spec["url"], archive)

    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        # Pick one model variant; mixing scans and CAD in the same sweep would confound
        # mesh quality with the thing being measured.
        prefer = spec.get("prefer")
        variants = sorted({n.split("/")[0] for n in names if "/" in n})
        chosen = prefer if prefer in variants else (variants[0] if variants else "")
        wanted = [n for n in names
                  if n.startswith(chosen + "/") and n.endswith((".ply", ".json"))]
        if not wanted:
            wanted = [n for n in names if n.endswith((".ply", ".json"))]
        print(f"  variants in archive: {variants or '(flat)'} -> using {chosen or '(flat)'}")
        for n in wanted:
            target = os.path.join(out_dir, os.path.basename(n))
            if os.path.exists(target) and not force:
                continue
            with z.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())

    with open(os.path.join(out_dir, "ATTRIBUTION.txt"), "w") as f:
        f.write(spec["attribution"])

    plys = sorted(f for f in os.listdir(out_dir) if f.endswith(".ply"))
    print(f"  extracted {len(plys)} meshes to {out_dir}  [{spec['license']}]")
    _summarise(out_dir, plys)
    return out_dir


def _summarise(out_dir: str, plys) -> None:
    """Report sizes and the published symmetry annotations, if present."""
    info_path = os.path.join(out_dir, "models_info.json")
    if not os.path.exists(info_path):
        print("  (no models_info.json -- no published symmetry annotations)")
        return
    with open(info_path) as f:
        info = json.load(f)

    n_disc = n_cont = 0
    print(f"\n  {'object':<14} {'diameter_mm':>12} {'symmetry':>22}")
    for key in sorted(info, key=lambda k: int(k)):
        rec = info[key]
        disc = len(rec.get("symmetries_discrete", []))
        cont = len(rec.get("symmetries_continuous", []))
        n_disc += bool(disc)
        n_cont += bool(cont)
        sym = ("continuous" if cont else f"{disc} discrete" if disc else "none")
        print(f"  obj_{int(key):06d}  {rec.get('diameter', 0):12.1f} {sym:>22}")
    print(f"\n  {len(info)} objects: {n_cont} with continuous symmetry, "
          f"{n_disc} with discrete, {len(info) - n_cont - n_disc} asymmetric")
    print("  -> these annotations are the external validation set for geometry/ambiguity.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="tless", choices=sorted(DATASETS))
    ap.add_argument("--list", action="store_true", help="show what is available and exit")
    ap.add_argument("--force", action="store_true", help="re-extract over existing files")
    args = ap.parse_args()

    if args.list:
        for name, spec in DATASETS.items():
            print(f"{name:<10} {spec['license']:<12} {spec['url']}")
        return

    print(f"Fetching {args.dataset} -> {MESH_ROOT}")
    fetch(args.dataset, force=args.force)


if __name__ == "__main__":
    main()
