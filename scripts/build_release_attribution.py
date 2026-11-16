#!/usr/bin/env python3
"""Per-image attribution and a per-source licence table for a release of a model trained on data_v13.

Joins data_v13/_index.tsv with the two 2026-11-12 licence audits and the image sidecars:

  rows with a licence in the index (collected 2026-11-12/13, and the generated text renders): licence and tier
      from the index, author / origin / attribution text from the sidecar;
  rows from the labelled part of data_v12: data_gen/license_audit_labelled/manifest.tsv;
  rows from the 11 legacy folders of data_v12: data_gen/license_audit/manifest.tsv;
  set-level attribution text for both: data_gen/license_audit_labelled/attribution.tsv.

Writes
  release/attribution.tsv.gz   one row per training image: image, source, license, status, author, title, url,
                               attribution
  release/SOURCES.md           one row per source folder: images, licences, status, upstream
  release/sources.json         the same numbers plus totals by licence family (for the model card)

status: "verified" (licence read from the upstream licence file or page), "conditions" (verified, carries
attribution, share-alike, copyleft or a trademark note), "stated" (the licence the source or its mirror states;
upstream page not fetched), "generated" (drawn by scripts/gen_rendered_text.py).

  python3 scripts/build_release_attribution.py
"""

import collections
import csv
import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data_v13"
OUT = REPO / "release"
csv.field_size_limit(10**9)

FAMILIES = ["Public domain / CC0", "Permissive (keep notice)", "Attribution", "Share-alike / copyleft", "Other"]


def family_one(lic):
    s = lic.upper()
    if re.search(r"CC0|PUBLIC ?DOMAIN|PDM|UNLICENSE|WTFPL", s):
        return 0
    if re.search(r"-SA|GPL|MPL|OFL|EPL|CDDL|FREE ?ART|ARTISTIC", s):
        return 3
    if re.search(r"CC-BY|OGA-BY|CC BY", s):
        return 2
    if re.search(r"MIT|APACHE|BSD|ISC|ZLIB|BSL|EXPAT|X11|HPND", s):
        return 1
    return 4


def family(lic):
    """Licence family of an SPDX-like expression: the strictest term of an AND, the most permissive of an OR."""
    alts = [max(family_one(t) for t in re.split(r"\s+AND\s+", alt)) for alt in re.split(r"\s+OR\s+", lic)]
    return FAMILIES[min(alts)]


def upstream(url):
    m = re.match(r"(https?://(?:github|gitlab|codeberg)\.[a-z]+/[^/]+/[^/#?]+)", url)
    if m:
        return m.group(1)
    m = re.match(r"(https?://[^/]+)", url)
    return m.group(1) if m else ""


def sidecar(dst):
    try:
        return json.loads((DATA / f"{dst}.json").read_text())
    except (OSError, ValueError):
        return {}


def clean(v):
    return " ".join(str(v).split()) if v else ""


def main():
    index = list(csv.DictReader(open(DATA / "_index.tsv"), delimiter="\t"))
    lab = {r["path"]: r for r in csv.DictReader(open(REPO / "data_gen/license_audit_labelled/manifest.tsv"),
                                                delimiter="\t")}
    leg = {r["path"]: r for r in csv.DictReader(open(REPO / "data_gen/license_audit/manifest.tsv"), delimiter="\t")}
    # set-level attribution text of both audits, keyed by the manifests' sub_collection
    sets = {r["scope"]: r for r in csv.DictReader(open(REPO / "data_gen/license_audit_labelled/attribution.tsv"),
                                                  delimiter="\t")}
    with ThreadPoolExecutor(16) as pool:
        sidecars = list(pool.map(sidecar, (r["dst"] for r in index), chunksize=2000))

    OUT.mkdir(exist_ok=True)
    per = collections.defaultdict(lambda: {"n": 0, "lic": collections.Counter(), "status": collections.Counter(),
                                           "up": collections.Counter()})
    fam = collections.Counter()
    fam_status = collections.Counter()
    missing = 0
    with gzip.open(OUT / "attribution.tsv.gz", "wt", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["image", "source", "license", "status", "author", "title", "url", "attribution"])
        for r, sc in zip(index, sidecars):
            origin = sc.get("origin")
            url = origin if isinstance(origin, str) and origin.startswith("http") else ""
            author, title = clean(sc.get("author")), clean(sc.get("entry") or sc.get("name"))
            attribution = clean(sc.get("attribution"))
            m = None
            if r["license"]:
                lic = r["license"]
                status = "generated" if r["group"] == "generated" else {"A": "verified"}.get(r["tier"], "conditions")
            elif r["src"] in lab:
                m = lab[r["src"]]
                lic = m["verified_license_spdx"]
                status = m["verdict"]
                if not lic or lic == "UNVERIFIED" or status == "unknown":
                    lic, status = m["sidecar_license"] or lic, "stated"
                author = author or clean(m["author"])
                url = url or m["source_url"]
            elif r["src"] in leg:
                m = leg[r["src"]]
                lic, status = m["proposed_license"], m["verdict"]
            else:
                lic, status = "", "stated"
                missing += 1
            if m and m["sub_collection"] in sets:
                a = sets[m["sub_collection"]]
                author, url = author or clean(a["author"]), url or a["source_url"]
                attribution = attribution or clean(a["attribution_text"])
            w.writerow([r["dst"], r["source"], lic, status, author, title, url, attribution])
            p = per[r["source"]]
            p["n"] += 1
            p["lic"][lic] += 1
            p["status"][status] += 1
            if url:
                p["up"][upstream(url)] += 1
            fam[family(lic)] += 1
            fam_status[(family(lic), status)] += 1

    order = {"generated": 0, "verified": 1, "conditions": 2, "stated": 3}
    lines = ["# Sources of the training images", "",
             f"{len(index):,} images from {len(per)} source folders. One row per folder; the licence and "
             "attribution of every image are in `attribution.tsv.gz`. Status: *verified* means the licence was "
             "read from the upstream licence file or page; *conditions* means verified with attribution, "
             "share-alike, copyleft or a trademark note; *stated* means the licence the source or its mirror "
             "states, with the upstream page not fetched; *generated* means drawn by a script in this "
             "repository. Licence checking was best effort and is not legal advice.", "",
             "| Source | Images | Licences (images) | Status (images) | Upstream |", "|---|---:|---|---|---|"]
    summary = {}
    for src, p in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        lic = "; ".join(f"{k or 'not recorded'} ({v:,})" for k, v in p["lic"].most_common(4))
        if len(p["lic"]) > 4:
            lic += f"; {len(p['lic']) - 4} more"
        status = "; ".join(f"{k} ({v:,})" for k, v in sorted(p["status"].items(), key=lambda kv: order[kv[0]]))
        up = p["up"].most_common(1)
        up = up[0][0] if up and up[0][1] >= 0.5 * p["n"] else ""
        lines.append(f"| {src} | {p['n']:,} | {lic} | {status} | {up} |")
        summary[src] = {"images": p["n"], "licenses": dict(p["lic"]), "status": dict(p["status"]), "upstream": up}
    (OUT / "SOURCES.md").write_text("\n".join(lines) + "\n")
    totals = {"images": len(index), "sources": len(per), "missing_licence_record": missing,
              "by_family": {k: fam[k] for k in FAMILIES},
              "by_family_status": {f"{k[0]} | {k[1]}": v for k, v in sorted(fam_status.items())},
              "by_status": dict(collections.Counter(s for p in per.values() for s in p["status"].elements()))}
    (OUT / "sources.json").write_text(json.dumps({"totals": totals, "sources": summary}, indent=1))
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
