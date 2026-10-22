"""``mind`` command-line interface for working with ``.mind`` files.

    mind create  <file> [--modality M]
    mind ingest  <file> (--npy V.npy [--ids ids.txt] | --text lines.txt [--dim D]) [--space S]
    mind cluster <file> [--k K] [--space S] [--collection C] [--note N]
    mind evolve  <file> [--k K] [--space S] [--collection C] [--note N]
    mind graph   <file> [--k K] [--space S]
    mind query   <file> (--item ID | --text STR | --npy q.npy) [--k K] [--space S]
    mind explain <file> --item ID [--space S]
    mind recon-store   <file> <subject_dir> [--profile lossless|analysis] [--effort E]
    mind recon-restore <file> <dest> --item ID     # rebuild the tree byte-for-byte
    mind recon-files   <file> --item ID
    mind recon-verify  <file> [--item ID] [--deep]
    mind checkpoint <file> NAME [--note N]
    mind verify  <file> [NAME]          # NAME -> verify save point; else byte integrity
    mind validate <file>                # semantic / structural validation
    mind inspect <file>
    mind history <file>
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import allocate as _allocate
from . import audit as _audit
from . import checkpoint as _cp
from . import cluster as _cluster
from . import collections as _collections
from . import evolve as _evolve
from . import findings as _findings
from . import fuse as _fuse
from . import graph as _graph
from . import ingest as _ingest
from . import lazy as _lazy
from . import query as _query
from . import reconall as _reconall
from . import registry as _registry
from . import render as _render
from . import retrieve as _retrieve
from . import sign as _sign
from . import spaces as _spaces
from . import stats as _stats
from . import summary as _summary
from . import validate as _validate
from .embedders import HashEmbedder
from .file import MindFile
from .index import load_all_vectors


def _attach_reconstructable_embedder(mf: MindFile, space: str) -> None:
    sp = _spaces.get_space(mf, space)
    if sp and (sp["embedder_id"] or "").startswith("synthetic-hash") and sp["dim"]:
        mf.embedder = HashEmbedder(dim=sp["dim"], id=sp["embedder_id"])


def _pj(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_create(a) -> None:
    mf = MindFile.create(a.file, modality=a.modality, overwrite=a.overwrite)
    print(f"created {mf.path} (uuid {mf.meta_get('file_uuid')})")
    mf.close()


def cmd_ingest(a) -> None:
    with MindFile.open(a.file) as mf:
        ids_arg = None
        if a.ids:
            ids_arg = [ln.strip() for ln in open(a.ids, encoding="utf-8") if ln.strip()]
        if a.npy:
            new = _ingest.add_vectors(mf, np.load(a.npy), space=a.space, external_ids=ids_arg)
        elif a.text:
            lines = [ln.rstrip("\n") for ln in open(a.text, encoding="utf-8") if ln.strip()]
            mf.embedder = HashEmbedder(dim=a.dim)
            new = _ingest.add_items(mf, lines, space=a.space, external_ids=ids_arg or lines)
        else:
            raise SystemExit("ingest needs --npy or --text")
        print(f"ingested {len(new)} items into space '{a.space}' (ids {new[0]}..{new[-1]})")


def cmd_cluster(a) -> None:
    with MindFile.open(a.file) as mf:
        cid = _cluster.run_clustering(mf, a.space, k=a.k, collection=a.collection, note=a.note)
        print(f"clustering in space '{a.space}' -> id {cid}")
        for c in _cluster.clusters_of(mf, cid):
            print(f"  [{c['local_index']}] {c['label']}  size={c['size']}")


def cmd_evolve(a) -> None:
    with MindFile.open(a.file) as mf:
        res = _evolve.recluster(mf, a.space, k=a.k, collection=a.collection, note=a.note)
        print(f"reclustered space '{a.space}' -> id {res['clustering_id']} (parent {res['parent_id']})")
        for s in res["splits"]:
            into = ", ".join(f"{i['label']}({i['count']})" for i in s["into"])
            print(f"  SPLIT  {s['parent_label']}  ->  {into}")
        for m in res["merges"]:
            frm = ", ".join(f"{i['label']}({i['count']})" for i in m["from"])
            print(f"  MERGE  {frm}  ->  {m['into_label']}")
        if not res["splits"] and not res["merges"]:
            print("  (no significant splits or merges detected)")


def cmd_graph(a) -> None:
    with MindFile.open(a.file) as mf:
        n = _graph.build_knn_edges(mf, k=a.k, space=a.space)
        print(f"wrote {n} kNN edges (k={a.k}, space='{a.space}')")


def cmd_query(a) -> None:
    with MindFile.open(a.file) as mf:
        if a.item is not None:
            ids, mat = load_all_vectors(mf, a.space)
            pos = int(np.where(ids == a.item)[0][0])
            res = _query.search(mf, mat[pos], k=a.k, space=a.space)
        elif a.text is not None:
            _attach_reconstructable_embedder(mf, a.space)
            res = _query.search_text(mf, a.text, k=a.k, space=a.space)
        elif a.npy is not None:
            res = _query.search(mf, np.load(a.npy), k=a.k, space=a.space)
        else:
            raise SystemExit("query needs --item, --text or --npy")
        _pj(res)


def cmd_explain(a) -> None:
    with MindFile.open(a.file) as mf:
        _pj(_query.explain(mf, a.item, k=a.k, space=a.space))


def cmd_checkpoint(a) -> None:
    with MindFile.open(a.file) as mf:
        info = _cp.create_checkpoint(mf, a.name, note=a.note)
        print(f"checkpoint '{info['name']}' saved: {info['item_count']} items, "
              f"{info['vector_count']} vectors, root {info['manifest_sha256'][:16]}...")


def cmd_verify(a) -> None:
    with MindFile.open(a.file) as mf:
        rep = _cp.verify_checkpoint(mf, a.name) if a.name else _cp.verify_integrity(mf)
        _pj(rep)
        sys.exit(0 if rep["status"] == "ok" else 1)


def cmd_fuse(a) -> None:
    with MindFile.open(a.file) as mf:
        spaces = [s.strip() for s in a.spaces.split(",")]
        key = tuple(s.strip() for s in a.key.split(","))
        res = _fuse.fuse(mf, spaces, key=key, out_space=a.out)
        print(f"fused {spaces} -> space '{a.out}': created {len(res['created'])} "
              f"derived item(s), skipped {len(res['skipped'])} incomplete group(s)")


def cmd_derivations(a) -> None:
    with MindFile.open(a.file) as mf:
        rep = _fuse.verify_derivations(mf)
        _pj({"verify": rep, "derivations": _fuse.list_derivations(mf)})
        sys.exit(0 if rep["status"] == "ok" else 1)


def cmd_allocate(a) -> None:
    with MindFile.open(a.file) as mf:
        budget = {"max_new_vectors": a.max_new_vectors} if a.max_new_vectors else None
        res = _allocate.orchestrate(mf, budget=budget)
        print(f"plan #{res['plan_id']}: proposed {res['proposed']}, "
              f"approved {res['approved']}, rendered {res['rendered']}")
        for act in res["actions"]:
            mark = {"done": "OK", "skipped": "--", "failed": "!!"}.get(act["render_status"], "?")
            extra = act.get("render_detail") or {}
            print(f"  [{mark}] {act['action']:<13} {act['review_status']:<8} "
                  f"{act['review_reason']:<28} {extra if act['render_status']=='done' else ''}")


def cmd_materialize(a) -> None:
    with MindFile.open(a.file) as mf:
        n = _lazy.materialize_space(mf, a.space)
        print(f"materialized {n} virtual vector(s) in space '{a.space}'")


def cmd_summary(a) -> None:
    with MindFile.open(a.file) as mf:
        if a.json:
            _pj(_summary.summary(mf, with_centroids=a.centroids))
        else:
            print(_summary.summary_text(mf))


def cmd_findings(a) -> None:
    with MindFile.open(a.file) as mf:
        if a.discover:
            _findings.discover(mf)
        found = _findings.list_findings(mf)
        for f in found:
            print("-", f["statement"])
        if not found:
            print("(no findings recorded; run `mind findings <file> --discover` to compute them)")


def cmd_retrieve(a) -> None:
    with MindFile.open(a.file) as mf:
        print(_retrieve.context(mf, a.query, space=a.space, k=a.k))


def cmd_modalities(a) -> None:
    from . import modality as _modality
    rows = _modality.catalog()
    print(f"{len(rows)} supported file types:")
    for r in rows:
        exts = ", ".join(r["extensions"])
        deps = f"  [needs: {', '.join(r['deps'])}]" if r["deps"] else ""
        print(f"  {r['modality']:9} {exts:24} -> spaces {', '.join(r['spaces'])}{deps}")
        print(f"  {'':9} {r['storage']}")


def cmd_card(a) -> None:
    with MindFile.open(a.file) as mf:
        print(_render.card(mf, a.item))


def cmd_digest(a) -> None:
    with MindFile.open(a.file) as mf:
        print(_render.digest(mf, subject_id=a.subject))


def cmd_compact(a) -> None:
    with MindFile.open(a.file) as mf:
        n = _lazy.compact_space(mf, a.space)
        print(f"compacted {n} derived vector(s) in space '{a.space}' (lossless; reconstructs on call)")


def cmd_validate(a) -> None:
    with MindFile.open(a.file) as mf:
        rep = _validate.validate(mf)
        _pj(rep)
        sys.exit(1 if rep["errors"] else 0)


def cmd_sign(a) -> None:
    import os
    if os.path.exists(a.keyfile):
        signer = _sign.Ed25519Signer(open(a.keyfile, "rb").read())
        gen = ""
    else:
        signer = _sign.Ed25519Signer()
        with open(a.keyfile, "wb") as f:
            f.write(signer.private_bytes())
        gen = f" (generated private key -> {a.keyfile}; keep it secret)"
    with MindFile.open(a.file) as mf:
        info = _sign.sign_state(mf, signer, note=a.note)
    print(f"signed state {info['state_root'][:16]}... with key {info['key_id']}{gen}")


def cmd_signatures(a) -> None:
    with MindFile.open(a.file) as mf:
        rep = _sign.verify_signatures(mf)
    _pj(rep)
    sys.exit(0 if rep["status"] == "ok" else 1)


def cmd_link(a) -> None:
    with MindFile.open(a.file) as mf:
        info = _registry.link_file(mf, a.target, role=a.role, require_signed=a.require_signed,
                                   note=a.note)
    print(f"linked {info['path']} (uuid {(info['uuid'] or '?')[:8]}, "
          f"state {info['state_root'][:16]}..., key {info['key_id']})")


def cmd_refs(a) -> None:
    with MindFile.open(a.file) as mf:
        rep = _registry.verify_refs(mf)
    _pj(rep)
    sys.exit(0 if rep["status"] == "ok" else 1)


def cmd_recon_store(a) -> None:
    from . import modality as _modality
    with MindFile.open(a.file) as mf:
        def show(n, total, rel, raw, code):
            if a.verbose:
                print(f"  [{n}/{total}] {rel} {raw/1e6:.2f}MB -> {code/1e6:.2f}MB")
        item = _modality.get("reconall").ingest(
            mf, a.subject_dir, profile=a.profile, effort=a.effort,
            subject_id=a.subject_id, progress=show)
        rep = _reconall.manifest(mf, item)
    print(f"item {item}: {rep['subject']} ({rep['profile']}) "
          f"{rep['n_stored']}/{rep['n_files']} files, "
          f"{rep['total_bytes']/1e6:.1f}MB -> {rep['stored_bytes']/1e6:.1f}MB "
          f"= {rep['total_bytes']/max(rep['stored_bytes'],1):.2f}x")
    if rep["omitted"]:
        print(f"  omitted {len(rep['omitted'])} files (listed with hashes in the manifest)")


def cmd_recon_restore(a) -> None:
    with MindFile.open(a.file) as mf:
        rep = _reconall.restore_subject(mf, a.item, a.dest)
    print(f"{rep['status']}: restored {rep['restored']} files to {rep['dest']}")
    if rep["failed"]:
        print("  FAILED:", ", ".join(rep["failed"]))
    sys.exit(0 if rep["status"] == "ok" else 1)


def cmd_recon_files(a) -> None:
    with MindFile.open(a.file) as mf:
        rows = _reconall.list_files(mf, a.item)
    for r in rows:
        print(f"  {r['size']/1e6:8.2f}MB -> {r['code_bytes']/1e6:7.3f}MB  "
              f"{r['codec']:22s} {r['relpath']}")
    print(f"{len(rows)} files")


def cmd_recon_verify(a) -> None:
    with MindFile.open(a.file) as mf:
        rep = _reconall.verify_files(mf, deep=a.deep)
        tree = _reconall.verify_tree(mf, a.item) if a.item else None
    _pj({"files": rep, "tree": tree})
    ok = rep["status"] == "ok" and (tree is None or tree["status"] == "ok")
    sys.exit(0 if ok else 1)


def cmd_audit(a) -> None:
    with MindFile.open(a.file) as mf:
        rep = _audit.audit(mf, deep=a.deep)
        _pj(rep)
        sys.exit(0 if rep["status"] == "ok" else 1)


def cmd_history(a) -> None:
    with MindFile.open(a.file) as mf:
        for h in mf.history():
            print(f"{h['ts']}  {h['event']:<11} {json.dumps(h['detail'])}")


def cmd_inspect(a) -> None:
    with MindFile.open(a.file) as mf:
        print(f"# {mf.path.name}")
        for k, v in mf.meta_all().items():
            print(f"  {k}: {v}")
        print("\n## counts")
        for k, v in mf.counts().items():
            print(f"  {k}: {v}")
        print("\n## spaces")
        defs = _lazy.list_defs(mf)
        virtual = {}
        for d in defs:
            if not d["materialized"]:
                virtual[d["space"]] = virtual.get(d["space"], 0) + 1
        for s in _spaces.list_spaces(mf):
            tail = f"  (+{virtual[s['name']]} virtual)" if s["name"] in virtual else ""
            print(f"  {s['name']}: dim={s['dim']} embedder={s['embedder_id']} "
                  f"vectors={s['n_vectors']}{tail}")
        print("\n## collections")
        for c in _collections.list_collections(mf):
            print(f"  {c['name']} ({c['kind']}): {c['size']} items")
        derivs = _fuse.list_derivations(mf)
        if derivs:
            print("\n## derivations (fused)")
            for d in derivs:
                print(f"  {d['external_id']} via {d['method']} key={d['fusion_key']}")
        print("\n## clustering versions")
        for c in _cluster.clustering_versions(mf):
            print(f"  [{c['space']}] v{c['version']} (id {c['id']}, parent {c['parent_id']}) "
                  f"{c['note'] or ''}")
        print("\n## save points")
        for cp in _cp.list_checkpoints(mf):
            print(f"  {cp['name']}: {cp['item_count']} items / {cp['vector_count']} vectors, "
                  f"root {cp['manifest_sha256'][:16]}... {cp['note'] or ''}")
        corrs = _stats.list_correlations(mf)
        if corrs:
            print("\n## correlations")
            for c in corrs:
                v = f"{c['value']:.3f}" if c["value"] is not None else "n/a"
                print(f"  {c['kind']}: {c['a_ref']} vs {c['b_ref']}  {c['method']}={v} (n={c['n']})")
        print("\n## evolution (splits/merges)")
        ev = _query.evolution_report(mf)
        if not ev:
            print("  (none)")
        for h in ev:
            d = h["detail"]
            if h["event"] == "split":
                into = ", ".join(f"{i['label']}({i['count']})" for i in d["into"])
                print(f"  SPLIT {d['parent_label']} -> {into}")
            else:
                frm = ", ".join(f"{i['label']}({i['count']})" for i in d["from"])
                print(f"  MERGE {frm} -> {d['into_label']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mind", description="Work with .gnos files.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("create"); sp.add_argument("file")
    sp.add_argument("--modality", default="generic"); sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_create)

    sp = sub.add_parser("ingest"); sp.add_argument("file")
    sp.add_argument("--npy"); sp.add_argument("--text"); sp.add_argument("--ids")
    sp.add_argument("--dim", type=int, default=16); sp.add_argument("--space", default="default")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("cluster"); sp.add_argument("file")
    sp.add_argument("--k", type=int); sp.add_argument("--space", default="default")
    sp.add_argument("--collection"); sp.add_argument("--note"); sp.set_defaults(func=cmd_cluster)

    sp = sub.add_parser("evolve"); sp.add_argument("file")
    sp.add_argument("--k", type=int); sp.add_argument("--space", default="default")
    sp.add_argument("--collection"); sp.add_argument("--note"); sp.set_defaults(func=cmd_evolve)

    sp = sub.add_parser("graph"); sp.add_argument("file")
    sp.add_argument("--k", type=int, default=5); sp.add_argument("--space", default="default")
    sp.set_defaults(func=cmd_graph)

    sp = sub.add_parser("query"); sp.add_argument("file")
    sp.add_argument("--item", type=int); sp.add_argument("--text"); sp.add_argument("--npy")
    sp.add_argument("--k", type=int, default=5); sp.add_argument("--space", default="default")
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("explain"); sp.add_argument("file")
    sp.add_argument("--item", type=int, required=True); sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--space", default="default"); sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("checkpoint"); sp.add_argument("file"); sp.add_argument("name")
    sp.add_argument("--note"); sp.set_defaults(func=cmd_checkpoint)

    sp = sub.add_parser("verify"); sp.add_argument("file"); sp.add_argument("name", nargs="?")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("fuse"); sp.add_argument("file")
    sp.add_argument("--spaces", required=True, help="comma-separated source spaces, e.g. bandpower,mri")
    sp.add_argument("--key", default="subject_id,session_id")
    sp.add_argument("--out", default="fused"); sp.set_defaults(func=cmd_fuse)

    sp = sub.add_parser("derivations"); sp.add_argument("file"); sp.set_defaults(func=cmd_derivations)

    sp = sub.add_parser("allocate"); sp.add_argument("file")
    sp.add_argument("--max-new-vectors", type=int, dest="max_new_vectors")
    sp.set_defaults(func=cmd_allocate)

    sp = sub.add_parser("materialize"); sp.add_argument("file")
    sp.add_argument("--space", required=True); sp.set_defaults(func=cmd_materialize)

    sp = sub.add_parser("compact"); sp.add_argument("file")
    sp.add_argument("--space", required=True); sp.set_defaults(func=cmd_compact)

    sp = sub.add_parser("summary"); sp.add_argument("file")
    sp.add_argument("--json", action="store_true"); sp.add_argument("--centroids", action="store_true")
    sp.set_defaults(func=cmd_summary)

    sp = sub.add_parser("card"); sp.add_argument("file"); sp.add_argument("--item", type=int, required=True)
    sp.set_defaults(func=cmd_card)

    sp = sub.add_parser("digest"); sp.add_argument("file"); sp.add_argument("--subject")
    sp.set_defaults(func=cmd_digest)

    sp = sub.add_parser("findings"); sp.add_argument("file")
    sp.add_argument("--discover", action="store_true"); sp.set_defaults(func=cmd_findings)

    sp = sub.add_parser("retrieve"); sp.add_argument("file")
    sp.add_argument("--query", required=True); sp.add_argument("--space")
    sp.add_argument("--k", type=int, default=5); sp.set_defaults(func=cmd_retrieve)

    sp = sub.add_parser("validate"); sp.add_argument("file"); sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("recon-store", help="fold a FreeSurfer recon-all subject tree in")
    sp.add_argument("file"); sp.add_argument("subject_dir")
    sp.add_argument("--profile", default="lossless", choices=sorted(_reconall.PROFILES))
    sp.add_argument("--effort", default="balanced", choices=("fast", "balanced", "max"))
    sp.add_argument("--subject-id", dest="subject_id")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_recon_store)

    sp = sub.add_parser("recon-restore", help="write a stored subject tree back out")
    sp.add_argument("file"); sp.add_argument("dest")
    sp.add_argument("--item", type=int, required=True); sp.set_defaults(func=cmd_recon_restore)

    sp = sub.add_parser("recon-files", help="list the stored files of a subject")
    sp.add_argument("file"); sp.add_argument("--item", type=int, required=True)
    sp.set_defaults(func=cmd_recon_files)

    sp = sub.add_parser("recon-verify", help="re-check stored files (and the file set)")
    sp.add_argument("file"); sp.add_argument("--item", type=int)
    sp.add_argument("--deep", action="store_true"); sp.set_defaults(func=cmd_recon_verify)

    sp = sub.add_parser("audit"); sp.add_argument("file")
    sp.add_argument("--deep", action="store_true"); sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("sign"); sp.add_argument("file")
    sp.add_argument("--keyfile", required=True); sp.add_argument("--note")
    sp.set_defaults(func=cmd_sign)

    sp = sub.add_parser("signatures"); sp.add_argument("file"); sp.set_defaults(func=cmd_signatures)

    sp = sub.add_parser("link"); sp.add_argument("file"); sp.add_argument("target")
    sp.add_argument("--role"); sp.add_argument("--note")
    sp.add_argument("--require-signed", action="store_true", dest="require_signed")
    sp.set_defaults(func=cmd_link)

    sp = sub.add_parser("refs"); sp.add_argument("file"); sp.set_defaults(func=cmd_refs)
    sp = sub.add_parser("history"); sp.add_argument("file"); sp.set_defaults(func=cmd_history)
    sp = sub.add_parser("inspect"); sp.add_argument("file"); sp.set_defaults(func=cmd_inspect)
    sp = sub.add_parser("modalities", help="list every supported file type")
    sp.set_defaults(func=cmd_modalities)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
