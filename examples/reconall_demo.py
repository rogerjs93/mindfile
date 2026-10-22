"""Fold a REAL FreeSurfer recon-all subject into one .gnos, restore it, break it.

Run against any recon-all subject directory::

    python examples/reconall_demo.py <subject_dir> [out_dir]

What it proves, in order:

1. the whole tree stores losslessly and how much smaller it gets, per directory;
2. the restored tree is byte-for-byte identical to the original (compared
   independently of the library, by re-hashing the files on disk);
3. the bundle seals: checkpoint + Ed25519 signature + deep audit;
4. a single flipped bit anywhere in the stored data is caught and localised;
5. the `analysis` profile, for when the regenerable intermediates aren't wanted.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from mindfile import MindFile, audit, checkpoint, modality, reconall, render, sign


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def main(subject_dir: str, out_dir: str | None = None) -> int:
    subject = Path(subject_dir)
    if not subject.is_dir():
        print(f"not a directory: {subject}")
        return 2
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="reconall_demo_"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "subject.gnos"

    print(f"== 1. store {subject} ==")
    n_files = sum(1 for p in subject.rglob("*") if p.is_file())
    raw = sum(p.stat().st_size for p in subject.rglob("*") if p.is_file())
    print(f"   {n_files} files, {raw/1e6:.1f}MB")

    mf = MindFile.create(path, modality="reconall", overwrite=True)
    per_dir: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    t0 = time.time()

    def progress(n, total, rel, orig, code):
        top = rel.split("/")[0] if "/" in rel else "."
        per_dir[top][0] += orig
        per_dir[top][1] += code
        if n % 10 == 0 or orig > 5e6:
            print(f"   [{n:3d}/{total}] {rel:44s} {orig/1e6:7.2f} -> {code/1e6:6.2f}MB")

    item = modality.ingest(mf, "reconall", subject, subject_id=subject.name,
                           progress=progress)
    took = time.time() - t0
    man = reconall.manifest(mf, item)
    print(f"\n   stored in {took/60:.1f} min")
    print(f"   {'directory':12s} {'original':>12s} {'stored':>12s}  ratio")
    for d, (o, c) in sorted(per_dir.items(), key=lambda kv: -kv[1][0]):
        print(f"   {d:12s} {o/1e6:10.1f}MB {c/1e6:10.1f}MB  {o/max(c,1):5.2f}x")
    disk = path.stat().st_size
    print(f"   {'TOTAL':12s} {raw/1e6:10.1f}MB {disk/1e6:10.1f}MB  {raw/disk:5.2f}x"
          f"   (.gnos file on disk, including its index)")
    print("   codecs used:", man["codecs"])
    print(f"   solid blocks: {man.get('blocks', 0)} "
          f"(saved {man.get('block_saved_bytes', 0)/1e6:.1f}MB)")
    print("\n   card:", render.card(mf, item))

    print("\n== 2. restore and compare byte-for-byte ==")
    dest = out / "restored"
    if dest.exists():
        shutil.rmtree(dest)
    t0 = time.time()
    rep = reconall.restore_subject(mf, item, dest)
    print(f"   restored {rep['restored']} files in {time.time()-t0:.0f}s -> {rep['status']}")
    before, after = tree_hashes(subject), tree_hashes(dest)
    same = before == after
    print(f"   independent re-hash of every file: {'IDENTICAL' if same else 'MISMATCH'}")
    if not same:
        for k in sorted(set(before) | set(after)):
            if before.get(k) != after.get(k):
                print(f"     differs: {k}")

    print("\n== 3. seal the bundle ==")
    checkpoint.create_checkpoint(mf, "recon-sealed")
    signer = sign.Ed25519Signer()
    sign.sign_state(mf, signer)
    sign.trust_key(mf, signer.public_key)
    t0 = time.time()
    rep = audit.audit(mf, deep=True)
    print(f"   audit(deep) = {rep['status']} corrupted={rep['corrupted']} "
          f"({time.time()-t0:.0f}s)")
    print(f"   signature trusted={sign.verify_signatures(mf)['trusted_attested']}")
    mf.close()

    print("\n== 4. flip ONE bit deep inside the stored data ==")
    broken = out / "tampered.gnos"
    shutil.copy2(path, broken)
    bad = MindFile.open(broken)
    row = bad.conn.execute(
        "SELECT id, relpath, code FROM files ORDER BY code_bytes DESC LIMIT 1").fetchone()
    blob = bytearray(row["code"])
    blob[len(blob) // 2] ^= 0x01
    bad.conn.execute("UPDATE files SET code=? WHERE id=?", (bytes(blob), row["id"]))
    bad.conn.commit()
    rep = audit.audit(bad)
    print(f"   corrupted {row['relpath']} by one bit")
    print(f"   audit = {rep['status']} corrupted={rep['corrupted']}")
    print(f"   pinpointed: {reconall.verify_files(bad)['bad']}")
    bad.close()

    print("\n== 5. analysis profile (regenerable intermediates dropped) ==")
    ap = out / "subject-analysis.gnos"
    mf2 = MindFile.create(ap, modality="reconall", overwrite=True)
    t0 = time.time()
    rep2 = reconall.store_subject(mf2, subject, subject_id=subject.name, profile="analysis")
    print(f"   kept {rep2['n_stored']}/{rep2['n_files']} files, "
          f"{rep2['total_bytes']/1e6:.1f}MB of source -> "
          f"{ap.stat().st_size/1e6:.1f}MB on disk "
          f"= {rep2['total_bytes']/ap.stat().st_size:.1f}x  ({time.time()-t0:.0f}s)")
    print(f"   {rep2['n_omitted']} omitted files remain listed with their hashes")
    print(f"   tree manifest: {reconall.verify_tree(mf2, rep2['item_id'])['status']}")
    mf2.close()

    print(f"\nartifacts in {out}")
    return 0 if same else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:3]))
