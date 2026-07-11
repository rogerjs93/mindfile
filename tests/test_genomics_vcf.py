from mindfile.modality import GenomicsProcessor

VCF = """##fileformat=VCFv4.0
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
19\t111\t.\tA\tC\t9.6\tPASS\t.
19\t112\t.\tA\tG\t10\tPASS\t.
20\t14370\trs6054257\tG\tA\t29\tPASS\t.
"""


def test_variants_parses_vcf_text_by_line_not_char():
    g = GenomicsProcessor()
    v = g._variants(VCF)                          # raw VCF *text* (regression: was char-by-char)
    assert v == ["19:111:A:C", "19:112:A:G", "20:14370:G:A"]


def test_variants_accepts_list_and_single_token():
    g = GenomicsProcessor()
    assert g._variants(["chr1:100:A>G", "chr2:200:C>T"]) == ["chr1:100:A>G", "chr2:200:C>T"]
    assert g._variants("chr7:5:A>T") == ["chr7:5:A>T"]     # single token, not 11 chars


def test_variants_reads_vcf_path(tmp_path):
    p = tmp_path / "s.vcf"
    p.write_text(VCF, encoding="utf-8")
    assert GenomicsProcessor()._variants(str(p)) == ["19:111:A:C", "19:112:A:G", "20:14370:G:A"]
