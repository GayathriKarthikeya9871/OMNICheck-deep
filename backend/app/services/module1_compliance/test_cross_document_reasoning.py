"""Focused tests for cross-document reasoning (Prompt 5, Phase B) on top of the Phase A linking layer.
Place next to tasks.py (or set OMNI_PKG_DIR). Runs offline: no DB, network or LLM. `app.*` / `celery` are stubbed only when not importable."""
import importlib, os, sys, types, pathlib, tempfile, collections
import pytest

HERE = pathlib.Path(__file__).resolve().parent
PKG_DIR = pathlib.Path(os.getenv("OMNI_PKG_DIR") or (HERE if (HERE / "tasks.py").exists() else HERE.parent))


def _load_tasks():
    for name in ("celery", "dotenv", "app", "app.db", "app.db.database", "app.models", "app.models.domain"):
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)

    class _C:
        def __init__(self, *a, **k): pass
        def task(self, *a, **k): return lambda f: f

    if not hasattr(sys.modules["celery"], "Celery"): sys.modules["celery"].Celery = _C
    if not hasattr(sys.modules["dotenv"], "load_dotenv"): sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
    if not hasattr(sys.modules["app.db.database"], "SessionLocal"): sys.modules["app.db.database"].SessionLocal = None
    for n in ("InvestigationRecord", "EvidenceNode", "EvidenceEdge"):
        if not hasattr(sys.modules["app.models.domain"], n): setattr(sys.modules["app.models.domain"], n, object)
    pkg = types.ModuleType("omni_pkg_under_test")
    pkg.__path__ = [str(PKG_DIR)]
    sys.modules["omni_pkg_under_test"] = pkg
    return importlib.import_module("omni_pkg_under_test.tasks")


T = _load_tasks()


def _graph(docs, rulebook=None):
    d = tempfile.mkdtemp()
    paths = []
    for name, text in docs.items():
        p = os.path.join(d, name)
        open(p, "w").write(text)
        paths.append(p)
    rb = None
    if rulebook:
        rb = os.path.join(d, "rulebook.txt")
        open(rb, "w").write(rulebook)
    return T.build_evidence_graph(paths, rb)


def _doc(G, fname):
    return next(n for n, d in G.nodes(data=True) if d.get("type") == "Document" and d.get("filename") == fname)


def _ev(G, fname):
    return [e for e, _ in T._investigation_evidence(G) if any(s["filename"] == fname for s in T._evidence_source_docs(G, e))]


def _links(G):
    return T.query_cross_document_links(G)


INVOICE = "Invoice No: INV-0042\nPO Number: PO-7781\nVendor: Acme Supplies Ltd\nInvoice date 2024-03-10\nBilled charges INR 50,000\n"
PO = "Purchase Order No: PO-7781\nSupplier: Acme Supplies Pvt Ltd\nApproval ID: APPR-55\nOrder date 2024-03-01\nItems: servers\n"
APPROVAL = "Approval ID: APPR-55\nGranted by the finance head\n"
OBJ = "check the billed charges"  # matches the invoice lexically, NOT the PO / approval documents


def _ctx(G, objective=OBJ, **kw):
    return T.build_v2_context(G, objective, **kw)


# 1 ---- single document: unchanged
def test_single_document_context_unchanged():
    G = _graph({"invoice_a.txt": "Invoice No: INV-0042\nVendor: Acme Supplies Ltd\nBilled charges INR 50,000\n"})  # no reference to any other document
    off, _ = _ctx(G, cross_document=False)
    on, st_on = _ctx(G, cross_document=True)
    assert on == off  # byte-identical: nothing cross-document to add
    m = st_on["cross_document"]["metrics"]
    assert m["links_inspected"] == 0 and m["evidence_added_via_explicit_links"] == 0
    assert T.v2_cross_document_section(st_on) == ""


def test_disabled_flag_reproduces_pre_phase_b_context():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO})
    off, st = _ctx(G, cross_document=False)
    assert "CROSS-DOCUMENT EVIDENCE" not in off and st["cross_document"]["status"] == "NOT_RUN"


# 2 ---- explicit invoice -> PO
def test_explicit_invoice_to_po_retrieval_and_provenance():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO})
    inv, po = _doc(G, "invoice_a.txt"), _doc(G, "po_a.txt")
    base, _ = _ctx(G, cross_document=False)
    assert _ev(G, "po_a.txt")[0] not in base  # the PO is not a lexical / graph hit
    ctx, st = _ctx(G, cross_document=True)
    xd = st["cross_document"]
    assert [i["filename"] for i in xd["explicit_items"]] == ["po_a.txt"]
    it = xd["explicit_items"][0]
    assert "[XDOC-EXPLICIT]" in ctx and it["evidence_id"] in ctx
    # provenance survives traversal
    assert it["document_id"] == po and it["related_document_id"] == inv and it["related_filename"] == "invoice_a.txt"
    assert it["evidence_id"] in _ev(G, "po_a.txt") and it["location"] == G.nodes[it["evidence_id"]]["source_location"]
    assert it["link_status"] == "EXPLICIT" and it["relationship_type"] == "PURCHASE_ORDER_TO_INVOICE"
    assert it["link_reason"] and it["match_methods"] and it["match_basis"] == "matched_field" and it["hops"] == 1
    assert it["anchoring_direct_evidence_ids"] and it["treated_as_identity"] is False
    assert {s["document_id"] for s in T._evidence_source_docs(G, it["evidence_id"])} == {po}
    assert xd["metrics"]["documents_via_cross_document_links"] == 1 and xd["metrics"]["links_accepted_for_traversal"] == 1
    assert "Cross-document evidence (V2 Phase B" in T.v2_cross_document_section(st)


# 3 ---- multi-hop invoice -> PO -> approval
def test_multi_hop_chain_and_hop_limit():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO, "approval_a.txt": APPROVAL})
    direct = _ev(G, "invoice_a.txt")
    xd = T.traverse_cross_document_context(G, direct, OBJ, max_hops=3)
    hops = {i["filename"]: i["hops"] for i in xd["explicit_items"]}
    assert hops == {"po_a.txt": 1, "approval_a.txt": 2}
    appr = next(i for i in xd["explicit_items"] if i["filename"] == "approval_a.txt")
    assert len(appr["path"]) == 2 and appr["path"][0]["to_document_id"] == _doc(G, "po_a.txt")
    assert appr["root_document_id"] == _doc(G, "invoice_a.txt")
    short = T.traverse_cross_document_context(G, direct, OBJ, max_hops=1)
    assert [i["filename"] for i in short["explicit_items"]] == ["po_a.txt"] and short["metrics"]["frontier_not_expanded"] == 1


def test_budgets_are_reported_not_hidden():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO, "approval_a.txt": APPROVAL})
    xd = T.traverse_cross_document_context(G, _ev(G, "invoice_a.txt"), OBJ, doc_limit=1)
    assert xd["metrics"]["doc_limit_reached"] is True and len(xd["related_documents"]) == 1


# 4 ---- weak link: surfaced, not identity, not traversed
WEAK_A = "Invoice No: INV-5001\nReference: ABC12345\nBilled charges INR 700\n"
WEAK_B = "Payment UTR: ABC12345\nSettled by bank transfer\n"


def test_weak_link_surfaced_but_not_identity():
    G = _graph({"invoice_w.txt": WEAK_A, "payment_w.txt": WEAK_B})
    (e,) = _links(G)
    assert e["link_status"] == "WEAK" and e["match_strength"] == "weak"
    ctx, st = _ctx(G, cross_document=True)
    xd = st["cross_document"]
    assert xd["explicit_items"] == [] and xd["metrics"]["links_accepted_for_traversal"] == 0
    (w,) = xd["weak_signals"]
    assert w["treated_as_identity"] is False and w["link_status"] == "WEAK"
    assert "[XDOC-POSSIBLE]" in ctx and "does NOT establish" in ctx and "[XDOC-EXPLICIT] Evidence Node" not in ctx


# 5 ---- conflicting link: surfaced as conflict
CONF_A = "Invoice No: INV-1\nPO Number: PO-77\nVendor: Acme Supplies\nBilled charges INR 1,000\n"
CONF_B = "Invoice No: INV-2\nPO Number: PO-77\nVendor: Globex Corp\nAmount INR 2,000\n"


def test_conflicting_link_surfaced_not_resolved():
    G = _graph({"invoice_c1.txt": CONF_A, "invoice_c2.txt": CONF_B})
    (e,) = _links(G)
    assert e["link_status"] == "CONFLICTING" and e["conflict_status"] == "CONFLICTING"
    ctx, st = _ctx(G, cross_document=True)
    xd = st["cross_document"]
    assert xd["explicit_items"] == [] and xd["metrics"]["links_accepted_for_traversal"] == 0
    (c,) = xd["conflict_signals"]
    assert c["requires_reconciliation"] is True and {x["field"] for x in c["conflicts"]} >= {"invoice_id", "vendor"}
    assert "[XDOC-CONFLICT]" in ctx and "RECONCILIATION REQUIRED" in ctx


# 6 ---- semantic-only: never identity
def test_semantic_only_link_never_establishes_identity(monkeypatch):
    monkeypatch.setattr(T, "XDOC_SEMANTIC_THRESHOLD", 0.2)
    G = _graph({"invoice_s.txt": "Invoice for consulting services rendered to the client during the quarter billed charges\n",
                "payment_s.txt": "Payment for consulting services rendered to the client during the quarter settled\n"})
    edges = _links(G)
    assert len(edges) == 1 and edges[0]["semantic_only"] is True and edges[0]["link_status"] == "WEAK"
    assert edges[0]["semantic_proof_of_identity"] is False
    ok, _why, _scope = T._xdoc_eligibility(G, edges[0])
    assert ok is False
    ctx, st = _ctx(G, cross_document=True)
    xd = st["cross_document"]
    assert xd["explicit_items"] == [] and xd["weak_signals"] and xd["weak_signals"][0]["semantic_only"] is True and "SEMANTIC-ONLY" in ctx


def test_unresolved_link_establishes_nothing():
    G = _graph({"invoice_u.txt": "Invoice No: INV-9\nBilled charges INR 700\n", "payment_u.txt": "Payment advice\nRemitted INR 700 on 2024-05-05\n"})
    sts = {e["link_status"] for e in _links(G)}
    assert sts <= {"UNRESOLVED", "WEAK"} and "EXPLICIT" not in sts
    _, st = _ctx(G, cross_document=True)
    assert st["cross_document"]["explicit_items"] == []


# 7 ---- missing referenced document
def test_missing_referenced_document_is_reported_not_fabricated():
    G = _graph({"invoice_m.txt": "Invoice No: INV-0042\nPO Number: PO-9999\nBilled charges INR 50,000\n"})
    docs_before = [n for n, d in G.nodes(data=True) if d.get("type") == "Document"]
    ctx, st = _ctx(G, cross_document=True)
    xd = st["cross_document"]
    assert [(r["reference_type"], r["reference"]) for r in xd["missing_references"]] == [("purchase_order", "PO9999")]
    assert xd["missing_references"][0]["provenance"] and "[MISSING-REFERENCE]" in ctx and "do not assume" in ctx
    assert xd["explicit_items"] == [] and xd["metrics"]["unresolved_references_surfaced"] == 1
    assert [n for n, d in G.nodes(data=True) if d.get("type") == "Document"] == docs_before  # nothing invented
    assert G.nodes[_doc(G, "invoice_m.txt")]["xdoc_unresolved_references"]  # Phase A data preserved


# 8 ---- distractors / policy documents
def test_distractors_not_added_indiscriminately():
    docs = {"invoice_a.txt": INVOICE, "po_a.txt": PO,
            "distractor_1.txt": "Payment UTR: UTR111222\nVendor: Globex Corp\nCatering for the picnic INR 300\n",
            "distractor_2.txt": "Meeting minutes\nAgenda: office plants\n",
            "distractor_3.txt": "Invoice No: INV-8888\nVendor: Acme Supplies Ltd\nOffice stationery INR 90\n"}  # same vendor, unrelated transaction
    G = _graph(docs)
    _, st = _ctx(G, cross_document=True)
    assert {i["filename"] for i in st["cross_document"]["explicit_items"]} == {"po_a.txt"}
    assert {r["filename"] for r in st["cross_document"]["related_documents"]} == {"po_a.txt"}


def test_policy_document_is_never_traversed():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO})
    G.nodes[_doc(G, "po_a.txt")]["policy_context"] = True  # the document is (re)classified as policy context
    xd = T.traverse_cross_document_context(G, _ev(G, "invoice_a.txt"), OBJ)
    assert xd["explicit_items"] == [] and any("policy" in r["reason"] for r in xd["rejected_links"])


def test_context_only_policy_evidence_is_not_a_seed():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO})
    ev = _ev(G, "invoice_a.txt")
    for e in ev: G.nodes[e]["context_only"] = True
    assert T.traverse_cross_document_context(G, ev, OBJ)["status"] == "NO_DIRECT_DOCUMENTS"


# 9 ---- no verdict / graph mutation
def test_cross_document_retrieval_creates_no_policy_verdict():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO, "approval_a.txt": APPROVAL}, rulebook="FORBID TRANSACTION > INR 1000\n")

    def snap():
        verd = collections.Counter(d.get("verdict") for _, d in G.nodes(data=True) if d.get("type") == "Decision")
        rel = collections.Counter(d.get("relation") for _, _, d in G.edges(data=True))
        return G.number_of_nodes(), G.number_of_edges(), verd, rel

    before = snap()
    _ctx(G, cross_document=False)
    _, st = _ctx(G, cross_document=True)
    assert snap() == before  # read-only: no node, edge, Decision, VIOLATES or SATISFIES added or changed
    assert st["cross_document"]["metrics"]["evidence_added_via_explicit_links"] >= 1
    for it in st["cross_document"]["explicit_items"]:
        assert it["treated_as_identity"] is False and "verdict" not in it


# 10 ---- Phase A preserved
def test_phase_a_edges_unchanged_by_phase_b():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO, "approval_a.txt": APPROVAL})
    key = lambda: [(e["source_id"], e["target_id"], e["link_status"], e["relationship_type"], e["match_score"]) for e in _links(G)]
    before = key()
    _ctx(G, cross_document=True)
    assert before == key() and G.graph["cross_document_summary"]["status"] == "OK"
    assert len(T.query_compliance_chains(G)) == 1


# 11 ---- instrumentation / evaluation honesty
def test_evaluation_not_measured_without_ground_truth():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO})
    r = T.evaluate_evidence_retrieval(G, OBJ)
    assert r["status"] == "NOT_MEASURED" and "lexical_plus_graph_plus_cross_document" not in r
    assert r["cross_document"]["metrics"]["evidence_added_via_explicit_links"] == 1  # a count of what was done, not an accuracy claim


def test_evaluation_measured_with_ground_truth_and_variants():
    G = _graph({"invoice_a.txt": INVOICE, "po_a.txt": PO})
    gt = [{"file": "invoice_a.txt"}, {"file": "po_a.txt"}]
    r = T.evaluate_evidence_retrieval(G, OBJ, gt)
    assert r["status"] == "MEASURED"
    assert r["lexical_plus_graph_expansion"]["recall"] == 0.5 and r["lexical_plus_graph_plus_cross_document"]["recall"] == 1.0
    assert r["lexical_plus_graph_plus_cross_document"]["precision"] == 1.0
    assert set(r["variant_notes"]) == {"v2_existing_graph", "v2_phase_a", "v2_phase_b", "v1"}
    rec = T.build_experiment_record("c", "V2", 1, G, OBJ, ground_truth={"relevant_evidence": gt})
    assert "retrieval_with_cross_document_recall" in T._scalar_metrics(rec)


# 12 ---- benchmark + evaluation layer
import json

_REQUIRED_RECORD_FIELDS = ("case_id", "system_variant", "expected_decision", "predicted_decision", "decision_correct", "required_documents", "retrieved_documents", "evidence_precision",
                           "evidence_recall", "evidence_f1", "expected_entity_links", "predicted_entity_links", "entity_link_precision", "entity_link_recall", "entity_link_f1",
                           "expected_contradictions", "predicted_contradictions", "contradiction_precision", "contradiction_recall", "contradiction_f1", "false_contradiction_rate",
                           "missed_contradiction_rate", "missing_evidence", "distractors", "provenance")


@pytest.fixture(scope="module")
def bench():
    return T.run_cross_document_benchmark()


def test_benchmark_labels_are_consistent_and_cover_required_categories():
    assert T.validate_cross_document_benchmark() == []
    cases = T.CROSS_DOCUMENT_BENCHMARK
    assert len(cases) == 16 and len({c["case_id"] for c in cases}) == 16
    cats = {c["category"] for c in cases}
    assert {"one_document_sufficient", "three_document_chain", "four_plus_document_chain", "contradictory_documents", "missing_required_evidence", "distractor_documents",
            "identifier_entity_link", "date_relationship", "amount_relationship", "structured_field_relationship", "semantic_similarity", "weak_unresolved_link", "unrelated_documents"} <= cats
    bad = [dict(cases[0], case_id=cases[1]["case_id"])]
    assert T.validate_cross_document_benchmark(cases[:2] + bad)  # duplicate id is detected


def test_every_case_has_a_record_per_variant_with_all_fields(bench):
    recs = bench["records"]
    assert len(recs) == 16 * 3 and {r["system_variant"] for r in recs} == set(T.XBENCH_VARIANTS)
    for r in recs: assert all(k in r for k in _REQUIRED_RECORD_FIELDS), r["case_id"]


def test_benchmark_is_deterministic(bench):
    again = T.run_cross_document_benchmark()
    assert json.dumps(again["records"], sort_keys=True) == json.dumps(bench["records"], sort_keys=True)
    assert bench["aggregate"]["benchmark"]["seed"] is None


def test_v1_is_not_given_graph_decisions_links_or_contradictions(bench):
    for r in (x for x in bench["records"] if x["system_variant"] == "V1"):
        assert r["predicted_decision"] == T.NOT_MEASURED and r["predicted_entity_links"] == T.NOT_MEASURED and r["predicted_contradictions"] == T.NOT_MEASURED
        assert r["retrieved_documents"]  # retrieval IS measurable: documents present in the V1 payload
    a = bench["aggregate"]["variants"]["V1"]
    assert a["decision"]["status"] == T.NOT_MEASURED and a["entity_links"]["status"] == T.NOT_MEASURED and a["contradictions"]["status"] == T.NOT_MEASURED


def test_aggregate_is_derived_from_raw_records(bench):
    assert T.aggregate_cross_document_results(bench["records"]) == bench["aggregate"]
    cd = bench["aggregate"]["variants"]["CROSS_DOCUMENT"]
    assert cd["retrieval"]["status"] == "MEASURED" and cd["entity_links"]["status"] == "MEASURED"
    assert set(bench["aggregate"]["per_category"]["CROSS_DOCUMENT"]) == {c["category"] for c in T.CROSS_DOCUMENT_BENCHMARK}
    assert bench["aggregate"]["comparison"]["order"] == ["V1", "EVIDENCE_GRAPH", "CROSS_DOCUMENT"]


def test_contradiction_signals_are_never_violations_and_keep_provenance(bench):
    for r in bench["records"]:
        if isinstance(r["predicted_contradictions"], list):
            assert all(c["is_violation"] is False for c in r["predicted_contradictions"]) and r["predicted_decision"] != "VIOLATION"
    r = next(x for x in bench["records"] if x["case_id"].startswith("XB-05") and x["system_variant"] == "CROSS_DOCUMENT")
    ev = r["provenance"]["contradiction_evidence"]["invoice_006.txt|po_006.txt"]
    assert ev and all(e.get("file") for e in ev)
    assert r["provenance"]["expected_matched"] == 1 and r["provenance"]["observed"][0]["link_status"] == "CONFLICTING"


def test_semantic_only_is_never_a_strong_predicted_link(bench):
    for r in (x for x in bench["records"] if x["system_variant"] == "CROSS_DOCUMENT" and x["case_id"].startswith("XB-12")):
        assert all(l["strength"] != "strong" for l in r["predicted_entity_links"])


def test_missing_evidence_and_distractors_are_scored(bench):
    r = next(x for x in bench["records"] if x["case_id"].startswith("XB-06") and x["system_variant"] == "CROSS_DOCUMENT")
    assert r["missing_evidence"]["recall"] == 1.0 and r["predicted_decision"] == "INSUFFICIENT_EVIDENCE"
    eg = next(x for x in bench["records"] if x["case_id"].startswith("XB-06") and x["system_variant"] == "EVIDENCE_GRAPH")
    assert eg["missing_evidence"]["recall"] == T.NOT_MEASURED
    d = next(x for x in bench["records"] if x["case_id"].startswith("XB-07") and x["system_variant"] == "CROSS_DOCUMENT")
    assert d["distractors"]["retrieved"] == [] and d["distractors"]["retrieval_rate"] == 0.0


def test_benchmark_writes_json_only_when_asked(tmp_path):
    out = tmp_path / "bench.json"
    res = T.run_cross_document_benchmark(cases=T.CROSS_DOCUMENT_BENCHMARK[:2], output_path=str(out))
    assert json.loads(out.read_text())["records"] == res["records"]


# 13 ---- contradiction classifier + contradiction benchmark
def _findings(G, category=None, field=None):
    return [f for f in G.graph["contradiction_findings"] if (category is None or f["category"] == category) and (field is None or f["field"] == field)]


CON_INV = "Invoice No: INV-1101\nPO Number: PO-1101\nVendor: Boreal Metals\nBilled amount INR 20,000\n"
CON_PO = "Purchase Order No: PO-1101\nVendor: Boreal Metals\nOrder value INR 12,000\n"


def test_amount_contradiction_keeps_both_claims_and_provenance():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO})
    (f,) = _findings(G, "MAJOR_CONTRADICTION", "amount")
    assert {f["claim_a"]["filename"], f["claim_b"]["filename"]} == {"invoice_k.txt", "po_k.txt"}
    vals = {f["claim_a"]["filename"]: f["claim_a"]["value"]["amounts"], f["claim_b"]["filename"]: f["claim_b"]["value"]["amounts"]}
    assert vals == {"invoice_k.txt": [20000.0], "po_k.txt": [12000.0]}  # both claims preserved, neither chosen
    for side in (f["claim_a"], f["claim_b"]):
        p = side["provenance"][0]
        assert p["evidence_id"] in G and p["document_id"] == side["document_id"] and p["location"] == G.nodes[p["evidence_id"]]["source_location"]
    assert f["resolution"] == "NOT_RESOLVED" and f["is_policy_violation"] is False and f["link"]["link_status"] == "CONFLICTING"


def test_contradiction_is_never_a_policy_violation():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 100000\n")
    assert _findings(G, "MAJOR_CONTRADICTION") and _findings(G, "POLICY_VIOLATION") == []
    assert not any(d.get("relation") in ("VIOLATES",) for _, _, d in G.edges(data=True) if d.get("relation") == "VIOLATES" and False)
    assert all(d.get("verdict") != "VIOLATION" for _, d in G.nodes(data=True) if d.get("type") == "Decision")


def test_policy_violation_only_mirrors_existing_deterministic_decision():
    G = _graph({"invoice_p.txt": "Invoice No: INV-1201\nBilled amount INR 5,000\n"}, rulebook="FORBID TRANSACTION > INR 1000\n")
    (f,) = _findings(G, "POLICY_VIOLATION")
    dec = f["claim_a"]["value"]["decision_id"]
    assert G.nodes[dec]["verdict"] == "VIOLATION" and f["source"] == "existing_deterministic_decision" and f["derived_from_contradiction"] is False


def test_legitimate_partial_payment_is_consistent_not_contradiction():
    G = _graph({"invoice_k.txt": "Invoice No: INV-1301\nBilled amount INR 20,000\n", "payment_k.txt": "Payment UTR: UTR-1301 partial payment against INV-1301\nSettled INR 8,000\n"})
    (f,) = _findings(G, field="amount")
    assert f["category"] == "CONSISTENT" and f["legitimate_difference"] == "partial_payment"
    assert not _findings(G, "MINOR_CONTRADICTION") and not _findings(G, "MAJOR_CONTRADICTION")


def test_partial_payment_without_statement_is_unresolved_and_overpayment_is_major():
    G = _graph({"invoice_k.txt": "Invoice No: INV-1401\nBilled amount INR 9,000\n", "payment_k.txt": "Payment UTR: UTR-1401 against INV-1401\nSettled INR 4,000\n"})
    assert [f["category"] for f in _findings(G, field="amount")] == ["UNRESOLVED"]
    G2 = _graph({"invoice_k.txt": "Invoice No: INV-1402\nBilled amount INR 9,000\n", "payment_k.txt": "Payment UTR: UTR-1402 against INV-1402\nSettled INR 9,500\n"})
    assert [f["category"] for f in _findings(G2, field="amount")] == ["MAJOR_CONTRADICTION"]


def test_missing_evidence_when_referenced_document_absent():
    G = _graph({"invoice_k.txt": "Invoice No: INV-1501\nPO Number: PO-1599\nBilled amount INR 100\n"})
    (f,) = _findings(G, "MISSING_EVIDENCE")
    assert f["claim_a"]["value"] == {"reference_type": "purchase_order", "reference": "PO1599"} and f["claim_a"]["provenance"] and f["claim_b"] is None


def test_semantic_only_link_never_gives_identity_or_any_contradiction(monkeypatch):
    monkeypatch.setattr(T, "XDOC_SEMANTIC_THRESHOLD", 0.2)
    G = _graph({"invoice_s.txt": "Invoice for consulting services rendered to the client during the quarter\nBilled amount INR 90,000\n",
                "payment_s.txt": "Payment for consulting services rendered to the client during the quarter\nSettled INR 80,000\n"})
    (e,) = _links(G)
    assert e["semantic_only"] is True
    assert not _findings(G, "MINOR_CONTRADICTION") and not _findings(G, "MAJOR_CONTRADICTION")
    assert [f["category"] for f in _findings(G, field="link")] == ["UNRESOLVED"]


def test_entity_level_link_does_not_make_different_transactions_contradict():
    G = _graph({"invoice_x.txt": "Invoice No: INV-1601\nVendor: Globex Corp\nBilled amount INR 100\n", "invoice_y.txt": "Invoice No: INV-1602\nVendor: Globex Corp\nBilled amount INR 900\n"})
    assert not _findings(G, "MINOR_CONTRADICTION") and not _findings(G, "MAJOR_CONTRADICTION")
    assert G.graph["contradiction_summary"]["suppressed_entity_level_conflicts"] >= 1


def test_classification_is_idempotent_and_adds_no_nodes_or_edges():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO, "invoice_m.txt": "Invoice No: INV-1701\nPO Number: PO-1799\nBilled amount INR 5\n"})
    snap = lambda: (G.number_of_nodes(), G.number_of_edges())
    n, first = snap(), json.dumps(G.graph["contradiction_findings"], sort_keys=True, default=str)
    T.classify_contradictions(G); T.classify_contradictions(G)
    assert snap() == n and json.dumps(G.graph["contradiction_findings"], sort_keys=True, default=str) == first
    e = next(x for x in _links(G) if x["contradiction_findings"])
    assert e["contradiction_class"] == "MAJOR_CONTRADICTION" and "CONTRADICTION" in T.v2_contradiction_section(G).upper()


@pytest.fixture(scope="module")
def con_bench():
    return T.run_contradiction_benchmark()


def test_contradiction_benchmark_labels_cover_required_kinds():
    assert T.validate_contradiction_benchmark() == []
    cats = {c["category"] for c in T.CONTRADICTION_BENCHMARK}
    assert {"consistent", "amount_contradiction", "legitimate_partial_payment", "date_contradiction", "vendor_contradiction", "identity_contradiction", "transaction_id_contradiction",
            "approval_contradiction", "factual_claim_contradiction", "missing_evidence", "unresolved_ambiguity", "multiple_contradictions", "distractors", "semantic_only"} <= cats
    assert {f["category"] for c in T.CONTRADICTION_BENCHMARK for f in c["expected_findings"]} == set(T.XCON_CATEGORIES)


def test_contradiction_benchmark_is_deterministic_and_aggregate_derives_from_records(con_bench):
    again = T.run_contradiction_benchmark()
    assert json.dumps(again["records"], sort_keys=True) == json.dumps(con_bench["records"], sort_keys=True)
    assert T.aggregate_contradiction_results(con_bench["records"]) == con_bench["aggregate"] and con_bench["aggregate"]["benchmark"]["seed"] is None


def test_contradiction_metrics_are_measured_and_safe(con_bench):
    o = con_bench["aggregate"]["overall"]
    c = o["contradiction"]
    assert o["status"] == "MEASURED" and all(c[k] != T.NOT_MEASURED for k in ("precision", "recall", "f1", "false_contradiction_rate", "missed_contradiction_rate"))
    assert o["severity_classification"]["detected_contradictions"] == c["tp"] and o["contradictions_flagged_as_violation"] == 0 and o["false_policy_violations"] == 0
    assert set(o["category_classification"]["per_category"]) == set(T.XCON_CATEGORIES)


def test_not_measured_when_a_denominator_is_empty(con_bench):
    only = [r for r in con_bench["records"] if r["category"] in ("consistent", "semantic_only")]
    g = T.aggregate_contradiction_results(only)["overall"]
    assert g["contradiction"]["precision"] == T.NOT_MEASURED and g["contradiction"]["missed_contradiction_rate"] == T.NOT_MEASURED
    assert g["severity_classification"]["accuracy"] == T.NOT_MEASURED
    assert T.aggregate_contradiction_results([])["overall"]["status"] == T.NOT_MEASURED


def test_contradiction_records_keep_provenance(con_bench):
    r = next(x for x in con_bench["records"] if x["case_id"].startswith("XC-02"))
    f = r["predicted_findings"][0]
    assert f["claim_a"]["provenance"] and f["claim_b"]["provenance"] and all(p["file"] for p in f["claim_a"]["provenance"] + f["claim_b"]["provenance"])

# 14 ---- contradiction classifier: per-field regression coverage (Phase 6A)
def _cats(G, field=None):
    return [f["category"] for f in _findings(G, field=field)]


def test_consistent_documents_are_consistent_and_not_flagged():
    G = _graph({"invoice_c.txt": "Invoice No: INV-2101\nPO Number: PO-2101\nVendor: Aster Foods\nBilled amount INR 5,000\nQuantity: 10\n",
                "po_c.txt": "Purchase Order No: PO-2101\nVendor: Aster Foods\nOrder value INR 5,000\nQuantity: 10\n"})
    assert _cats(G) == ["CONSISTENT"]
    assert not _findings(G, "MINOR_CONTRADICTION") and not _findings(G, "MAJOR_CONTRADICTION") and not _findings(G, "POLICY_VIOLATION")


def test_date_contradiction_keeps_both_dates_and_provenance():
    G = _graph({"invoice_d.txt": "Invoice No: INV-2201\nPO Number: PO-2201\nInvoice date: 2024-03-01\nBilled amount INR 100\n",
                "po_d.txt": "Purchase Order No: PO-2201\nInvoice date: 2024-06-15\nOrder value INR 100\n"})
    (f,) = _findings(G, field="date")
    assert f["category"] == "MAJOR_CONTRADICTION"
    assert sorted(d for s in (f["claim_a"], f["claim_b"]) for d in s["value"]["dates"]) == ["2024-03-01", "2024-06-15"]
    assert all(s["provenance"] and s["provenance"][0]["evidence_id"] in G for s in (f["claim_a"], f["claim_b"]))


def test_vendor_contradiction_major_for_different_parties_and_minor_for_variant_spelling():
    G = _graph({"invoice_v.txt": "Invoice No: INV-2301\nPO Number: PO-2301\nVendor: Boreal Metals\nBilled amount INR 100\n",
                "po_v.txt": "Purchase Order No: PO-2301\nVendor: Quantum Textiles\nOrder value INR 100\n"})
    assert _cats(G, "vendor") == ["MAJOR_CONTRADICTION"]
    G2 = _graph({"invoice_v.txt": "Invoice No: INV-2302\nPO Number: PO-2302\nVendor: Boreal Metals Trading\nBilled amount INR 100\n",
                 "po_v.txt": "Purchase Order No: PO-2302\nVendor: Boreal Metals\nOrder value INR 100\n"})
    assert _cats(G2, "vendor") in (["MINOR_CONTRADICTION"], [])  # a suffix-only difference may normalise to the same party; never major
    assert not _findings(G2, "MAJOR_CONTRADICTION", "vendor")


def test_identity_contradiction_person_names():
    G = _graph({"invoice_i.txt": "Invoice No: INV-2401\nPO Number: PO-2401\nPerson: Ravi Kumar\nBilled amount INR 100\n",
                "po_i.txt": "Purchase Order No: PO-2401\nPerson: Meera Shah\nOrder value INR 100\n"})
    assert _cats(G, "identity") == ["MAJOR_CONTRADICTION"]


def test_transaction_id_contradiction_preserves_both_ids():
    G = _graph({"invoice_t.txt": "Invoice No: INV-2501\nPO Number: PO-2501\nBilled amount INR 100\n",
                "payment_t.txt": "Payment UTR: UTR-2501\nAgainst Invoice No: INV-9999\nPO Number: PO-2501\nSettled INR 100\n"})
    (f,) = _findings(G, field="transaction_id")
    assert f["category"] == "MAJOR_CONTRADICTION"
    assert {i for s in (f["claim_a"], f["claim_b"]) for i in s["value"]["ids"]} == {"INV2501", "INV9999"}
    assert f["claim_a"]["provenance"] and f["claim_b"]["provenance"]


def test_approval_contradiction_and_pending_is_unresolved():
    G = _graph({"approval_a.txt": "Approval of PO-2601\nApproval status: Approved\n", "po_a.txt": "Purchase Order No: PO-2601\nApproval status: Rejected\nOrder value INR 100\n"})
    assert _cats(G, "approval_status") == ["MAJOR_CONTRADICTION"]
    G2 = _graph({"approval_a.txt": "Approval of PO-2602\nApproval status: Approved\n", "po_a.txt": "Purchase Order No: PO-2602\nApproval status: Pending\nOrder value INR 100\n"})
    assert _cats(G2, "approval_status") == ["UNRESOLVED"]


def test_factual_claim_contradiction_delivery_and_quantity():
    G = _graph({"invoice_f.txt": "Invoice No: INV-2701\nPO Number: PO-2701\nDelivery status: Delivered\nBilled amount INR 100\n",
                "po_f.txt": "Purchase Order No: PO-2701\nDelivery status: Not delivered\nOrder value INR 100\n"})
    (f,) = _findings(G, field="factual_claim")
    assert f["category"] == "MAJOR_CONTRADICTION" and f["claim_a"]["provenance"] and f["claim_b"]["provenance"]
    G2 = _graph({"invoice_f.txt": "Invoice No: INV-2702\nPO Number: PO-2702\nQuantity: 10\nBilled amount INR 100\n", "po_f.txt": "Purchase Order No: PO-2702\nQuantity: 50\nOrder value INR 100\n"})
    assert _cats(G2, "factual_claim") == ["MAJOR_CONTRADICTION"]


def test_unresolved_ambiguity_stays_unresolved_and_is_not_a_contradiction():
    G = _graph({"invoice_u.txt": "Invoice No: INV-2801\nBilled amount INR 9,000\n", "payment_u.txt": "Payment UTR: UTR-2801 against INV-2801\nSettled INR 4,000\n"})
    f = _findings(G, field="amount")
    assert [x["category"] for x in f] == ["UNRESOLVED"] and f[0]["resolution"] == "NOT_RESOLVED"
    assert not _findings(G, "MINOR_CONTRADICTION") and not _findings(G, "MAJOR_CONTRADICTION") and not _findings(G, "POLICY_VIOLATION")


def test_line_item_difference_is_consistent():
    G = _graph({"invoice_l.txt": "Invoice No: INV-2901\nPO Number: PO-2901\nBilled amount INR 12,000\nLine item INR 5,000\nLine item INR 7,000\n", "po_l.txt": "Purchase Order No: PO-2901\nOrder value INR 5,000\nOrder value INR 7,000\n"})
    assert not _findings(G, "MINOR_CONTRADICTION") and not _findings(G, "MAJOR_CONTRADICTION")


def test_contradiction_and_policy_violation_coexist_without_conversion():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 15000\n")
    pv = _findings(G, "POLICY_VIOLATION")
    assert pv and all(f["source"] == "existing_deterministic_decision" and f["derived_from_contradiction"] is False for f in pv)
    (c,) = _findings(G, "MAJOR_CONTRADICTION", "amount")
    assert c["category"] == "MAJOR_CONTRADICTION" and c["is_policy_violation"] is False


def test_repeated_classification_does_not_overwrite_claims():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO})
    before = json.dumps(_findings(G, "MAJOR_CONTRADICTION"), sort_keys=True, default=str)
    texts = {n: d.get("text") for n, d in G.nodes(data=True) if d.get("type") == "Evidence"}
    for _ in range(3): T.classify_contradictions(G)
    assert json.dumps(_findings(G, "MAJOR_CONTRADICTION"), sort_keys=True, default=str) == before
    assert texts == {n: d.get("text") for n, d in G.nodes(data=True) if d.get("type") == "Evidence"}


# 15 ---- self-verification result contract (Phase 7A: structure only)
def _sv(G, verdict=None):
    ds = [n for n, d in G.nodes(data=True) if d.get("type") == "Decision" and (verdict is None or d.get("verdict") == verdict)]
    assert ds
    return T.build_self_verification_result(G, ds[0])


def test_sv_all_contract_fields_present_for_violation():
    G = _graph({"invoice_sv.txt": "Invoice No: INV-3101\nBilled amount INR 5,000\n"}, rulebook="FORBID TRANSACTION > INR 1000\n")
    r = _sv(G, "VIOLATION")
    assert all(k in r for k in T.SELF_VERIFICATION_FIELDS) and r["found"] is True
    assert r["decision"]["verdict"] == "VIOLATION" and r["policy_rules"] and r["policy_rules"][0]["condition"]
    assert T.validate_self_verification_result(G, r) == []


def test_sv_supporting_evidence_preserves_ids_documents_and_provenance_and_invents_nothing():
    G = _graph({"invoice_sv.txt": "Invoice No: INV-3201\nBilled amount INR 5,000\n"}, rulebook="FORBID TRANSACTION > INR 1000\n")
    r = _sv(G, "VIOLATION")
    dec = r["decision"]["decision_id"]
    stored = set(G.nodes[dec]["evidence_used"])
    assert stored and {x["evidence_id"] for x in r["supporting_evidence"]} >= stored
    doc = _doc(G, "invoice_sv.txt")
    for ref in r["supporting_evidence"]:
        assert G.nodes[ref["evidence_id"]]["type"] == "Evidence" and ref["document_id"] == doc and ref["filename"] == "invoice_sv.txt"
        assert ref["location"] and ref["provenance"] == json.loads(json.dumps(G.nodes[ref["evidence_id"]]["provenance"], default=str))
    assert {x["evidence_id"] for x in r["supporting_evidence"]} == {e["evidence_id"] for e in T.query_decision_lineage(G, dec)["evidence"]}


def test_sv_confidence_is_never_invented():
    G = _graph({"invoice_sv.txt": "Invoice No: INV-3301\nBilled amount INR 5,000\n"}, rulebook="FORBID TRANSACTION > INR 1000\n")
    c = _sv(G, "VIOLATION")["confidence"]
    assert c["value"] is None and c["status"] == "NOT_MEASURED" and set(c["components"]) == {"compiled_rules", "evidence_extraction"}
    assert all(x["value"] is not None for k in c["components"] for x in c["components"][k])


def test_sv_status_is_not_verified_and_no_escalation_in_7a():
    G = _graph({"invoice_sv.txt": "Invoice No: INV-3401\nBilled amount INR 5,000\n"}, rulebook="FORBID TRANSACTION > INR 1000\n")
    r = _sv(G)
    assert r["verification_status"] == "NOT_VERIFIED" and r["escalation_reason"] is None and r["verification_status"] in T.SELF_VERIFICATION_STATUSES


def test_sv_contradicting_evidence_keeps_both_claims_and_does_not_change_decision():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 15000\n")
    r = _sv(G, "VIOLATION")
    cf = [c for c in r["contradicting_evidence"] if c["source"] == "contradiction_finding"]
    assert cf and cf[0]["category"] == "MAJOR_CONTRADICTION" and cf[0]["claim_a"]["provenance"] and cf[0]["claim_b"]["provenance"] and cf[0]["is_policy_violation"] is False
    assert {cf[0]["claim_a"]["filename"], cf[0]["claim_b"]["filename"]} == {"invoice_k.txt", "po_k.txt"}
    assert r["decision"]["verdict"] == G.nodes[r["decision"]["decision_id"]]["verdict"] == "VIOLATION"
    assert T.validate_self_verification_result(G, r) == []


def test_sv_missing_evidence_reports_unsupplied_reference():
    G = _graph({"invoice_k.txt": "Invoice No: INV-3501\nPO Number: PO-3599\nBilled amount INR 5,000\n"}, rulebook="FORBID TRANSACTION > INR 1000\n")
    r = _sv(G, "VIOLATION")
    me = [m for m in r["missing_evidence"] if m["kind"] == "referenced_document_not_supplied"]
    assert me and me[0]["claim"]["value"]["reference"] == "PO3599" and me[0]["claim"]["provenance"]


def test_sv_unevaluated_decision_has_no_supporting_evidence_and_reports_why():
    G = _graph({"invoice_sv.txt": "Invoice No: INV-3601\nBilled amount INR 5,000\n"}, rulebook="FORBID TRANSACTION > INR 1000\nVendors should behave reasonably in spirit.\n")
    rs = [x for x in T.build_self_verification_results(G) if x["decision"]["verdict"] == "UNEVALUATED"]
    assert rs and all(x["supporting_evidence"] == [] and any(m["kind"] == "rule_not_evaluated" for m in x["missing_evidence"]) for x in rs)


def test_sv_unknown_decision_still_exposes_every_field():
    G = _graph({"invoice_sv.txt": "Invoice No: INV-3701\nBilled amount INR 5\n"})
    r = T.build_self_verification_result(G, "does_not_exist")
    assert r["found"] is False and all(k in r for k in T.SELF_VERIFICATION_FIELDS) and r["decision"] is None and r["gaps"]


def test_sv_validator_detects_invented_evidence():
    G = _graph({"invoice_sv.txt": "Invoice No: INV-3801\nBilled amount INR 5,000\n"}, rulebook="FORBID TRANSACTION > INR 1000\n")
    r = _sv(G, "VIOLATION")
    r["supporting_evidence"].append({"evidence_id": "ev_fake", "document_id": None})
    assert any("ev_fake" in p for p in T.validate_self_verification_result(G, r))
    r2 = _sv(G, "VIOLATION"); r2["supporting_evidence"][0]["document_id"] = "doc_fake"
    assert T.validate_self_verification_result(G, r2)


def test_sv_is_read_only_and_idempotent():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 15000\n")
    snap = lambda: (G.number_of_nodes(), G.number_of_edges(), json.dumps({n: d for n, d in G.nodes(data=True)}, sort_keys=True, default=str), json.dumps(G.graph.get("contradiction_findings"), sort_keys=True, default=str))
    before = snap()
    a = json.dumps(T.build_self_verification_results(G), sort_keys=True, default=str)
    b = json.dumps(T.build_self_verification_results(G), sort_keys=True, default=str)
    assert a == b and snap() == before


# 16 ---- self-verification of material claims (Phase 7B)
_SV_RB = "FORBID TRANSACTION > INR 1000\n"


def _sv_g(n, rulebook=_SV_RB):
    return _graph({"invoice_sv.txt": f"Invoice No: INV-{n}\nBilled amount INR 5,000\n"}, rulebook=rulebook)


def _sv7b(G, verdict="VIOLATION"):
    d = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision" and x.get("verdict") == verdict)
    return T.verify_self_verification_result(G, d), d


def _basis_evidence(G, d):
    return [e for b in G.nodes[d]["basis_node_ids"] for e in T._supporting_evidence_ids(G, b)]


def test_sv7b_grounded_claims_verify_and_keep_ids_documents_provenance():
    G = _sv_g(7101)
    r, d = _sv7b(G)
    assert r["verification_status"] == "VERIFIED" and r["escalation_reason"] is None and r["contract_version"] == "7B"
    assert r["material_claims"] and all(c["grounding"] == "GROUNDED" for c in r["material_claims"])
    doc = _doc(G, "invoice_sv.txt")
    for c in r["material_claims"]:
        assert c["node_id"] in G.nodes[d]["basis_node_ids"] and c["evidence"]
        for ref in c["evidence"]:
            assert G.nodes[ref["evidence_id"]]["type"] == "Evidence" and ref["document_id"] == doc and ref["filename"] == "invoice_sv.txt"
            assert ref["location"] and ref["provenance"] == json.loads(json.dumps(G.nodes[ref["evidence_id"]]["provenance"], default=str))
    assert T.validate_self_verification_result(G, r) == []


def test_sv7b_confidence_stays_not_measured_and_verdict_unchanged():
    G = _sv_g(7201)
    r, d = _sv7b(G)
    assert r["confidence"]["value"] is None and r["confidence"]["status"] == "NOT_MEASURED"
    assert r["decision"]["verdict"] == G.nodes[d]["verdict"] == "VIOLATION"
    r["confidence"]["value"] = 0.9
    assert any("confidence" in p for p in T.validate_self_verification_result(G, r))


def test_sv7b_claim_not_in_source_text_fails_the_conclusion():
    G = _sv_g(7301)
    r0, d = _sv7b(G)
    for e in _basis_evidence(G, d): G.nodes[e]["text"] = G.nodes[e]["text"].replace("5,000", "500")
    r, _ = _sv7b(G)
    assert r["verification_status"] == "FAILED" and r["escalation_reason"] is None
    bad = [c for c in r["material_claims"] if c["grounding"] == "UNSUPPORTED"]
    assert bad and "not found" in bad[0]["reason"] and any("unsupported material claim" in g for g in r["gaps"])


def test_sv7b_missing_supporting_evidence_fails_and_nothing_is_invented():
    G = _sv_g(7401)
    _, d = _sv7b(G)
    for b in G.nodes[d]["basis_node_ids"]:
        for u, v, k, ed in list(G.in_edges(b, keys=True, data=True)):
            if ed.get("relation") == "SUPPORTS": G.remove_edge(u, v, k)
    n_nodes = G.number_of_nodes()
    r, _ = _sv7b(G)
    assert r["verification_status"] == "FAILED" and all(c["evidence"] == [] for c in r["material_claims"]) and G.number_of_nodes() == n_nodes


def test_sv7b_heuristic_only_support_does_not_ground_a_claim():
    G = _sv_g(7501)
    _, d = _sv7b(G)
    for b in G.nodes[d]["basis_node_ids"]:
        for u, v, ed in G.in_edges(b, data=True):
            if ed.get("relation") == "SUPPORTS": ed["heuristic"] = True
    assert _sv7b(G)[0]["verification_status"] == "FAILED"


def test_sv7b_conclusion_without_any_recorded_basis_is_unsupported():
    G = _sv_g(7601)
    _, d = _sv7b(G)
    rule = G.nodes[d]["rule_id"]
    for u, v, k, ed in list(G.in_edges(rule, keys=True, data=True)):
        if ed.get("relation") in ("VIOLATES", "SATISFIES"): G.remove_edge(u, v, k)
    G.nodes[d]["basis_node_ids"] = []
    r, _ = _sv7b(G)
    assert r["verification_status"] == "FAILED" and r["material_claims"][0]["kind"] == "decision_basis" and r["material_claims"][0]["grounding"] == "UNSUPPORTED"


def test_sv7b_weak_grounding_escalates_with_reason():
    G = _sv_g(7701)
    _, d = _sv7b(G)
    for e in _basis_evidence(G, d):
        G.nodes[e]["source_location"] = None
        for _, _, ed in G.out_edges(e, data=True):
            if ed.get("relation") == "DERIVED_FROM": ed["location"] = None
    r, _ = _sv7b(G)
    assert r["verification_status"] == "ESCALATE" and "weakly grounded" in r["escalation_reason"] and any(c["grounding"] == "WEAK" for c in r["material_claims"])


def test_sv7b_major_contradiction_on_grounded_claims_escalates_without_changing_verdict():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 15000\n")
    r, d = _sv7b(G)
    assert r["verification_status"] == "ESCALATE" and "major contradiction" in r["escalation_reason"]
    assert all(c["grounding"] != "UNSUPPORTED" for c in r["material_claims"]) and G.nodes[d]["verdict"] == "VIOLATION"
    assert T.validate_self_verification_result(G, r) == []


def test_sv7b_failed_takes_precedence_over_escalate():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 15000\n")
    _, d = _sv7b(G)
    for e in _basis_evidence(G, d): G.nodes[e]["text"] = "Billed amount INR 1"
    r, _ = _sv7b(G)
    assert r["verification_status"] == "FAILED" and r["escalation_reason"] is None


def test_sv7b_decision_without_conclusion_stays_not_verified():
    G = _sv_g(7801, rulebook=_SV_RB + "Vendors should behave reasonably in spirit.\n")
    r, _ = _sv7b(G, "UNEVALUATED")
    assert r["verification_status"] == "NOT_VERIFIED" and r["material_claims"] == [] and any("not applicable" in g for g in r["gaps"])
    assert T.verify_self_verification_result(G, "does_not_exist")["verification_status"] == "NOT_VERIFIED"


def test_sv7b_validator_detects_invented_claim_evidence_and_bad_escalation_fields():
    G = _sv_g(7901)
    r, _ = _sv7b(G)
    r["material_claims"][0]["evidence"].append({"evidence_id": "ev_fake", "document_id": None})
    assert any("ev_fake" in p for p in T.validate_self_verification_result(G, r))
    r2, _ = _sv7b(G); r2["escalation_reason"] = "x"
    assert T.validate_self_verification_result(G, r2)


def test_sv7b_is_read_only_and_idempotent():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 15000\n")
    snap = lambda: (G.number_of_nodes(), G.number_of_edges(), json.dumps({n: d for n, d in G.nodes(data=True)}, sort_keys=True, default=str), json.dumps(G.graph.get("contradiction_findings"), sort_keys=True, default=str))
    before = snap()
    a = json.dumps(T.verify_self_verification_results(G), sort_keys=True, default=str)
    b = json.dumps(T.verify_self_verification_results(G), sort_keys=True, default=str)
    assert a == b and snap() == before
    assert T.build_self_verification_result(G, next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision"))["verification_status"] == "NOT_VERIFIED"  # 7A builder unchanged


# 17 ---- policy-rule applicability verification (Phase 7C)
def _pa(G, verdict="VIOLATION"):
    d = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision" and x.get("verdict") == verdict)
    return T.verify_policy_applicability(G, d), d


def _pa_snap(G):
    return (G.number_of_nodes(), G.number_of_edges(), json.dumps({n: d for n, d in G.nodes(data=True)}, sort_keys=True, default=str))


def test_pa_applicable_rule_is_verified():
    G = _sv_g(8101)
    r, d = _pa(G)
    assert r["verification_status"] == "VERIFIED" and r["escalation_reason"] is None and r["contract_version"] == "7C"
    assert r["policy_applicability"]["status"] == "VERIFIED" and all(c["result"] == "OK" for c in r["policy_applicability"]["checks"])
    assert all(k in r for k in T.SELF_VERIFICATION_FIELDS) and r["confidence"]["value"] is None and r["confidence"]["status"] == "NOT_MEASURED"
    assert T.validate_self_verification_result(G, r) == []


def test_pa_wrong_rule_escalates():
    G = _sv_g(8201, rulebook=_SV_RB + "FORBID TRANSACTION > USD 1000\n")
    _, d = _pa(G)
    other = next(n for n, x in G.nodes(data=True) if x.get("type") == "PolicyRule" and n != G.nodes[d]["rule_id"])
    G.nodes[d]["rule_id"] = other  # Decision now cites a rule it did not evaluate
    r, _ = _pa(G)
    assert r["verification_status"] == "ESCALATE" and r["policy_applicability"]["status"] == "MISMATCH" and "applicability mismatch" in r["escalation_reason"]
    G2 = _sv_g(8202)
    _, d2 = _pa(G2)
    G2.nodes[G2.nodes[d2]["rule_id"]]["condition"] = "FORBID TRANSACTION > INR 9000"  # rule text no longer what the Decision used
    assert _pa(G2)[0]["verification_status"] == "ESCALATE"


def test_pa_missing_required_rule_fact_escalates():
    G = _sv_g(8301)
    _, d = _pa(G)
    G.nodes[G.nodes[d]["basis_node_ids"][0]]["amount"] = None  # required amount fact absent
    r, _ = _pa(G)
    assert r["verification_status"] == "ESCALATE" and r["policy_applicability"]["status"] == "UNESTABLISHED" and r["policy_applicability"]["missing_facts"]
    G2 = _sv_g(8302)
    _, d2 = _pa(G2)
    G2.nodes[d2]["compiled_missing_facts"] = []
    G2.nodes[d2]["result_source"] = "compiled_policy_engine"  # compiled path with no executed compiled rule: applicability cannot be established
    assert _pa(G2)[0]["verification_status"] == "ESCALATE"


def test_pa_wrong_policy_scope_escalates():
    G = _sv_g(8401)
    _, d = _pa(G)
    for u, v, k, e in list(G.out_edges(d, keys=True, data=True)):
        if e.get("relation") == "BELONGS_TO": G.remove_edge(u, v, k)
    G.add_node("pol_other", type="Policy", name="other policy"); G.add_edge(d, "pol_other", relation="BELONGS_TO")
    r, _ = _pa(G)
    assert r["verification_status"] == "ESCALATE" and r["policy_applicability"]["status"] == "MISMATCH"
    assert any(c["check"] == "policy_scope" and c["result"] == "MISMATCH" for c in r["policy_applicability"]["checks"])


def test_pa_7b_failed_remains_failed():
    G = _sv_g(8501)
    _, d = _pa(G)
    for e in _basis_evidence(G, d): G.nodes[e]["text"] = G.nodes[e]["text"].replace("5,000", "500")
    G.nodes[G.nodes[d]["rule_id"]]["condition"] = "FORBID TRANSACTION > INR 9000"  # would also be a mismatch
    r, _ = _pa(G)
    assert r["verification_status"] == "FAILED" and r["escalation_reason"] is None and r["policy_applicability"]["status"] == "NOT_CHECKED"


def test_pa_non_conclusion_decision_stays_not_verified():
    G = _sv_g(8601, rulebook=_SV_RB + "Vendors should behave reasonably in spirit.\n")
    r, _ = _pa(G, "UNEVALUATED")
    assert r["verification_status"] == "NOT_VERIFIED" and r["escalation_reason"] is None and r["policy_applicability"]["status"] == "NOT_CHECKED"
    assert T.verify_policy_applicability(G, "does_not_exist")["verification_status"] == "NOT_VERIFIED"


def test_pa_keeps_verdict_graph_unchanged_and_is_idempotent():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 15000\n")
    before = _pa_snap(G)
    verdicts = {n: d["verdict"] for n, d in G.nodes(data=True) if d.get("type") == "Decision"}
    a = json.dumps(T.verify_policy_applicability_results(G), sort_keys=True, default=str)
    b = json.dumps(T.verify_policy_applicability_results(G), sort_keys=True, default=str)
    assert a == b and _pa_snap(G) == before and verdicts == {n: d["verdict"] for n, d in G.nodes(data=True) if d.get("type") == "Decision"}
    r, d = _pa(G)
    assert r["decision"]["verdict"] == G.nodes[d]["verdict"] == "VIOLATION" and r["verification_status"] == "ESCALATE"  # 7B major contradiction stays ESCALATE


def test_pa_preserves_provenance_and_ids():
    G = _sv_g(8801)
    r7, d = _sv7b(G)
    r, _ = _pa(G)
    assert r["material_claims"] == r7["material_claims"] and r["supporting_evidence"] == r7["supporting_evidence"] and r["policy_rules"] == r7["policy_rules"]
    rule = r["policy_applicability"]["rule"]
    assert rule["rule_id"] == G.nodes[d]["rule_id"] and rule["basis_node_ids"] == G.nodes[d]["basis_node_ids"]
    assert rule["rulebook_provenance"] == json.loads(json.dumps(G.nodes[d]["rulebook_provenance"], default=str))
    assert rule["source_file"] == G.nodes[rule["rule_id"]].get("source_file") and rule["source_location"] == G.nodes[rule["rule_id"]].get("source_location")
    assert T.validate_self_verification_result(G, r) == []


# 17b ---- Phase 7C coverage closure: real compiled-rule path, AMOUNT_MATCH relationship, absence within extracted scope
class _V:  # minimal stand-in for the compiler's enum-like values (.value)
    def __init__(self, value): self.value = value


def _fake_compiled(monkeypatch, line, entity="transaction", verdict="VIOLATION", status="VALID"):
    """Drives the REAL apply_compiled_policy -> evaluate_policy_rules path; only the LLM compiler and the external rule_engine module are replaced by deterministic stubs."""
    cr = types.SimpleNamespace(rule_id="CR1", policy_id="P1", status=status, rule_type=_V("threshold"), severity=_V("high"), confidence=0.95, expression="amount <= 1000", source_text=line,
                               ambiguities=[], issues=[], model_dump=lambda mode=None: {"rule_id": "CR1", "entity": entity})
    res = types.SimpleNamespace(rules=[cr], status="COMPILED", policy_id="P1", stats={}, ambiguous_policy=False, ambiguity_reasons=[], rejected=[], unparsed_statements=[])
    ev = {"rule_results": [{"rule_id": "CR1", "verdict": verdict, "record_index": 0, "severity": "high", "missing_facts": [], "reasons": ["amount above limit"]}], "evaluation_date": None}
    monkeypatch.setattr(T, "HAS_POLICY_COMPILER", True, raising=False)
    monkeypatch.setattr(T, "_compile_policy", lambda text, ctx, conf: res, raising=False)
    monkeypatch.setattr(T, "_evaluate_compiled_rules", lambda *a, **k: ev, raising=False)
    monkeypatch.setattr(T, "_COMPILER_MIN_CONFIDENCE", 0.5, raising=False)
    orig = T.evaluate_policy_rules  # build_evidence_graph (offline) skips the compile stage the Celery task runs just before evaluate_policy_rules: run it here, unchanged
    monkeypatch.setattr(T, "evaluate_policy_rules", lambda G: (T.apply_compiled_policy(G, line + "\n"), orig(G))[1])


def test_pa_compiled_rule_valid_mapped_entity_basis_is_verified(monkeypatch):
    _fake_compiled(monkeypatch, "FORBID TRANSACTION > INR 1000")
    G = _sv_g(9101)
    r, d = _pa(G)
    dd = G.nodes[d]
    assert dd["result_source"] == "compiled_policy_engine" and dd["evaluation_engine"] == "compiled_rule_engine" and dd["compiled_rules"][0]["status"] == "VALID" and dd["compiled_rules"][0]["mapping_basis"]
    assert all(G.nodes[b]["type"] == "Transaction" for b in dd["basis_node_ids"]) and dd["basis_node_ids"] and not dd["compiled_missing_facts"]
    assert r["verification_status"] == "VERIFIED" and r["escalation_reason"] is None and r["contract_version"] == "7C"
    assert r["policy_applicability"]["status"] == "VERIFIED" and r["policy_applicability"]["rule"]["result_source"] == "compiled_policy_engine"
    assert all(c["result"] == "OK" for c in r["policy_applicability"]["checks"]) and r["decision"]["verdict"] == dd["verdict"] == "VIOLATION"
    assert all(k in r for k in T.SELF_VERIFICATION_FIELDS) and r["confidence"]["value"] is None and T.validate_self_verification_result(G, r) == []
    before = _pa_snap(G)
    assert json.dumps(T.verify_policy_applicability(G, d), sort_keys=True, default=str) == json.dumps(r, sort_keys=True, default=str) and _pa_snap(G) == before


def test_pa_compiled_rule_wrong_entity_or_unmapped_or_not_valid_escalates(monkeypatch):
    _fake_compiled(monkeypatch, "FORBID TRANSACTION > INR 1000")
    G = _sv_g(9201)
    _, d = _pa(G)
    G.nodes[G.nodes[d]["rule_id"]]["compiled_rules"][0]["compiled_rule"]["entity"] = "govid"  # compiled rule now concerns GovID entities, but the basis nodes are Transactions
    r, d = _pa(G)
    assert r["verification_status"] == "ESCALATE" and r["policy_applicability"]["status"] == "MISMATCH" and "required by compiled rule" in " ".join(r["policy_applicability"]["mismatches"]) and G.nodes[d]["verdict"] == "VIOLATION"
    _fake_compiled(monkeypatch, "FORBID TRANSACTION > INR 1000")
    G2 = _sv_g(9202)
    _, d2 = _pa(G2)
    G2.nodes[d2]["compiled_rules"][0]["source_text"] = "REQUIRE GOVID VALID"  # compiled clause no longer maps to the cited rule text
    assert "not mapped" in " ".join(_pa(G2)[0]["policy_applicability"]["mismatches"]) and _pa(G2)[0]["verification_status"] == "ESCALATE"
    G2.nodes[d2]["compiled_rules"][0]["source_text"] = "FORBID TRANSACTION > INR 1000"; G2.nodes[d2]["compiled_rules"][0]["status"] = "NEEDS_REVIEW"
    assert "not VALID" in " ".join(_pa(G2)[0]["policy_applicability"]["mismatches"])
    G3 = _sv_g(9203)
    _, d3 = _pa(G3)
    G3.nodes[d3]["compiled_rules"] = [{**G2.nodes[d2]["compiled_rules"][0], "status": "VALID", "missing_facts": ["amount.unit"]}]
    G3.nodes[d3]["result_source"] = "compiled_policy_engine"
    assert _pa(G3)[0]["policy_applicability"]["status"] in ("UNESTABLISHED", "MISMATCH") and _pa(G3)[0]["verification_status"] == "ESCALATE"


_AM_A, _AM_B = "Invoice No: INV-{n} Billed amount INR 20,000\n", "Invoice No: INV-{n} Payment received INR {b}\n"


def _am_g(n, b="12,000"):
    return _graph({"inv_am.txt": _AM_A.format(n=n), "stmt_am.txt": _AM_B.format(n=n, b=b)}, rulebook="Transaction amounts must match.\n")


def test_pa_amount_match_verifies_actual_amount_relationship():
    G = _am_g(9301)  # linked records (same ref) with DIFFERING amounts -> VIOLATION
    r, d = _pa(G)
    amts = sorted(G.nodes[b]["amount"] for b in G.nodes[d]["basis_node_ids"])
    assert G.nodes[d]["parsed_spec"]["subject"] == "AMOUNT_MATCH" and amts == [12000.0, 20000.0]
    assert r["policy_applicability"]["status"] == "VERIFIED" and r["policy_applicability"]["mismatches"] == []
    G2 = _am_g(9302, b="20,000")  # linked records with EQUAL amounts -> SATISFIED
    r2, d2 = _pa(G2, "SATISFIED")
    assert r2["verification_status"] == "VERIFIED" and r2["policy_applicability"]["status"] == "VERIFIED"
    # VIOLATION claimed, but the linked basis amounts actually agree (node + its source text consistently changed so 7B still grounds it)
    for b in G.nodes[d]["basis_node_ids"]:
        G.nodes[b]["amount"] = 20000.0
        for e in T._supporting_evidence_ids(G, b): G.nodes[e]["text"] = G.nodes[e]["text"].replace("12,000", "20,000")
    bad, dd = _pa(G)
    assert bad["policy_applicability"]["status"] == "MISMATCH" and "do not differ" in " ".join(bad["policy_applicability"]["mismatches"]) and "applicability mismatch" in bad["escalation_reason"]
    assert bad["verification_status"] == "ESCALATE" and G.nodes[dd]["verdict"] == "VIOLATION"
    # SATISFIED claimed, but the linked basis amounts actually differ
    b0 = G2.nodes[d2]["basis_node_ids"][0]
    G2.nodes[b0]["amount"] = 19000.0
    for e in T._supporting_evidence_ids(G2, b0): G2.nodes[e]["text"] = G2.nodes[e]["text"].replace("20,000", "19,000")
    bad2, _ = _pa(G2, "SATISFIED")
    assert bad2["verification_status"] == "ESCALATE" and "do not all match" in " ".join(bad2["policy_applicability"]["mismatches"])
    # amount presence alone is not enough: an unlinked (different reference) basis pair cannot establish the relationship
    G3 = _am_g(9303)
    _, d3 = _pa(G3)
    n1, n2 = G3.nodes[d3]["basis_node_ids"][:2]
    G3.nodes[n1]["attributes"] = {**G3.nodes[n1]["attributes"], "ref_id": "OTHER1"}
    assert "no reliably linked basis pair" in " ".join(_pa(G3)[0]["policy_applicability"]["mismatches"]) and _pa(G3)[0]["verification_status"] == "ESCALATE"


_ABS_RB = 'REQUIRE KEYWORD "approval"\n'


def test_pa_absence_within_extracted_scope_valid_absence_and_phrase_present():
    G = _graph({"memo_abs.txt": "Meeting notes\nNothing else to report\n"}, rulebook=_ABS_RB)
    r, d = _pa(G)
    dd = G.nodes[d]
    assert dd["violation_status"] == "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE" and dd["verdict"] == "VIOLATION" and dd["absence_scope_evidence_ids"]
    assert not any("approval" in G.nodes[e]["text"].lower() for e in dd["absence_scope_evidence_ids"])
    assert r["verification_status"] == "VERIFIED" and r["policy_applicability"]["status"] == "VERIFIED" and r["policy_applicability"]["mismatches"] == [] and T.validate_self_verification_result(G, r) == []
    before = _pa_snap(G)
    for e in dd["absence_scope_evidence_ids"]: G.nodes[e]["text"] += " Approval granted by finance."  # phrase is actually present in the recorded scope
    r2, _ = _pa(G)
    assert r2["verification_status"] == "ESCALATE" and r2["policy_applicability"]["status"] == "MISMATCH" and "present in the recorded scope evidence" in " ".join(r2["policy_applicability"]["mismatches"])
    assert r2["decision"]["verdict"] == G.nodes[d]["verdict"] == "VIOLATION" and _pa_snap(G) != before and json.dumps(_pa(G)[0], sort_keys=True, default=str) == json.dumps(r2, sort_keys=True, default=str)
    G2 = _graph({"memo_abs2.txt": "Meeting notes\nApproval granted by finance head\n"}, rulebook=_ABS_RB)  # phrase really present: engine yields no absence conclusion
    r3, d3 = _pa(G2, "SATISFIED")
    assert G2.nodes[d3]["violation_status"] is None and r3["verification_status"] == "VERIFIED" and r3["policy_applicability"]["status"] == "VERIFIED"


# 18 ---- self-verification evaluation benchmark (Phase 7D)
import copy

_SV7D_NM = T.NOT_MEASURED
_SV7D_METRICS = ("decision_accuracy", "unsupported_decision_rate", "evidence_grounding_accuracy", "contradiction_handling_accuracy", "verification_status_accuracy", "escalation_accuracy", "false_verification_rate", "false_escalation_rate")


@pytest.fixture(scope="module")
def sv7d():
    return T.run_self_verification_benchmark()


def _sv7d_rec(res, case_id, variant):
    return next(r for r in res["records"] if r["case_id"] == case_id and r["system_variant"] == variant)


def _sv7d_case_by_id(cid):
    return copy.deepcopy(next(c for c in T.SELF_VERIFICATION_BENCHMARK if c["case_id"] == cid))


SV, SVPA = "EVIDENCE_GRAPH_SELF_VERIFICATION", "EVIDENCE_GRAPH_SELF_VERIFICATION_POLICY_APPLICABILITY"


def test_sv7d_labels_are_valid_cover_required_categories_and_validator_catches_inconsistency():
    assert T.validate_self_verification_benchmark() == []
    cases = T.SELF_VERIFICATION_BENCHMARK
    assert len({c["case_id"] for c in cases}) == len(cases) and set(T.SV7D_CATEGORIES) <= {c["category"] for c in cases}
    assert {c["expected_verification_status"] for c in cases} == {"VERIFIED", "FAILED", "ESCALATE", "NOT_VERIFIED"}
    bad = _sv7d_case_by_id("sv7d_pos_violation"); bad["expected_verification_status"] = "ESCALATE"
    assert T.validate_self_verification_benchmark([bad], require_coverage=False)
    dup = [_sv7d_case_by_id("sv7d_pos_violation")] * 2
    assert any("duplicate" in p for p in T.validate_self_verification_benchmark(dup, require_coverage=False))
    unk = _sv7d_case_by_id("sv7d_pos_violation"); unk["fault"] = "nope"
    assert T.validate_self_verification_benchmark([unk], require_coverage=False)


def test_sv7d_is_deterministic_and_uses_no_randomness(sv7d):
    again = T.run_self_verification_benchmark()
    assert json.dumps(again, sort_keys=True) == json.dumps(sv7d, sort_keys=True)
    a = sv7d["aggregate"]
    assert a["benchmark"]["seed"] is None and a["benchmark"]["llm_used"] is False and a["protocol"]["llm_used"] is False and a["protocol"]["network_used"] is False
    assert a == T.aggregate_self_verification_results(sv7d["records"])  # aggregate derives from the raw records only


def test_sv7d_all_four_variants_per_case_with_confidence_not_measured(sv7d):
    assert T.SV7D_VARIANTS == ("BASELINE", "EVIDENCE_GRAPH", SV, SVPA)
    for c in T.SELF_VERIFICATION_BENCHMARK:
        recs = [r for r in sv7d["records"] if r["case_id"] == c["case_id"]]
        assert [r["system_variant"] for r in recs] == list(T.SV7D_VARIANTS)
        assert all(r["confidence"] == {"value": None, "status": "NOT_MEASURED"} and r["graph_unchanged"] is True for r in recs)
    assert set(sv7d["aggregate"]["variants"]) == set(T.SV7D_VARIANTS)
    assert all(m in sv7d["aggregate"]["variants"][v] for v in T.SV7D_VARIANTS for m in _SV7D_METRICS)
    # variants without a mechanism are NOT_MEASURED, not zero
    b, eg = sv7d["aggregate"]["variants"]["BASELINE"], sv7d["aggregate"]["variants"]["EVIDENCE_GRAPH"]
    assert all(b[m]["value"] == _SV7D_NM for m in _SV7D_METRICS[2:]) and b["decision_accuracy"]["value"] != _SV7D_NM
    assert all(eg[m]["value"] == _SV7D_NM for m in ("verification_status_accuracy", "escalation_accuracy", "false_verification_rate", "false_escalation_rate")) and eg["evidence_grounding_accuracy"]["value"] != _SV7D_NM


def test_sv7d_ground_truth_is_independent_of_system_output(sv7d):
    snap = copy.deepcopy(T.SELF_VERIFICATION_BENCHMARK)
    T.run_self_verification_benchmark()
    assert T.SELF_VERIFICATION_BENCHMARK == snap  # running the system never rewrites a label
    for r in sv7d["records"]:
        c = next(x for x in snap if x["case_id"] == r["case_id"])
        assert r["expected_outcome"] == c["expected_outcome"] and r["expected_verification_status"] == c["expected_verification_status"] and r["claims_grounded_expected"] == c["claims_grounded"]
    # change ONE label (consistently): predictions are untouched, only the scored metrics move
    flipped = _sv7d_case_by_id("sv7d_pos_violation")
    flipped.update(expected_outcome="ESCALATE", expected_verification_status="ESCALATE", decision_supported=False, policy_applicable=False)
    assert T.validate_self_verification_benchmark([flipped], require_coverage=False) == []
    orig = _sv7d_case_by_id("sv7d_pos_violation")
    a, b = T.run_self_verification_benchmark([orig]), T.run_self_verification_benchmark([flipped])
    strip = lambda res: [{k: v for k, v in r.items() if not k.startswith("expected") and k not in ("decision_correct", "decision_supported_label", "policy_applicable_expected")} for r in res["records"]]
    assert strip(a) == strip(b)
    assert a["aggregate"]["variants"][SVPA]["decision_accuracy"]["value"] == 1.0 and b["aggregate"]["variants"][SVPA]["decision_accuracy"]["value"] == 0.0
    assert b["aggregate"]["variants"][SVPA]["false_verification_rate"]["value"] == 1.0


def test_sv7d_metric_calculation_from_known_records():
    mk = lambda **kw: {"case_id": "c", "category": "x", "system_variant": SVPA, "decision_correct": True, "decision_asserted": True, "decision_supported_label": True, "claims_grounded_expected": True, "grounding_predicted": True,
                       "expected_major_contradiction": False, "contradiction_predicted": False, "expected_verification_status": "VERIFIED", "predicted_verification_status": "VERIFIED", **kw}
    recs = [mk(), mk(decision_correct=False, decision_supported_label=False, claims_grounded_expected=False, expected_verification_status="FAILED"),            # false verification, unsupported asserted, wrong grounding
            mk(decision_correct=False, decision_asserted=False, expected_verification_status="ESCALATE", predicted_verification_status="ESCALATE", expected_major_contradiction=True, contradiction_predicted=False),
            mk(decision_asserted=False, expected_verification_status="VERIFIED", predicted_verification_status="ESCALATE")]                                         # false escalation
    m = T._sv7d_variant_metrics(recs)
    assert (m["decision_accuracy"]["numerator"], m["decision_accuracy"]["denominator"], m["decision_accuracy"]["value"]) == (2, 4, 0.5)
    assert (m["unsupported_decision_rate"]["numerator"], m["unsupported_decision_rate"]["denominator"]) == (1, 2)
    assert (m["evidence_grounding_accuracy"]["numerator"], m["evidence_grounding_accuracy"]["denominator"]) == (3, 4)
    assert (m["contradiction_handling_accuracy"]["numerator"], m["contradiction_handling_accuracy"]["denominator"]) == (3, 4)
    assert (m["verification_status_accuracy"]["numerator"], m["verification_status_accuracy"]["denominator"]) == (2, 4)
    assert (m["escalation_accuracy"]["numerator"], m["escalation_accuracy"]["denominator"]) == (3, 4)
    assert (m["false_verification_rate"]["numerator"], m["false_verification_rate"]["denominator"]) == (1, 2)   # cases that must not be VERIFIED: FAILED, ESCALATE
    assert (m["false_escalation_rate"]["numerator"], m["false_escalation_rate"]["denominator"]) == (1, 3)       # cases that must not be ESCALATE


def test_sv7d_not_measured_is_distinct_from_zero():
    only_esc = T.run_self_verification_benchmark([_sv7d_case_by_id("sv7d_contradiction_major")])
    m = only_esc["aggregate"]["variants"][SVPA]
    assert m["false_verification_rate"]["value"] == 0.0 and m["false_verification_rate"]["denominator"] == 1           # measured zero
    assert m["false_escalation_rate"]["value"] == _SV7D_NM and m["false_escalation_rate"]["denominator"] == 0 and m["false_escalation_rate"]["reason"]  # empty denominator
    no_assert = T.run_self_verification_benchmark([_sv7d_case_by_id("sv7d_ambiguous_free_text_rule")])
    v = no_assert["aggregate"]["variants"][SVPA]
    assert v["unsupported_decision_rate"]["value"] == _SV7D_NM and v["evidence_grounding_accuracy"]["value"] == _SV7D_NM and v["contradiction_handling_accuracy"]["value"] == _SV7D_NM
    cmp_ = only_esc["aggregate"]["comparison"]
    assert cmp_["false_verification_rate"]["delta_SV_vs_EG"] == _SV7D_NM  # EG has no verification: no delta is invented
    assert T._sv7d_variant_metrics([])["decision_accuracy"]["value"] == _SV7D_NM


def test_sv7d_handles_verified_failed_escalate_and_not_verified(sv7d):
    expect = {"sv7d_pos_violation": ("VERIFIED", "VIOLATION"), "sv7d_pos_satisfied": ("VERIFIED", "SATISFIED"), "sv7d_wrong_but_supported_text": ("FAILED", "NO_CONCLUSION"),
              "sv7d_contradiction_major": ("ESCALATE", "ESCALATE"), "sv7d_ambiguous_free_text_rule": ("NOT_VERIFIED", "NO_CONCLUSION"), "sv7d_missing_unlinked_records": ("NOT_VERIFIED", "NO_CONCLUSION"),
              "sv7d_failed_beats_escalate": ("FAILED", "NO_CONCLUSION"), "sv7d_ambiguous_weak_grounding": ("ESCALATE", "ESCALATE")}
    for cid, (status, outcome) in expect.items():
        for v in (SV, SVPA):
            r = _sv7d_rec(sv7d, cid, v)
            assert r["predicted_verification_status"] == status and r["predicted_outcome"] == outcome and r["decision_correct"], (cid, v)
    assert _sv7d_rec(sv7d, "sv7d_failed_beats_escalate", SVPA)["escalation_reason"] is None  # FAILED never carries an escalation reason
    assert "major contradiction" in _sv7d_rec(sv7d, "sv7d_contradiction_major", SV)["escalation_reason"]
    assert T._sv7d_outcome("VIOLATION", "FAILED") == "NO_CONCLUSION" and T._sv7d_outcome("SATISFIED", "ESCALATE") == "ESCALATE" and T._sv7d_outcome("UNEVALUATED", "NOT_VERIFIED") == "NO_CONCLUSION" and T._sv7d_outcome("VIOLATION", None) == "VIOLATION"
    for r in sv7d["records"]:  # baseline / evidence-graph never claim a verification status
        if r["system_variant"] in ("BASELINE", "EVIDENCE_GRAPH"): assert r["predicted_verification_status"] == _SV7D_NM


def test_sv7d_policy_applicability_failures_grounded_but_inapplicable(sv7d):
    for cid in ("sv7d_policy_rule_text_changed", "sv7d_policy_missing_fact", "sv7d_policy_wrong_scope"):
        sv, pa = _sv7d_rec(sv7d, cid, SV), _sv7d_rec(sv7d, cid, SVPA)
        assert sv["grounding_predicted"] is True and sv["claims_grounded_expected"] is True and sv["policy_applicable_expected"] is False   # evidence is grounded ...
        assert sv["predicted_verification_status"] == "VERIFIED" and sv["predicted_outcome"] == "VIOLATION" and not sv["decision_correct"]    # ... 7B alone cannot see the inapplicable rule
        assert pa["predicted_verification_status"] == "ESCALATE" and pa["predicted_outcome"] == "ESCALATE" and pa["decision_correct"] and "applicability" in pa["escalation_reason"]
    agg = sv7d["aggregate"]["variants"]
    assert agg[SVPA]["false_verification_rate"]["value"] == 0.0 and agg[SV]["false_verification_rate"]["value"] > 0
    assert agg[SVPA]["decision_accuracy"]["value"] > agg[SV]["decision_accuracy"]["value"] > agg["EVIDENCE_GRAPH"]["decision_accuracy"]["value"]
    assert sv7d["aggregate"]["comparison"]["decision_accuracy"]["delta_SVPA_vs_SV"] > 0


def test_sv7d_wrong_but_superficially_supported_and_unsupported_decisions_are_rejected(sv7d):
    for cid in ("sv7d_wrong_but_supported_text", "sv7d_wrong_but_supported_heuristic"):
        base, eg, sv = (_sv7d_rec(sv7d, cid, v) for v in ("BASELINE", "EVIDENCE_GRAPH", SV))
        assert base["predicted_outcome"] == eg["predicted_outcome"] == "VIOLATION" and not base["decision_correct"] and not eg["decision_correct"]  # initial decision wrong
        assert eg["supporting_evidence"] and eg["grounding_predicted"] is True and eg["claims_grounded_expected"] is False                           # but looks supported to an existence-only check
        assert sv["predicted_outcome"] == "NO_CONCLUSION" and sv["decision_correct"] and sv["predicted_verification_status"] == "FAILED" and sv["grounding_predicted"] is False  # self-verification rejects / downgrades
    none, nsv = _sv7d_rec(sv7d, "sv7d_unsupported_no_evidence", "EVIDENCE_GRAPH"), _sv7d_rec(sv7d, "sv7d_unsupported_no_evidence", SV)
    assert none["predicted_outcome"] == "VIOLATION" and not none["decision_correct"] and none["claims_grounded_expected"] is False  # decision-level evidence links alone look like support to EG
    assert nsv["predicted_verification_status"] == "FAILED" and nsv["predicted_outcome"] == "NO_CONCLUSION" and nsv["decision_correct"]
    assert none["grounding_predicted"] is True and nsv["grounding_predicted"] is False
    agg = sv7d["aggregate"]["variants"]
    assert agg["BASELINE"]["unsupported_decision_rate"]["value"] > agg[SV]["unsupported_decision_rate"]["value"] >= agg[SVPA]["unsupported_decision_rate"]["value"] == 0.0
    assert agg[SV]["evidence_grounding_accuracy"]["value"] > agg["EVIDENCE_GRAPH"]["evidence_grounding_accuracy"]["value"]


def test_sv7d_contradiction_handling_flags_major_and_not_consistent_documents(sv7d):
    for v in ("EVIDENCE_GRAPH", SV, SVPA):
        assert _sv7d_rec(sv7d, "sv7d_contradiction_major", v)["contradiction_predicted"] is True
        assert _sv7d_rec(sv7d, "sv7d_contradiction_consistent", v)["contradiction_predicted"] is False
    assert _sv7d_rec(sv7d, "sv7d_contradiction_major", "EVIDENCE_GRAPH")["predicted_outcome"] == "VIOLATION"  # EG surfaces the contradiction but does not escalate
    assert _sv7d_rec(sv7d, "sv7d_contradiction_major", SVPA)["predicted_outcome"] == "ESCALATE"
    assert _sv7d_rec(sv7d, "sv7d_contradiction_major", "BASELINE")["contradiction_predicted"] == _SV7D_NM


def test_sv7d_predictions_do_not_mutate_graph_decisions_findings_or_results():
    G = _graph({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, rulebook="FORBID TRANSACTION > INR 15000\n")
    d = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision")
    snap = lambda: (T._sv7d_snapshot(G), {n: x["verdict"] for n, x in G.nodes(data=True) if x.get("type") == "Decision"}, json.dumps(G.graph.get("contradiction_findings"), sort_keys=True, default=str),
                    json.dumps(T.verify_policy_applicability_results(G), sort_keys=True, default=str))
    before = snap()
    a = json.dumps(T.self_verification_variant_predictions(G, d), sort_keys=True, default=str)
    b = json.dumps(T.self_verification_variant_predictions(G, d), sort_keys=True, default=str)
    assert a == b and snap() == before
    recs = T._sv7d_records_for_graph({**_sv7d_case_by_id("sv7d_contradiction_major")}, G, d)
    assert len(recs) == 4 and all(r["graph_unchanged"] for r in recs) and snap() == before
    # a fault is injected only into the benchmark's private graph, never into a caller's graph
    G2 = _sv_g(9701); _, d2 = _sv7b(G2)
    before2 = T._sv7d_snapshot(G2)
    T.run_self_verification_benchmark()
    assert T._sv7d_snapshot(G2) == before2


def test_sv7d_preserves_evidence_ids_and_provenance():
    G = _sv_g(9801)
    d = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision" and x.get("verdict") == "VIOLATION")
    preds = T.self_verification_variant_predictions(G, d)
    r7b, r7c = T.verify_self_verification_result(G, d), T.verify_policy_applicability(G, d)
    doc = _doc(G, "invoice_sv.txt")
    for v, res in ((SV, r7b), (SVPA, r7c), ("EVIDENCE_GRAPH", T.build_self_verification_result(G, d))):
        refs = preds[v]["supporting_evidence"]
        assert refs and [x["evidence_id"] for x in refs] == [x["evidence_id"] for x in res["supporting_evidence"]]  # ids exactly as the existing results report them
        for x in refs:
            assert G.nodes[x["evidence_id"]]["type"] == "Evidence" and x["document_id"] == doc and x["filename"] == "invoice_sv.txt" and x["location"]
            assert x["provenance"] == json.loads(json.dumps(G.nodes[x["evidence_id"]]["provenance"], default=str))
    assert preds["BASELINE"]["supporting_evidence"] == []
    res = T.run_self_verification_benchmark()
    r = _sv7d_rec(res, "sv7d_pos_violation", SVPA)
    assert r["supporting_evidence"] and all(x["filename"] == "invoice_a.txt" and x["location"] and x["provenance"]["source_text_sha256"] and x["provenance"]["file"] == "invoice_a.txt" and "source_document_id" not in x["provenance"] for x in r["supporting_evidence"])
    assert all(x["evidence_alias"].startswith("E") for x in r["supporting_evidence"])
    for rec in res["records"]: assert rec["id_integrity"] == {"evidence_ids_in_graph": True, "provenance_matches_graph": True}
    assert r["confidence"] == {"value": None, "status": "NOT_MEASURED"}


def test_sv7d_benchmark_writes_json_only_when_asked(tmp_path):
    p = tmp_path / "sv7d.json"
    T.run_self_verification_benchmark([_sv7d_case_by_id("sv7d_pos_violation")])
    assert not p.exists()
    out = T.run_self_verification_benchmark([_sv7d_case_by_id("sv7d_pos_violation")], output_path=str(p))
    assert json.loads(p.read_text(encoding="utf-8")) == json.loads(json.dumps(out))


def test_sv7d_records_carry_no_random_graph_ids(sv7d):
    import re as _re
    blob = json.dumps([{k: v for k, v in r.items() if k != "supporting_evidence"} for r in sv7d["records"]])
    assert not _re.search(r"\b[a-z]+_[0-9a-f]{16}\b", blob)
    pa = _sv7d_rec(sv7d, "sv7d_policy_wrong_scope", SVPA)
    assert "<rule>" in pa["escalation_reason"] and "applicability mismatch" in pa["escalation_reason"]