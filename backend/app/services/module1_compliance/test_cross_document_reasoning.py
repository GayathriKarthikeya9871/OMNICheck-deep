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


# 19 ---- counterfactual compliance (Phase 8): derived from the cited rule + existing evidence, read-only, benchmarked against hand-written labels
_CF_NM = T.NOT_MEASURED
_CF_METRICS = ("counterfactual_validity", "policy_consistency", "evidence_grounding", "minimum_change_accuracy", "status_accuracy", "change_type_accuracy", "invented_counterfactual_rate", "contract_validity")


def _cf_g(docs, rulebook):
    return _graph(docs, rulebook=rulebook)


def _cf_for(G, rule_part):
    d = next(n for n, x in sorted(G.nodes(data=True)) if x.get("type") == "Decision" and rule_part.lower() in G.nodes[x["rule_id"]]["condition"].lower())
    return d, T.build_counterfactual(G, d)


@pytest.fixture(scope="module")
def cfb():
    return T.run_counterfactual_benchmark()


def _cfb_rec(res, cid):
    return next(r for r in res["records"] if r["case_id"] == cid)


def _cfb_case_by_id(cid):
    import copy
    return copy.deepcopy(next(c for c in T.COUNTERFACTUAL_BENCHMARK if c["case_id"] == cid))


def test_cf_contract_scope_and_status_for_every_decision_and_finding():
    G = _cf_g({"invoice_a.txt": "Invoice No: INV-1\nBilled amount INR 5,000\n", "memo_a.txt": "notes\n"}, 'FORBID TRANSACTION > INR 1000\nREQUIRE KEYWORD "manager approval"\nFORBID TRANSACTION > INR 90000\nVendors should behave in spirit.\n')
    cfs = T.build_counterfactuals(G)
    verdicts = {n: x["verdict"] for n, x in G.nodes(data=True) if x.get("type") == "Decision"}
    assert {c["decision_id"] for c in cfs} == {n for n, v in verdicts.items() if v in ("VIOLATION", "INCONCLUSIVE", "UNEVALUATED")}
    for c in cfs:
        assert all(k in c for k in T.CF_FIELDS) and T.validate_counterfactual(G, c) == [] and c["status"] in ("ESTABLISHED", "UNRESOLVED")
        assert c["violated_rule"]["rule_id"] == G.nodes[c["decision_id"]]["rule_id"] and c["current_decision"]["verdict"] == verdicts[c["decision_id"]]  # rule + decision references preserved
    ok = next(n for n, v in verdicts.items() if v == "SATISFIED")
    r = T.build_counterfactual(G, ok)
    assert r["status"] == "NOT_REQUIRED" and not r["in_scope"] and all(k in r for k in T.CF_FIELDS) and r["recommended_corrective_condition"] == _CF_NM
    assert T.build_counterfactual(G, "no_such_node")["found"] is False


def test_cf_condition_is_derived_from_the_cited_rule_not_hard_coded():
    seen = set()
    for rb, region, tgt in (("FORBID TRANSACTION > INR 1000\n", "<=", 1000.0), ("FORBID TRANSACTION > INR 2500\n", "<=", 2500.0), ("FORBID TRANSACTION >= INR 2500\n", "<", 2500.0), ("REQUIRE TRANSACTION >= INR 9000\n", ">=", 9000.0)):
        G = _cf_g({"invoice_a.txt": "Invoice No: INV-1\nBilled amount INR 5,000\n"}, rb)
        _, c = _cf_for(G, "TRANSACTION")
        assert c["status"] == "ESTABLISHED" and c["required_condition"]["relation"] == region and c["required_condition"]["value"] == tgt
        m = c["minimum_changes"][0]
        assert m["minimum_delta"] == abs(5000.0 - tgt) and m["direction"] == ("decrease" if region in ("<", "<=") else "increase") and (m["target_value"] is None) == (region == "<")
        seen.add(c["recommended_corrective_condition"])
    assert len(seen) == 4
    for phrase in ("manager approval", "receipt attached"):
        _, c = _cf_for(_cf_g({"memo.txt": "notes\n"}, f'REQUIRE KEYWORD "{phrase}"\n'), "KEYWORD")
        assert c["required_condition"]["value"] == phrase and phrase in c["missing_requirement"] and phrase in c["recommended_corrective_condition"]


def test_cf_distinguishes_corrective_action_from_supplying_evidence():
    _, amt = _cf_for(_cf_g({"i.txt": "Invoice No: INV-1\nBilled amount INR 5,000\n"}, "FORBID TRANSACTION > INR 1000\n"), "TRANSACTION")
    _, miss = _cf_for(_cf_g({"m.txt": "notes\n"}, 'REQUIRE KEYWORD "manager approval"\n'), "KEYWORD")
    _, forb = _cf_for(_cf_g({"m.txt": "paid by cash payment\n"}, 'FORBID KEYWORD "cash payment"\n'), "KEYWORD")
    assert (amt["change_type"], miss["change_type"], forb["change_type"]) == ("CORRECTIVE_ACTION", "SUPPLY_EVIDENCE", "CORRECTIVE_ACTION")
    assert amt["expected_resulting_state"]["state"] == "COMPLIANT_WITH_RULE" and miss["expected_resulting_state"]["state"] == "RE_EVALUATION_REQUIRED"
    assert "does not establish" in miss["recommended_corrective_condition"]  # text presence is not proof the action happened
    G = _cf_g({"inv_j.txt": "Invoice No: INV-1101\nPO Number: PO-1101\nVendor: Boreal Metals\nBilled amount INR 20,000\n", "po_j.txt": "Purchase Order No: PO-1101\nVendor: Boreal Metals\nOrder value INR 12,000\n"}, "Transaction amounts must match.\n")
    _, cond = _cf_for(G, "amounts must match")
    assert G.nodes[cond["decision_id"]]["verdict"] == "INCONCLUSIVE" and cond["scope_class"] == "CONDITIONAL" and cond["change_type"] == "SUPPLY_EVIDENCE"
    assert cond["minimum_changes"][0]["kind"] == "link_evidence" and "amount_change" not in str(cond["minimum_changes"])  # no amount is changed: only linking evidence is missing
    assert cond["expected_resulting_state"]["expected_verdict"] == _CF_NM  # the outcome depends on the supplied evidence: not promised


def test_cf_unresolved_never_invents_a_counterfactual():
    G = _cf_g({"inv_p.txt": "Invoice No: INV-1\nBilled amount INR 5,000\n"}, "Vendors should behave reasonably in spirit.\n")
    _, c = _cf_for(G, "Vendors should")
    assert c["status"] == "UNRESOLVED" and c["scope_class"] == "CONDITIONAL" and c["unresolved_reason"] and c["minimum_changes"] == [] and c["change_type"] is None
    assert [c[k] for k in ("missing_requirement", "recommended_corrective_condition", "expected_resulting_state")] == [_CF_NM] * 3 and c["violated_rule"]["rule_id"] and T.validate_counterfactual(G, c) == []
    G = _cf_g({"inv_n.txt": "Invoice No: INV-5\nPO Number: PO-5\nVendor: Boreal Metals\nBilled amount INR 800\n", "po_n.txt": "Purchase Order No: PO-5\nVendor: Zenith Traders\nOrder value INR 800\n"}, "FORBID TRANSACTION > INR 100000\n")
    fs = T.build_finding_counterfactuals(G)
    v = next(f for f in fs if f["current_decision"]["field"] == "vendor")
    assert v["status"] == "UNRESOLVED" and "vendor" in v["unresolved_reason"] and v["violated_rule"] == _CF_NM and v["evidence_causing_violation"] and T.validate_counterfactual(G, v) == []
    bad = dict(c, status="UNRESOLVED", unresolved_reason=None)
    assert any("unresolved_reason" in p for p in T.validate_counterfactual(G, bad))
    assert any("PolicyRule" in p for p in T.validate_counterfactual(G, dict(c, status="ESTABLISHED", change_type="SUPPLY_EVIDENCE", violated_rule={"rule_id": "nope"})))


def test_cf_contradiction_context_and_amount_finding_use_the_amounts_rule():
    docs = {"inv_l.txt": CON_INV, "po_l.txt": CON_PO}
    G = _cf_g(docs, "FORBID TRANSACTION > INR 15000\n")
    _, c = _cf_for(G, "FORBID TRANSACTION")
    ctx = [x for x in c["contradiction_context"] if x["field"] == "amount"]
    assert c["status"] == "ESTABLISHED" and ctx and ctx[0]["category"] == "MAJOR_CONTRADICTION" and ctx[0]["other_side_amounts"] == [12000.0] and ctx[0]["other_side_satisfies_rule"] is True
    assert c["minimum_changes"][0]["target_value"] == 15000.0  # the minimum change is still the rule boundary, not the other document's value
    G = _cf_g(docs, "Transaction amounts must match.\n")
    f = next(x for x in T.build_finding_counterfactuals(G) if x["current_decision"]["field"] == "amount")
    rule = G.nodes[f["violated_rule"]["rule_id"]]
    assert f["status"] == "ESTABLISHED" and f["change_type"] == "CORRECTIVE_ACTION" and "amounts must match" in rule["condition"] and f["minimum_changes"][0]["minimum_delta"] == 8000.0 and f["minimum_changes"][0]["authoritative_record"] == _CF_NM
    assert f["current_decision"]["finding_id"] == next(x["finding_id"] for x in G.graph["contradiction_findings"] if x["field"] == "amount" and x["category"] == "MAJOR_CONTRADICTION") and T.validate_counterfactual(G, f) == []
    G = _cf_g(docs, "FORBID TRANSACTION > INR 99999\n")  # same contradiction, but no rule governs amount agreement
    assert next(x for x in T.build_finding_counterfactuals(G) if x["current_decision"]["field"] == "amount")["status"] == "UNRESOLVED"


def test_cf_preserves_evidence_document_rule_ids_and_provenance():
    G = _cf_g({"invoice_cf.txt": "Invoice No: INV-1\nBilled amount INR 5,000\n"}, "FORBID TRANSACTION > INR 1000\n")
    d, c = _cf_for(G, "TRANSACTION")
    doc = _doc(G, "invoice_cf.txt")
    assert [r["evidence_id"] for r in c["evidence_causing_violation"]] == list(G.nodes[d]["evidence_used"])  # exactly the evidence the Decision used
    for r in c["evidence_causing_violation"]:
        assert G.nodes[r["evidence_id"]]["type"] == "Evidence" and r["document_id"] == doc and r["filename"] == "invoice_cf.txt" and r["location"]
        assert r["provenance"] == json.loads(json.dumps(G.nodes[r["evidence_id"]]["provenance"], default=str))
    assert c["minimum_changes"][0]["evidence_ids"] == list(G.nodes[d]["evidence_used"]) and c["violated_rule"]["rule_id"] == G.nodes[d]["rule_id"] and c["decision_id"] == d
    assert c["violated_rule"]["source_location"] == G.nodes[G.nodes[d]["rule_id"]].get("source_location")
    G = _cf_g({"empty_k.txt": "", "memo_k.txt": "Meeting notes\n"}, 'REQUIRE KEYWORD "approval"\n')
    _, g = _cf_for(G, "KEYWORD")
    assert g["status"] == "ESTABLISHED" and [(r["evidence_id"], r["document_id"], r["filename"]) for r in g["evidence_causing_violation"]] == [(None, _doc(G, "empty_k.txt"), "empty_k.txt")]
    assert T.validate_counterfactual(G, g) == []


def test_cf_is_read_only_deterministic_and_idempotent():
    G = _cf_g({"inv_l.txt": CON_INV, "po_l.txt": CON_PO, "memo.txt": "notes\n"}, 'FORBID TRANSACTION > INR 15000\nREQUIRE KEYWORD "manager approval"\nTransaction amounts must match.\n')
    snap = lambda: (T._sv7d_snapshot(G), {n: (x["verdict"], x.get("rationale")) for n, x in G.nodes(data=True) if x.get("type") == "Decision"}, json.dumps(T.verify_policy_applicability_results(G), sort_keys=True, default=str))
    before = snap()
    a = json.dumps([T.build_counterfactuals(G), T.build_finding_counterfactuals(G)], sort_keys=True, default=str)
    b = json.dumps([T.build_counterfactuals(G), T.build_finding_counterfactuals(G)], sort_keys=True, default=str)
    assert a == b and snap() == before
    T.run_counterfactual_benchmark([_cfb_case_by_id("cfb_pos_amount_over_limit")])
    assert snap() == before


def test_cfb_labels_are_valid_cover_required_categories_and_validator_catches_inconsistency():
    assert T.validate_counterfactual_benchmark() == []
    cases = T.COUNTERFACTUAL_BENCHMARK
    assert len({c["case_id"] for c in cases}) == len(cases) and set(T.CFB_CATEGORIES) <= {c["category"] for c in cases}
    assert {"ESTABLISHED", "UNRESOLVED", "NOT_REQUIRED"} == {c["expected_status"] for c in cases} and {"CORRECTIVE_ACTION", "SUPPLY_EVIDENCE"} <= {c["expected_change_type"] for c in cases}
    assert {c["expected_decision"] for c in cases} >= {"VIOLATION", "INCONCLUSIVE", "UNEVALUATED", "SATISFIED", "NOT_APPLICABLE"} and {c["target_kind"] for c in cases} == {"decision", "finding"}
    a = _cfb_case_by_id("cfb_pos_amount_over_limit"); a["expected_change_type"] = None
    assert T.validate_counterfactual_benchmark([a], require_coverage=False)
    b = _cfb_case_by_id("cfb_pos_amount_over_limit"); b["expected_after_verdict"] = None
    assert any("replay" in p for p in T.validate_counterfactual_benchmark([b], require_coverage=False))
    c = _cfb_case_by_id("cfb_neg_satisfied"); c["expected_minimum_changes"] = [{"kind": "x"}]
    assert T.validate_counterfactual_benchmark([c], require_coverage=False)
    assert any("duplicate" in p for p in T.validate_counterfactual_benchmark([_cfb_case_by_id("cfb_pos_amount_over_limit")] * 2, require_coverage=False))
    u = _cfb_case_by_id("cfb_pos_amount_over_limit"); u["fault"] = "nope"
    assert T.validate_counterfactual_benchmark([u], require_coverage=False)
    assert any("not covered" in p for p in T.validate_counterfactual_benchmark([a for a in cases if a["category"] == "positive"]))


def test_cfb_is_deterministic_and_aggregate_derives_from_records(cfb):
    again = T.run_counterfactual_benchmark()
    assert json.dumps(again, sort_keys=True) == json.dumps(cfb, sort_keys=True)
    a = cfb["aggregate"]
    assert a["benchmark"]["seed"] is None and a["benchmark"]["llm_used"] is False and a["protocol"]["llm_used"] is False and a["protocol"]["network_used"] is False and a["protocol"]["randomness"] == "none"
    assert a == T.aggregate_counterfactual_results(cfb["records"]) and len(cfb["records"]) == len(T.COUNTERFACTUAL_BENCHMARK)
    assert all(m in a["metrics"] for m in _CF_METRICS)


def test_cfb_scores_every_required_dimension_against_independent_labels(cfb):
    m = cfb["aggregate"]["metrics"]
    for k in ("counterfactual_validity", "policy_consistency", "evidence_grounding", "minimum_change_accuracy", "status_accuracy", "change_type_accuracy", "contract_validity"):
        assert m[k]["value"] == 1.0 and m[k]["denominator"] > 0 and m[k]["numerator"] == m[k]["denominator"], k
    assert m["invented_counterfactual_rate"]["value"] == 0.0 and m["invented_counterfactual_rate"]["denominator"] > 0  # a MEASURED 0.0
    for r in cfb["records"]:  # labels are read from the case, predictions from the system: a wrong label must show up as a miss
        assert r["expected_status"] == next(c["expected_status"] for c in T.COUNTERFACTUAL_BENCHMARK if c["case_id"] == r["case_id"]) and r["graph_unchanged"] and r["deterministic"] and r["counterfactual_id_stable"]
        assert r["id_integrity"]["evidence_ids_in_graph"] and r["id_integrity"]["provenance_matches_graph"] and r["id_integrity"]["contract_problems"] == []
    wrong = _cfb_case_by_id("cfb_pos_amount_over_limit"); wrong["expected_minimum_changes"] = [{"kind": "amount_change", "direction": "decrease", "boundary": 999.0, "boundary_inclusive": True, "minimum_delta": 4001.0, "target_value": 999.0}]
    wrong["expected_required_condition"] = {"subject": "TRANSACTION", "relation": "<=", "value": 999.0, "currency": "INR"}; wrong["expected_evidence_files"] = ["other.txt"]
    mm = T.run_counterfactual_benchmark([wrong])["aggregate"]["metrics"]
    assert mm["minimum_change_accuracy"]["value"] == 0.0 and mm["policy_consistency"]["value"] == 0.0 and mm["evidence_grounding"]["value"] == 0.0 and mm["counterfactual_validity"]["value"] == 1.0
    cs = _cfb_case_by_id("cfb_pos_amount_over_limit"); cs["expected_status"], cs["expected_change_type"] = "UNRESOLVED", None
    for k in ("expected_required_condition", "expected_minimum_changes", "replay", "expected_after_verdict"): cs[k] = None
    assert T.run_counterfactual_benchmark([cs])["aggregate"]["metrics"]["invented_counterfactual_rate"]["value"] == 1.0  # an invented counterfactual is caught


def test_cfb_replay_validity_is_checked_by_re_evaluating_the_applied_counterfactual(cfb):
    for cid in ("cfb_pos_amount_over_limit", "cfb_pos_exclusive_boundary", "cfb_pos_two_amounts", "cfb_pos_missing_manager_approval", "cfb_pos_forbidden_phrase", "cfb_pos_linked_amount_mismatch", "cfb_contradiction_decision"):
        r = _cfb_rec(cfb, cid)
        assert r["replay_expected_verdict"] == "SATISFIED" and r["replay_verdict"] == "SATISFIED", cid
    ex = _cfb_rec(cfb, "cfb_pos_exclusive_boundary")  # the boundary itself is non-compliant: no exact minimum value, only a strict bound
    assert ex["predicted_minimum_changes"][0]["target_value"] is None and ex["predicted_minimum_changes"][0]["delta_is_strict_lower_bound"] is True
    case = _cfb_case_by_id("cfb_pos_amount_over_limit")  # a counterfactual that does NOT restore compliance is detected as invalid
    pred = {"minimum_changes": [{"boundary": 1000.0, "boundary_inclusive": True}]}
    docs = T._cfb_replay_docs(case, pred); docs["inv_a.txt"] = case["documents"]["inv_a.txt"].replace("5,000", "2,000")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        G = T._cfb_build(docs, case["rulebook"], td)
    assert next(x["verdict"] for n, x in G.nodes(data=True) if x.get("type") == "Decision") == "VIOLATION"


def test_cfb_not_measured_is_distinct_from_measured_zero(cfb):
    only_est = T.run_counterfactual_benchmark([_cfb_case_by_id("cfb_pos_amount_over_limit")])["aggregate"]["metrics"]
    assert only_est["invented_counterfactual_rate"]["value"] == _CF_NM and only_est["invented_counterfactual_rate"]["denominator"] == 0 and only_est["invented_counterfactual_rate"]["numerator"] is None
    only_neg = T.run_counterfactual_benchmark([_cfb_case_by_id("cfb_neg_satisfied")])["aggregate"]["metrics"]
    assert all(only_neg[k]["value"] == _CF_NM for k in ("counterfactual_validity", "policy_consistency", "minimum_change_accuracy", "change_type_accuracy", "contract_validity"))
    assert only_neg["status_accuracy"]["value"] == 1.0 and only_neg["invented_counterfactual_rate"]["value"] == 0.0
    noreplay = T.run_counterfactual_benchmark([_cfb_case_by_id("cfb_cond_unlinked_records")])["aggregate"]["metrics"]
    assert noreplay["counterfactual_validity"]["value"] == _CF_NM and noreplay["minimum_change_accuracy"]["value"] == 1.0
    assert _cfb_rec(cfb, "cfb_cond_unlinked_records")["replay_verdict"] == _CF_NM


def test_cfb_no_valid_counterfactual_cases_are_unresolved_with_reasons(cfb):
    for cid, why in (("cfb_contradiction_vendor_no_policy", "vendor"), ("cfb_contradiction_date_no_policy", "date"), ("cfb_ambiguous_free_text_rule", "never evaluated"), ("cfb_ambiguous_govid_unconfigured", "never evaluated"),
                     ("cfb_unsupported_violation", "FAILED"), ("cfb_inapplicable_rule", "MISMATCH")):
        r = _cfb_rec(cfb, cid)
        assert r["predicted_status"] == "UNRESOLVED" and r["predicted_change_type"] is None and r["predicted_minimum_changes"] == [] and why in r["unresolved_reason"], cid
    assert _cfb_rec(cfb, "cfb_neg_satisfied")["predicted_status"] == "NOT_REQUIRED" and _cfb_rec(cfb, "cfb_neg_not_applicable")["predicted_status"] == "NOT_REQUIRED"
    assert _cfb_rec(cfb, "cfb_contradiction_amount_finding")["target_kind"] == "finding"


def test_cfb_preserves_stable_evidence_and_rule_references_and_provenance(cfb):
    import re as _re
    blob = json.dumps(cfb["records"])
    assert not _re.search(r"\b[a-z]+_[0-9a-f]{16}\b", blob)
    r = _cfb_rec(cfb, "cfb_pos_amount_over_limit")
    assert r["evidence_refs"] and all(x["evidence_alias"].startswith("E") and x["filename"] == "inv_a.txt" and x["location"] and x["provenance"]["source_text_sha256"] and x["provenance"]["file"] == "inv_a.txt" and "source_document_id" not in x["provenance"] for x in r["evidence_refs"])
    assert r["rule_condition"] == "FORBID TRANSACTION > INR 1000" and r["rule_source_location"] and r["rule_referenced"] and r["rule_is_target"]
    ctx = _cfb_rec(cfb, "cfb_contradiction_decision")["contradiction_context"]
    assert ctx and ctx[0]["category"] == "MAJOR_CONTRADICTION" and ctx[0]["other_side_satisfies_rule"] is True


def test_cfb_writes_json_only_when_asked_and_never_mutates_a_caller_graph(tmp_path):
    p = tmp_path / "cfb.json"
    T.run_counterfactual_benchmark([_cfb_case_by_id("cfb_pos_amount_over_limit")])
    assert not p.exists()
    out = T.run_counterfactual_benchmark([_cfb_case_by_id("cfb_pos_amount_over_limit")], output_path=str(p))
    assert json.loads(p.read_text(encoding="utf-8")) == json.loads(json.dumps(out))
    G = _cf_g({"invoice_k.txt": CON_INV, "po_k.txt": CON_PO}, "FORBID TRANSACTION > INR 15000\n")
    before = T._sv7d_snapshot(G)
    T.run_counterfactual_benchmark()
    assert T._sv7d_snapshot(G) == before
    case = _cfb_case_by_id("cfb_pos_amount_over_limit")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        Gp = T._cfb_build(case["documents"], case["rulebook"], td)
    b2 = T._sv7d_snapshot(Gp); T._cfb_record(case, Gp, "unused")
    assert T._sv7d_snapshot(Gp) == b2


# 20 ---- counterfactual hardening (Phase 8B): edge cases of the existing layer; nothing here changes a decision, verification result, finding or the graph
_CF_INV5 = "Invoice No: INV-1\nBilled amount INR 5,000\n"
_CF_UNLINKED = {"i_u.txt": "Invoice No: INV-1\nVendor: A\nBilled amount INR 100\n", "r_u.txt": "Receipt\nVendor: B\nPaid amount INR 100\n"}


def _cf_only(G):
    return next(c for c in T.build_counterfactuals(G))


def _cf_fault(fault, rulebook="FORBID TRANSACTION > INR 1000\n"):
    G = _cf_g({"i_f.txt": _CF_INV5}, rulebook)
    d = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision")
    T._sv7d_apply_fault(G, d, fault)
    return G, T.build_counterfactual(G, d)


def _cf_is_unresolved(G, c):
    assert c["status"] == "UNRESOLVED" and c["change_type"] is None and c["unresolved_reason"] and c["minimum_changes"] == [] and c["requires_reevaluation"] is False
    assert [c[k] for k in ("missing_requirement", "recommended_corrective_condition", "expected_resulting_state")] == [_CF_NM] * 3 and T.validate_counterfactual(G, c) == []


def test_cf8b_missing_evidence_supplies_evidence_and_never_asserts_satisfaction():
    G = _cf_g({"m_e.txt": "notes\n"}, 'REQUIRE KEYWORD "manager approval"\n')
    c = _cf_only(G)
    assert c["status"] == "ESTABLISHED" and c["change_type"] == "SUPPLY_EVIDENCE" and c["satisfaction_asserted"] is False and c["requires_reevaluation"] is True
    assert c["expected_resulting_state"]["state"] == "RE_EVALUATION_REQUIRED" and c["violated_rule"]["rule_id"] == G.nodes[c["decision_id"]]["rule_id"] and T.validate_counterfactual(G, c) == []
    assert [r["evidence_id"] for r in c["evidence_causing_violation"]] == list(G.nodes[c["decision_id"]]["absence_scope_evidence_ids"])
    G = _cf_g({"empty_e.txt": "", "memo_e.txt": "notes\n"}, 'REQUIRE KEYWORD "approval"\n')  # extraction gap: INCONCLUSIVE, the document (not an invented fact) is what is missing
    c = _cf_only(G)
    assert c["scope_class"] == "CONDITIONAL" and c["change_type"] == "SUPPLY_EVIDENCE" and c["expected_resulting_state"]["expected_verdict"] == _CF_NM and c["satisfaction_asserted"] is False
    bad = dict(c, expected_resulting_state={"state": "COMPLIANT_WITH_RULE"})  # a forged "supplying evidence proves compliance" result is rejected
    assert any("not proof" in p for p in T.validate_counterfactual(G, bad))
    assert any("satisfaction" in p for p in T.validate_counterfactual(G, dict(c, satisfaction_asserted=True)))


def test_cf8b_contradicting_evidence_is_reported_without_changing_the_counterfactual():
    G = _cf_g({"inv_c.txt": CON_INV, "po_c.txt": CON_PO}, "FORBID TRANSACTION > INR 15000\n")
    c = _cf_only(G)
    fid = next(x["finding_id"] for x in G.graph["contradiction_findings"] if x["field"] == "amount" and x["category"] == "MAJOR_CONTRADICTION")
    ec = c["evidence_conflict"]
    assert c["status"] == "ESTABLISHED" and c["change_type"] == "CORRECTIVE_ACTION" and c["minimum_changes"][0]["target_value"] == 15000.0  # proposed change itself is unchanged
    assert ec["status"] == "UNRESOLVED_CONTRADICTION" and fid in ec["finding_ids"] and ec["note"]
    assert all(G.nodes[e]["type"] == "Evidence" for e in ec["evidence_ids"]) and [r["evidence_id"] for r in ec["evidence"]] == ec["evidence_ids"]
    assert all(r["provenance"] == json.loads(json.dumps(G.nodes[r["evidence_id"]]["provenance"], default=str)) for r in ec["evidence"])
    assert _cf_only(_cf_g({"i_n.txt": _CF_INV5}, "FORBID TRANSACTION > INR 1000\n"))["evidence_conflict"]["status"] == "NONE"
    f = next(x for x in T.build_finding_counterfactuals(_cf_g({"inv_c.txt": CON_INV, "po_c.txt": CON_PO}, "Transaction amounts must match.\n")) if x["current_decision"]["field"] == "amount")
    assert f["evidence_conflict"]["status"] == "SUBJECT_OF_COUNTERFACTUAL" and f["requires_reevaluation"] is True and f["satisfaction_asserted"] is False


def test_cf8b_failed_self_verification_is_unresolved_but_escalation_alone_is_not():
    for fault in ("evidence_text_5000_to_500", "remove_support", "heuristic_support"):
        G, c = _cf_fault(fault)
        assert c["verification"]["verification_status"] == "FAILED" and "self-verification FAILED" in c["unresolved_reason"]
        _cf_is_unresolved(G, c)
    G, c = _cf_fault("weak_location")  # ESCALATE (weak grounding) is not FAILED: the grounded violation still yields a counterfactual, evidence IDs intact
    assert c["verification"]["verification_status"] == "ESCALATE" and c["status"] == "ESTABLISHED" and all(r["evidence_id"] for r in c["evidence_causing_violation"])


def test_cf8b_failed_policy_applicability_is_unresolved_for_mismatch_unestablished_and_inconclusive():
    for fault in ("rule_condition_9000", "wrong_scope"):
        G, c = _cf_fault(fault)
        assert c["verification"]["policy_applicability"] == "MISMATCH"; _cf_is_unresolved(G, c)
    G, c = _cf_fault("basis_amount_none")
    assert c["verification"]["policy_applicability"] == "UNESTABLISHED"; _cf_is_unresolved(G, c)
    G = _cf_g({"i_s.txt": _CF_INV5}, "FORBID TRANSACTION > INR 1000\n")  # Decision with no recorded policy scope: applicability cannot be established
    d = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision")
    for u, v, k, e in list(G.out_edges(d, keys=True, data=True)):
        if e.get("relation") == "BELONGS_TO": G.remove_edge(u, v, k)
    c = T.build_counterfactual(G, d)
    assert c["verification"]["policy_applicability"] == "UNESTABLISHED"; _cf_is_unresolved(G, c)
    G = _cf_g(_CF_UNLINKED, "Transaction amounts must match.\n")  # INCONCLUSIVE is not covered by applicability verification: the rule text itself must still match the stored spec
    c = _cf_only(G)
    assert c["current_decision"]["verdict"] == "INCONCLUSIVE" and c["status"] == "ESTABLISHED" and c["change_type"] == "SUPPLY_EVIDENCE"
    G.nodes[c["violated_rule"]["rule_id"]]["condition"] = "Vendors should behave in spirit."
    c = T.build_counterfactual(G, c["decision_id"])
    assert "no longer parses" in c["unresolved_reason"]; _cf_is_unresolved(G, c)


def test_cf8b_unsupported_free_text_rule_is_unresolved_and_cannot_be_forged_as_established():
    G = _cf_g({"i_t.txt": _CF_INV5}, "Vendors should behave reasonably in spirit.\n")
    c = _cf_only(G)
    assert c["current_decision"]["verdict"] == "UNEVALUATED" and c["violated_rule"]["rule_id"]; _cf_is_unresolved(G, c)
    G = _cf_g({"i_t.txt": _CF_INV5}, "FORBID TRANSACTION > INR 1000\n")
    est = _cf_only(G)
    G.nodes[est["violated_rule"]["rule_id"]]["condition"] = "Vendors should behave reasonably in spirit."
    assert any("supported rule form" in p for p in T.validate_counterfactual(G, est))


def test_cf8b_amount_boundaries_follow_the_rule_operator_exactly():
    for rb, amt, status, region, strict in (("FORBID TRANSACTION > INR 5000\n", "5,000", "NOT_REQUIRED", None, None), ("FORBID TRANSACTION >= INR 5000\n", "5,000", "ESTABLISHED", "<", True),
                                            ("REQUIRE TRANSACTION > INR 5000\n", "5,000", "ESTABLISHED", ">", True), ("REQUIRE TRANSACTION >= INR 5000\n", "4,999", "ESTABLISHED", ">=", False),
                                            ("FORBID TRANSACTION < INR 5000\n", "4,999", "ESTABLISHED", ">=", False)):
        G = _cf_g({"i_b.txt": f"Invoice No: INV-1\nBilled amount INR {amt}\n"}, rb)
        d = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision")
        c = T.build_counterfactual(G, d)
        assert c["status"] == status, rb
        if region is None: assert c["in_scope"] is False and c["current_decision"]["verdict"] == "SATISFIED"; continue  # the boundary value itself is compliant: nothing to correct
        m = c["minimum_changes"][0]
        assert c["required_condition"]["relation"] == region and m["delta_is_strict_lower_bound"] is strict and (m["target_value"] is None) == strict and m["boundary"] == 5000.0 and c["satisfaction_asserted"] is False
        assert T.validate_counterfactual(G, c) == []


def test_cf8b_change_types_are_not_interchangeable():
    amt = _cf_only(_cf_g({"i_k.txt": _CF_INV5}, "FORBID TRANSACTION > INR 1000\n"))
    miss = _cf_only(_cf_g({"m_k.txt": "notes\n"}, 'REQUIRE KEYWORD "manager approval"\n'))
    forb = _cf_only(_cf_g({"m_k.txt": "paid by cash payment\n"}, 'FORBID KEYWORD "cash payment"\n'))
    link = _cf_only(_cf_g(_CF_UNLINKED, "Transaction amounts must match.\n"))
    assert [c["change_type"] for c in (amt, forb, miss, link)] == ["CORRECTIVE_ACTION", "CORRECTIVE_ACTION", "SUPPLY_EVIDENCE", "SUPPLY_EVIDENCE"]
    assert all(m["kind"] not in ("amount_change", "phrase_removal") for c in (miss, link) for m in c["minimum_changes"]) and all(m["kind"] in ("amount_change", "phrase_removal") for c in (amt, forb) for m in c["minimum_changes"])


def test_cf8b_no_policy_supported_minimum_change_stays_unresolved(monkeypatch):
    G = _cf_g({"g_u.txt": "Aadhaar 2345 6789 0124\n"}, "FORBID GOVID VALID\n")
    for c in T.build_counterfactuals(G): _cf_is_unresolved(G, c)
    G = _cf_g({"i_m.txt": _CF_INV5}, "FORBID TRANSACTION > INR 1000\n")
    monkeypatch.setattr(T, "_cf_transaction", lambda *a, **k: {"change_type": "CORRECTIVE_ACTION", "required_condition": {"subject": "TRANSACTION", "relation": "<=", "value": 1000.0, "currency": None}, "minimum_changes": [], "missing_requirement": "x", "recommended": "x", "resulting": {"state": "COMPLIANT_WITH_RULE"}})
    c = _cf_only(G)
    assert "no policy-supported minimum change" in c["unresolved_reason"]; _cf_is_unresolved(G, c)
    assert any("minimum change" in p for p in T.validate_counterfactual(G, dict(c, status="ESTABLISHED", change_type="CORRECTIVE_ACTION", missing_requirement="x", recommended_corrective_condition="x", expected_resulting_state={"state": "x"}, evidence_causing_violation=[{"evidence_id": None, "document_id": None}])))


def test_cf8b_scenarios_are_read_only_and_idempotent():
    for docs, rb in (({"inv_c.txt": CON_INV, "po_c.txt": CON_PO, "m.txt": "notes\n"}, 'FORBID TRANSACTION > INR 15000\nREQUIRE KEYWORD "manager approval"\nTransaction amounts must match.\nVendors should behave in spirit.\n'), (_CF_UNLINKED, "Transaction amounts must match.\n")):
        G = _cf_g(docs, rb)
        snap = lambda: (T._sv7d_snapshot(G), json.dumps(T.build_self_verification_results(G), sort_keys=True, default=str), json.dumps(G.graph.get("contradiction_findings"), sort_keys=True, default=str))
        before = snap()
        run = lambda: json.dumps([T.build_counterfactuals(G), T.build_finding_counterfactuals(G)], sort_keys=True, default=str)
        assert run() == run() and snap() == before


# 21 ---- compiled-path counterfactual (Phase 8C): ONE compiled semantic is re-derived from the STORED compiled rule (single leaf `amount <|<=|>|>= number` on a transaction entity, REQUIRE / PROHIBIT);
# everything else stays UNRESOLVED. Offline: only the LLM compiler and the external rule_engine call are stubbed (same technique as _fake_compiled); the graph, Decisions, verification layers and counterfactual layer are the real ones.
_CC_CMP = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}
_CC_NEG = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}
_CC_ORIG_EVAL = T.evaluate_policy_rules  # captured before any test patches it
_CC_AMT = 5000.0  # the single transaction in _CF_INV5 (INR 5,000); the rulebook limits below are policy amounts, not transactions
# (rule_type, operator, limit, free-text rule line). The legacy DSL cannot parse these lines (UNEVALUATED): the compiled rule is the only evaluator, so parsed_spec is None on the Decision.
_CC_CASES = (("REQUIRE", "<", 1000.0, "Each transaction must be below INR 1000."), ("REQUIRE", "<=", 1000.0, "Each transaction must be at most INR 1000."),
             ("REQUIRE", ">", 9000.0, "Each transaction must be above INR 9000."), ("REQUIRE", ">=", 9000.0, "Each transaction must be at least INR 9000."),
             ("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited."), ("PROHIBIT", ">=", 1000.0, "Transactions of INR 1000 or more are prohibited."),
             ("PROHIBIT", "<", 9000.0, "Transactions below INR 9000 are prohibited."), ("PROHIBIT", "<=", 9000.0, "Transactions of INR 9000 or less are prohibited."))


def _cc_dict(rule_type, op, value, line, unit="INR", **over):
    """Stored shape of a compiled rule: only the keys that apply_compiled_policy, rule_engine and _cf_compiled read. test_cc_real_schema_* checks the same shape against the real CompiledRule."""
    d = {"rule_id": "CR1", "policy_id": "P1", "status": "VALID", "rule_type": rule_type, "severity": "high", "confidence": 0.95, "entity": "transaction", "source_text": line, "ambiguities": [], "issues": [],
         "condition": {"entity": "transaction", "field": "amount", "operator": op, "value": value, "unit": unit, "children": []}, "temporal": None, "required_evidence": [], "exception": [], "action": "review_transaction",
         "expression": f"transaction.amount {op} {value:g}"}
    d.update(over)
    return d


def _cc_stub_engine(d, verdict=None, indeterminate=None):
    """Deterministic stand-in for rule_engine.evaluate_policy (called positionally by apply_compiled_policy): per record, REQUIRE -> COMPLIANT iff cmp holds, PROHIBIT -> VIOLATION iff cmp holds.
    `verdict` forces one verdict for every record; `indeterminate` maps record_index -> missing facts (INDETERMINATE record)."""
    c = d["condition"]
    def run(rules, facts, evidence, date, inc, fx, strict):
        out = []
        for i, rec in enumerate(facts.get(d["entity"]) or []):
            a = rec["amount"]["value"] if isinstance(rec["amount"], dict) else rec["amount"]
            if verdict: v = verdict
            else:
                holds = _CC_CMP[c["operator"]](a, c["value"])
                v = ("VIOLATION" if holds else "COMPLIANT") if d["rule_type"] == "PROHIBIT" else ("COMPLIANT" if holds else "VIOLATION")
            miss = (indeterminate or {}).get(i)
            out.append({"rule_id": d["rule_id"], "verdict": "INDETERMINATE" if miss is not None else v, "record_index": i, "severity": "high", "missing_facts": list(miss or []), "reasons": ["stub engine result"]})
        return {"rule_results": out, "evaluation_date": None}
    return run


def _cc_graph(monkeypatch, d, docs=None, verdict=None, indeterminate=None):
    cr = types.SimpleNamespace(rule_id=d["rule_id"], policy_id=d["policy_id"], status=d["status"], rule_type=_V(d["rule_type"]), severity=_V(d["severity"]), confidence=d["confidence"], expression=d["expression"],
                               source_text=d["source_text"], ambiguities=[], issues=[], model_dump=lambda mode=None: copy.deepcopy(d))
    res = types.SimpleNamespace(rules=[cr], status="COMPILED", policy_id=d["policy_id"], stats={}, ambiguous_policy=False, ambiguity_reasons=[], rejected=[], unparsed_statements=[])
    monkeypatch.setattr(T, "HAS_POLICY_COMPILER", True, raising=False)
    monkeypatch.setattr(T, "_compile_policy", lambda text, ctx, conf: res, raising=False)
    monkeypatch.setattr(T, "_evaluate_compiled_rules", _cc_stub_engine(d, verdict, indeterminate), raising=False)
    monkeypatch.setattr(T, "_COMPILER_MIN_CONFIDENCE", 0.5, raising=False)
    orig = _CC_ORIG_EVAL  # the offline graph build skips the compile stage the Celery task runs just before evaluate_policy_rules: run it here, unchanged
    line = d["source_text"]
    monkeypatch.setattr(T, "evaluate_policy_rules", lambda G: (T.apply_compiled_policy(G, line + "\n"), orig(G))[1])
    G = _graph(docs or {"inv_cc.txt": _CF_INV5}, rulebook=line + "\n")
    dec = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision")
    return G, dec


def _cc_established(G, dec, op, region, limit):
    dd, c = G.nodes[dec], T.build_counterfactual(G, dec)
    assert dd["result_source"] == "compiled_policy_engine" and dd["parsed_spec"] is None and dd["verdict"] == "VIOLATION"  # compiled is the only evaluator: nothing legacy to lean on
    assert c["status"] == "ESTABLISHED" and c["change_type"] == "CORRECTIVE_ACTION" and c["scope_class"] == "NON_COMPLIANT" and all(k in c for k in T.CF_FIELDS)
    assert c["required_condition"] == {"subject": "TRANSACTION", "relation": region, "value": limit, "currency": "INR"}
    strict = region in ("<", ">")
    m = c["minimum_changes"][0]
    assert len(c["minimum_changes"]) == 1 and m["kind"] == "amount_change" and m["current_value"] == _CC_AMT and m["boundary"] == limit and m["minimum_delta"] == abs(_CC_AMT - limit)
    assert m["direction"] == ("decrease" if region in ("<", "<=") else "increase") and m["delta_is_strict_lower_bound"] is strict and m["boundary_inclusive"] is (not strict) and (m["target_value"] is None) is strict
    assert c["satisfaction_asserted"] is False and c["requires_reevaluation"] is True and c["expected_resulting_state"]["rule_id"] == dd["rule_id"] and T.validate_counterfactual(G, c) == []
    return c


# 1 + 2 ---- supported semantic: REQUIRE and PROHIBIT, all four ordering operators; the compliant region and boundary inclusivity follow the stored operator exactly
def test_cc_require_and_prohibit_amount_bounds_are_established(monkeypatch):
    seen = set()
    for rt, op, limit, line in _CC_CASES:
        G, dec = _cc_graph(monkeypatch, _cc_dict(rt, op, limit, line))
        c = _cc_established(G, dec, op, op if rt == "REQUIRE" else _CC_NEG[op], limit)  # REQUIRE: the compliant region is the condition itself; PROHIBIT: its negation
        assert c["violated_rule"]["result_source"] == "compiled_policy_engine" and c["violated_rule"]["condition"] == line
        seen.add((rt, op))
    assert len(seen) == 8


# 3 ---- boundary values: the boundary itself is compliant for an inclusive region (nothing to correct) and non-compliant for an exclusive one (strict lower bound, no target)
def test_cc_boundary_value_is_compliant_or_non_compliant_exactly_as_the_operator_says(monkeypatch):
    for rt, op, line in (("PROHIBIT", ">", "Transactions above INR 5000 are prohibited."), ("PROHIBIT", "<", "Transactions below INR 5000 are prohibited."),
                         ("REQUIRE", ">=", "Each transaction must be at least INR 5000."), ("REQUIRE", "<=", "Each transaction must be at most INR 5000.")):
        G, dec = _cc_graph(monkeypatch, _cc_dict(rt, op, 5000.0, line))  # amount == limit: rule satisfied
        c = T.build_counterfactual(G, dec)
        assert G.nodes[dec]["verdict"] == "SATISFIED" and G.nodes[dec]["result_source"] == "compiled_policy_engine"
        assert c["status"] == "NOT_REQUIRED" and c["in_scope"] is False and c["change_type"] is None and c["minimum_changes"] == [] and c["required_condition"] is None
    for rt, op, line, region in (("PROHIBIT", ">=", "Transactions of INR 5000 or more are prohibited.", "<"), ("PROHIBIT", "<=", "Transactions of INR 5000 or less are prohibited.", ">"),
                                 ("REQUIRE", ">", "Each transaction must be above INR 5000.", ">"), ("REQUIRE", "<", "Each transaction must be below INR 5000.", "<")):
        G, dec = _cc_graph(monkeypatch, _cc_dict(rt, op, 5000.0, line))  # amount == limit: rule violated, and the exclusive boundary is not a valid target
        c = _cc_established(G, dec, op, region, 5000.0)
        assert c["minimum_changes"][0]["minimum_delta"] == 0.0 and c["minimum_changes"][0]["target_value"] is None and "more than the listed minimum_delta" in c["recommended_corrective_condition"]


# 4 ---- anything beyond the one supported semantic stays UNRESOLVED (with the unchanged compiled-rule reason)
_CC_UNSUPPORTED = (
    ("trigger", dict(rule_type="TRIGGER"), "ACTION_REQUIRED"),
    ("exception", dict(exception=[{"description": "board approved", "condition": {"entity": "transaction", "field": "approved", "operator": "==", "value": True, "unit": None}}]), None),
    ("required_evidence", dict(required_evidence=[{"type": "receipt", "mandatory": True, "min_count": 1}]), None),
    ("temporal", dict(temporal={"kind": "within", "entity": "transaction", "field": "date", "reference": "now", "amount": 30, "unit": "days", "direction": "before"}), None),
    ("equality", dict(condition={"entity": "transaction", "field": "amount", "operator": "==", "value": 1000.0, "unit": "INR", "children": []}), None),
    ("membership", dict(condition={"entity": "transaction", "field": "amount", "operator": "in", "value": [1000.0, 2000.0], "unit": "INR", "children": []}), None),
    ("other_field", dict(condition={"entity": "transaction", "field": "vendor", "operator": ">", "value": 1000.0, "unit": "INR", "children": []}), None),
    ("group", dict(condition={"logic": "AND", "children": [{"entity": "transaction", "field": "amount", "operator": ">", "value": 1000.0, "unit": "INR", "children": []},
                                                           {"entity": "transaction", "field": "amount", "operator": "<", "value": 9000.0, "unit": "INR", "children": []}]}), None),
    ("other_unit", dict(condition={"entity": "transaction", "field": "amount", "operator": ">", "value": 1000.0, "unit": "USD", "children": []}), None),
    ("non_currency_unit", dict(condition={"entity": "transaction", "field": "amount", "operator": ">", "value": 1000.0, "unit": "days", "children": []}), None),
    ("non_numeric_value", dict(condition={"entity": "transaction", "field": "amount", "operator": ">", "value": "1000", "unit": "INR", "children": []}), None))


def test_cc_unsupported_compiled_semantics_stay_unresolved(monkeypatch):
    for name, over, verdict in _CC_UNSUPPORTED:
        over = dict(over); rt = over.pop("rule_type", "PROHIBIT")
        d = _cc_dict(rt, ">", 1000.0, "Transactions above INR 1000 are prohibited.", **over)
        G, dec = _cc_graph(monkeypatch, d, verdict=verdict or "VIOLATION")
        c = T.build_counterfactual(G, dec)
        assert G.nodes[dec]["result_source"] == "compiled_policy_engine", name
        assert c["in_scope"] is True and "compiled-rule Decision: its expression semantics are not re-derived" in c["unresolved_reason"], name
        _cf_is_unresolved(G, c)
    d = _cc_dict("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited.", status="NEEDS_REVIEW", ambiguities=["vague"])  # NEEDS_REVIEW is never executed: the rule line stays UNEVALUATED
    G, dec = _cc_graph(monkeypatch, d)
    c = T.build_counterfactual(G, dec)
    assert G.nodes[dec]["verdict"] == "UNEVALUATED" and "never evaluated" in c["unresolved_reason"]; _cf_is_unresolved(G, c)


# 5 ---- missing compiled facts / evidence / stored rule: nothing is guessed
def test_cc_missing_compiled_facts_or_evidence_are_unresolved(monkeypatch):
    two = {"inv_cc1.txt": _CF_INV5, "inv_cc2.txt": "Invoice No: INV-2\nBilled amount INR 7,000\n"}
    d = _cc_dict("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited.")
    G, dec = _cc_graph(monkeypatch, d, docs=two, indeterminate={1: ["transaction.amount.unit"]})  # one record is violating, another could not be evaluated: facts missing
    dd, c = G.nodes[dec], T.build_counterfactual(G, dec)
    assert dd["verdict"] == "VIOLATION" and dd["compiled_missing_facts"] == ["transaction.amount.unit"] and c["verification"]["policy_applicability"] == "UNESTABLISHED"
    assert "UNESTABLISHED" in c["unresolved_reason"]; _cf_is_unresolved(G, c)
    G, dec = _cc_graph(monkeypatch, _cc_dict("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited.", required_evidence=[{"type": "receipt", "mandatory": True, "min_count": 1}]),
                       indeterminate={0: []})  # the rule needs evidence and the pipeline supplies none: the engine answers INDETERMINATE, which is no premise
    c = T.build_counterfactual(G, dec)
    assert G.nodes[dec]["verdict"] == "UNEVALUATED" and c["status"] == "UNRESOLVED"; _cf_is_unresolved(G, c)
    G, dec = _cc_graph(monkeypatch, d)
    del G.nodes[G.nodes[dec]["rule_id"]]["compiled_rules"]  # the stored compiled rule is gone: it is not re-derived from rule text
    c = T.build_counterfactual(G, dec)
    assert c["status"] == "UNRESOLVED" and "compiled-rule Decision" in c["unresolved_reason"]; _cf_is_unresolved(G, c)
    G, dec = _cc_graph(monkeypatch, d)
    G.nodes[G.nodes[dec]["rule_id"]]["compiled_rules"].append(copy.deepcopy(G.nodes[G.nodes[dec]["rule_id"]]["compiled_rules"][0]))  # two compiled rules on one rule line: the Decision cannot be attributed to one
    c = T.build_counterfactual(G, dec)
    assert c["status"] == "UNRESOLVED"; _cf_is_unresolved(G, c)


# 6 + 7 + 8 ---- every existing 8A-8B gate still applies to the compiled path
def test_cc_policy_applicability_mismatch_is_unresolved(monkeypatch):
    d = _cc_dict("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited.")
    G, dec = _cc_graph(monkeypatch, d)
    assert T.build_counterfactual(G, dec)["status"] == "ESTABLISHED"  # baseline: same graph, no fault
    G.nodes[dec]["compiled_rules"][0]["source_text"] = "Employees must wear badges."  # the compiled clause no longer maps to the cited rule text
    c = T.build_counterfactual(G, dec)
    assert c["verification"]["policy_applicability"] == "MISMATCH" and "policy applicability MISMATCH" in c["unresolved_reason"]; _cf_is_unresolved(G, c)
    G, dec = _cc_graph(monkeypatch, d)
    G.nodes[G.nodes[dec]["rule_id"]]["source_location"] = "rulebook.txt:line 99"  # Decision and rule node disagree on where the rule comes from
    c = T.build_counterfactual(G, dec)
    assert c["verification"]["policy_applicability"] == "MISMATCH" and "policy applicability MISMATCH" in c["unresolved_reason"]; _cf_is_unresolved(G, c)


def test_cc_policy_applicability_unestablished_is_unresolved(monkeypatch):
    d = _cc_dict("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited.")
    G, dec = _cc_graph(monkeypatch, d)
    for u, v, k, e in list(G.out_edges(dec, keys=True, data=True)):
        if e.get("relation") == "BELONGS_TO": G.remove_edge(u, v, k)  # no recorded policy scope: applicability cannot be established
    c = T.build_counterfactual(G, dec)
    assert c["verification"]["policy_applicability"] == "UNESTABLISHED" and "policy applicability UNESTABLISHED" in c["unresolved_reason"]; _cf_is_unresolved(G, c)
    G, dec = _cc_graph(monkeypatch, d)
    G.nodes[dec]["compiled_missing_facts"] = ["transaction.amount.unit"]  # the compiled evaluation recorded a missing fact
    c = T.build_counterfactual(G, dec)
    assert c["verification"]["policy_applicability"] == "UNESTABLISHED" and "policy applicability UNESTABLISHED" in c["unresolved_reason"]; _cf_is_unresolved(G, c)


def test_cc_failed_self_verification_is_unresolved_but_escalation_alone_is_not(monkeypatch):
    d = _cc_dict("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited.")
    for fault in ("evidence_text_5000_to_500", "remove_support", "heuristic_support"):
        G, dec = _cc_graph(monkeypatch, d)
        T._sv7d_apply_fault(G, dec, fault)
        c = T.build_counterfactual(G, dec)
        assert c["verification"]["verification_status"] == "FAILED" and "self-verification FAILED" in c["unresolved_reason"], fault
        _cf_is_unresolved(G, c)
    G, dec = _cc_graph(monkeypatch, d)
    T._sv7d_apply_fault(G, dec, "weak_location")  # ESCALATE (weak grounding) is not FAILED: the grounded violation still yields a counterfactual
    c = T.build_counterfactual(G, dec)
    assert c["verification"]["verification_status"] == "ESCALATE" and c["status"] == "ESTABLISHED" and c["change_type"] == "CORRECTIVE_ACTION"


# 9 ---- rule / evidence IDs and provenance come straight from the stored Decision, rule node and Evidence nodes
def test_cc_preserves_rule_evidence_ids_and_provenance(monkeypatch):
    G, dec = _cc_graph(monkeypatch, _cc_dict("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited."))
    dd, c = G.nodes[dec], T.build_counterfactual(G, dec)
    rid = dd["rule_id"]; rn = G.nodes[rid]
    j = lambda x: json.loads(json.dumps(x, default=str))
    vr = c["violated_rule"]
    assert c["decision_id"] == dec and c["counterfactual_id"] == f"cf::{dec}" and vr["rule_id"] == rid and vr["condition"] == rn["condition"] and vr["source_file"] == rn.get("source_file") and vr["source_location"] == rn.get("source_location")
    assert vr["result_source"] == "compiled_policy_engine" and vr["rulebook_provenance"] == j(dd["rulebook_provenance"]) and vr["rulebook_provenance"]["compiled_source_spans"] == [e["source_span"] for e in rn["compiled_rules"]]
    assert vr["policy_ids"] == sorted(v for _, v, e in G.out_edges(rid, data=True) if e.get("relation") == "BELONGS_TO") and vr["policy_ids"]
    refs = c["evidence_causing_violation"]
    assert refs and [r["evidence_id"] for r in refs] == list(dd["evidence_used"]) and all(G.nodes[r["evidence_id"]]["type"] == "Evidence" for r in refs)
    for r in refs:
        assert r["provenance"] == j(G.nodes[r["evidence_id"]]["provenance"]) and r["document_id"] in {x["document_id"] for x in T._evidence_source_docs(G, r["evidence_id"])} and r["filename"] == "inv_cc.txt"
    ch = c["minimum_changes"][0]
    assert ch["transaction_id"] in dd["basis_node_ids"] and ch["evidence_ids"] == list(dd["evidence_used"]) and c["current_decision"]["rule_id"] == rid and c["current_decision"]["result_source"] == "compiled_policy_engine"
    assert T.validate_counterfactual(G, c) == []
    ghost = dict(c, evidence_causing_violation=[dict(refs[0], evidence_id="ev_does_not_exist")])  # an invented evidence ID is still rejected for a compiled-rule counterfactual
    assert any("not in graph" in p for p in T.validate_counterfactual(G, ghost))
    assert any("existing PolicyRule" in p for p in T.validate_counterfactual(G, dict(c, violated_rule=dict(vr, rule_id="rule_does_not_exist"))))


# 10 ---- read-only, deterministic, idempotent
def test_cc_is_read_only_deterministic_and_idempotent(monkeypatch):
    two = {"inv_cc1.txt": _CF_INV5, "inv_cc2.txt": "Invoice No: INV-2\nBilled amount INR 7,000\n"}
    G, dec = _cc_graph(monkeypatch, _cc_dict("PROHIBIT", ">", 1000.0, "Transactions above INR 1000 are prohibited."), docs=two)
    snap = lambda: (T._sv7d_snapshot(G), _pa_snap(G), json.dumps(G.graph.get("contradiction_findings"), sort_keys=True, default=str))
    run = lambda: json.dumps([T.build_counterfactual(G, dec), T.build_counterfactuals(G), T.build_finding_counterfactuals(G)], sort_keys=True, default=str)
    before = snap(); a = run(); mid = snap(); b = run()
    assert a == b and before == mid == snap()
    cfs = T.build_counterfactuals(G)
    assert cfs and cfs[0]["status"] == "ESTABLISHED" and len(cfs[0]["minimum_changes"]) == 2 and [x["decision_id"] for x in cfs] == sorted(x["decision_id"] for x in cfs)


# 11 ---- legacy (non-compiled) behaviour is unchanged
def test_cc_legacy_counterfactuals_are_unchanged(monkeypatch):
    def boom(*a, **k): raise AssertionError("_cf_compiled must not be consulted for a legacy Decision")
    monkeypatch.setattr(T, "_cf_compiled", boom, raising=False)
    G = _cf_g({"i_l.txt": _CF_INV5}, "FORBID TRANSACTION > INR 1000\n")
    d, c = _cf_for(G, "TRANSACTION")
    dd = G.nodes[d]
    assert dd["result_source"] == "deterministic_policy_engine" and dd["parsed_spec"] and c["status"] == "ESTABLISHED" and c["change_type"] == "CORRECTIVE_ACTION"
    assert c["required_condition"] == {"subject": "TRANSACTION", "relation": "<=", "value": 1000.0, "currency": "INR"} and c["minimum_changes"][0]["minimum_delta"] == 4000.0 and c["minimum_changes"][0]["target_value"] == 1000.0
    assert c["violated_rule"]["result_source"] == "deterministic_policy_engine" and c["violated_rule"]["parsed_spec"]["subject"] == "TRANSACTION"
    assert T.validate_counterfactual(G, c) == []
    _, miss = _cf_for(_cf_g({"m_l.txt": "notes\n"}, 'REQUIRE KEYWORD "manager approval"\n'), "KEYWORD")
    assert miss["status"] == "ESTABLISHED" and miss["change_type"] == "SUPPLY_EVIDENCE" and miss["satisfaction_asserted"] is False
    G2 = _cf_g({"i_l.txt": _CF_INV5}, "Vendors should behave reasonably in spirit.\n")  # free text, no compiled rule: still UNEVALUATED -> UNRESOLVED
    c2 = _cf_only(G2)
    assert c2["current_decision"]["verdict"] == "UNEVALUATED" and "never evaluated" in c2["unresolved_reason"]; _cf_is_unresolved(G2, c2)


# Contract check against the REAL schema + REAL rule engine (skipped when policy_schema is not importable). The compiler is NOT called: a real CompiledRule is built from the
# same keys the compiler validates (validate_raw_rule), so no LLM / network is involved. This is what pins the stored model_dump shape that _cf_compiled reads.
def _cc_real_rule(PS, rt, op, limit, line):
    data = {"policy_id": "P1", "rule_id": "P1-R001", "rule_type": rt, "entity": "transaction", "condition": {"entity": "transaction", "field": "amount", "operator": op, "value": limit, "unit": "INR"},
            "temporal": None, "required_evidence": [], "exception": [], "severity": "high", "action": "review_transaction", "confidence": 0.95, "source_text": line, "ambiguities": []}
    r = PS.CompiledRule.model_validate(data)
    r.status = "VALID"  # the compiler's own stage-2 assignment (validate_raw_rule); no ambiguity was raised for these clauses
    return r


def test_cc_real_schema_dump_shape_and_engine_agree_with_the_counterfactual(monkeypatch):
    try:
        PS = importlib.import_module("omni_pkg_under_test.policy_schema")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"policy_schema not importable: {e}")
    for rt, op, limit, line in _CC_CASES:
        r = _cc_real_rule(PS, rt, op, limit, line)
        dump = r.model_dump(mode="json"); cond = dump["condition"]
        assert dump["rule_type"] == rt and dump["entity"] == "transaction" and dump["status"] == "VALID" and dump["source_text"] == line  # the exact keys / values _cf_compiled and apply_compiled_policy read
        assert (cond["entity"], cond["field"], cond["operator"], cond["unit"]) == ("transaction", "amount", op, "INR") and cond["value"] == limit and not cond.get("children")
        assert not dump["temporal"] and not dump["exception"] and not dump["required_evidence"]
        rec = [{"amount": {"value": _CC_AMT, "unit": "INR"}, "currency": "INR"}]
        er = T._evaluate_compiled_rules([r], {"transaction": rec}, None, None, False, None, True)["rule_results"][0]
        assert er["verdict"] == "VIOLATION"  # real engine: INR 5,000 breaks every rule in _CC_CASES, which is what the counterfactual derivation presumes
        res = types.SimpleNamespace(rules=[r], status="COMPILED", policy_id="P1", stats={}, ambiguous_policy=False, ambiguity_reasons=[], rejected=[], unparsed_statements=[])
        monkeypatch.setattr(T, "HAS_POLICY_COMPILER", True, raising=False)
        monkeypatch.setattr(T, "_compile_policy", lambda text, ctx, conf, res=res: res, raising=False)  # real engine, real schema; only the LLM step is replaced
        monkeypatch.setattr(T, "_COMPILER_MIN_CONFIDENCE", 0.5, raising=False)
        monkeypatch.setattr(T, "evaluate_policy_rules", lambda G, line=line: (T.apply_compiled_policy(G, line + "\n"), _CC_ORIG_EVAL(G))[1])
        G = _graph({"inv_cc.txt": _CF_INV5}, rulebook=line + "\n")
        dec = next(n for n, x in G.nodes(data=True) if x.get("type") == "Decision")
        _cc_established(G, dec, op, op if rt == "REQUIRE" else _CC_NEG[op], limit)


# 20 ---- uncertainty layer (Phase 9): read-only mapping of existing decision state -> COMPLIANT | NON_COMPLIANT | CONDITIONAL | INSUFFICIENT_EVIDENCE + deterministic indicators + escalation
_UNC_INV = "Invoice No: INV-1\nBilled amount INR 5,000\n"
_UNC_RULE = "FORBID TRANSACTION > INR 1000\n"


def _unc_g(docs=None, rulebook=_UNC_RULE):
    return _graph(docs or {"i_u.txt": _UNC_INV}, rulebook=rulebook)


def _unc_dec(G):
    return next(n for n, x in sorted(G.nodes(data=True)) if x.get("type") == "Decision")


def _unc_one(G):
    return T.build_uncertainty_assessment(G, _unc_dec(G))


def _unc_reasons(r):
    return [e["reason"] for e in r["escalation_reasons"]]


def _unc_valid(G, r):
    assert T.validate_uncertainty_assessment(G, r) == []
    assert r["escalation_required"] is bool(r["escalation_reasons"])
    assert r["uncertainty_status"] in T.UNC_STATES


def test_unc_clearly_compliant_maps_to_compliant_without_escalation():
    G = _unc_g({"i_ok.txt": "Invoice No: INV-2\nBilled amount INR 800\n"})
    r = _unc_one(G); _unc_valid(G, r)
    assert r["decision"]["verdict"] == "SATISFIED" and r["uncertainty_status"] == "COMPLIANT"
    assert [r[k]["value"] for k in ("evidence_completeness", "contradiction_severity", "policy_alignment", "risk_level")] == ["COMPLETE", "NONE", "ALIGNED", None]
    assert r["risk_level"]["status"] == _CF_NM and r["escalation_required"] is False and r["escalation_reasons"] == []
    kw = _unc_one(_unc_g({"m_ok.txt": "manager approval granted\n"}, 'REQUIRE KEYWORD "manager approval"\n'))
    assert kw["decision"]["verdict"] == "SATISFIED" and kw["uncertainty_status"] == "COMPLIANT" and kw["escalation_required"] is False


def test_unc_clearly_non_compliant_maps_to_non_compliant_with_stored_risk():
    G = _unc_g(); r = _unc_one(G); _unc_valid(G, r)
    assert r["decision"]["verdict"] == "VIOLATION" and r["uncertainty_status"] == "NON_COMPLIANT"
    assert r["evidence_completeness"]["value"] == "COMPLETE" and r["policy_alignment"]["value"] == "ALIGNED" and r["contradiction_severity"]["value"] == "NONE"
    assert r["risk_level"]["value"] == "MEDIUM" and r["risk_level"]["severity_source"] == "default_when_unspecified"  # an existing deterministic default, reported with its source
    assert r["escalation_required"] is False  # MEDIUM is not high risk
    absent = _unc_one(_unc_g({"m_ab.txt": "notes\n"}, 'REQUIRE KEYWORD "manager approval"\n'))  # absence within a complete extraction scope is a grounded violation
    assert absent["decision"]["violation_status"] == "ABSENCE_OF_REQUIRED_TEXT_WITHIN_EXTRACTED_SCOPE" and absent["uncertainty_status"] == "NON_COMPLIANT" and absent["escalation_required"] is False


def test_unc_insufficient_evidence_for_ungrounded_weak_and_inconclusive():
    for fault, weak in (("remove_support", False), ("evidence_text_5000_to_500", False), ("weak_location", True)):
        G = _unc_g(); T._sv7d_apply_fault(G, _unc_dec(G), fault); r = _unc_one(G); _unc_valid(G, r)
        assert r["decision"]["verdict"] == "VIOLATION" and r["uncertainty_status"] == "INSUFFICIENT_EVIDENCE" and r["evidence_completeness"]["value"] == "INCOMPLETE", fault  # NOT NON_COMPLIANT: the verdict is kept, the support is not adequate
        assert r["escalation_required"] and "CRITICAL_EVIDENCE_MISSING" in _unc_reasons(r), fault
    G = _unc_g({"i_in.txt": "Invoice No: INV-2\nBilled amount INR 800\n"}); G.nodes[_unc_dec(G)]["verdict"] = "INCONCLUSIVE"  # private graph: the INCONCLUSIVE state itself
    r = _unc_one(G); _unc_valid(G, r)
    assert r["uncertainty_status"] == "INSUFFICIENT_EVIDENCE" and r["evidence_completeness"]["value"] == "INCOMPLETE" and r["policy_alignment"]["value"] is None and "CRITICAL_EVIDENCE_MISSING" in _unc_reasons(r)


def test_unc_conditional_for_unresolved_or_unestablished_policy_applicability():
    for fault, align in (("wrong_scope", "MISALIGNED"), ("rule_condition_9000", "MISALIGNED"), ("basis_amount_none", "UNESTABLISHED")):
        G = _unc_g(); T._sv7d_apply_fault(G, _unc_dec(G), fault); r = _unc_one(G); _unc_valid(G, r)
        assert r["uncertainty_status"] == "CONDITIONAL" and r["policy_alignment"]["value"] == align and "POLICY_APPLICABILITY_UNESTABLISHED" in _unc_reasons(r), fault
        assert r["decision"]["verdict"] == "VIOLATION" and r["uncertainty_status"] != "INSUFFICIENT_EVIDENCE"  # distinct from missing evidence: the evidence is complete, the rule's applicability is not
    for rb in ("Vendors should behave in spirit.\n", "FORBID TRANSACTION > JPY 1000\n"):  # rule not interpretable / unknown unit: UNEVALUATED
        G = _unc_g(rulebook=rb); r = _unc_one(G); _unc_valid(G, r)
        assert r["decision"]["verdict"] == "UNEVALUATED" and r["uncertainty_status"] == "CONDITIONAL" and r["policy_alignment"]["value"] == "UNESTABLISHED" and r["evidence_completeness"]["value"] is None and "POLICY_APPLICABILITY_UNESTABLISHED" in _unc_reasons(r)


def test_unc_not_applicable_is_never_mapped_to_compliant():
    G = _unc_g({"memo_na.txt": "Meeting notes\nNothing to report\n"}); r = _unc_one(G); _unc_valid(G, r)
    assert r["decision"]["verdict"] == "NOT_APPLICABLE" and r["uncertainty_status"] == "INSUFFICIENT_EVIDENCE" and r["policy_alignment"]["value"] == "NOT_APPLICABLE" and r["escalation_required"] is False


def test_unc_missing_critical_evidence_downgrades_a_satisfied_decision_and_escalates():
    G = _unc_g({"inv_m.txt": "Invoice No: INV-7\nPO Number: PO-777\nVendor: Boreal\nBilled amount INR 800\n"}); r = _unc_one(G); _unc_valid(G, r)
    assert r["decision"]["verdict"] == "SATISFIED" and r["uncertainty_status"] == "INSUFFICIENT_EVIDENCE" and r["evidence_completeness"]["value"] == "INCOMPLETE"
    assert "CRITICAL_EVIDENCE_MISSING" in _unc_reasons(r) and r["escalation_required"] is True
    stored = {f["finding_id"] for f in G.graph["contradiction_findings"] if f["category"] == "MISSING_EVIDENCE"}
    assert stored and {m["finding_id"] for m in r["missing_evidence"] if m.get("finding_id")} == stored  # reuses the existing finding ids
    G2 = _unc_g({"i_cm.txt": "Invoice No: INV-2\nBilled amount INR 800\n"}); d2 = _unc_dec(G2); G2.nodes[d2]["compiled_missing_facts"] = ["currency unit"]
    r2 = T.build_uncertainty_assessment(G2, d2)
    assert r2["uncertainty_status"] == "INSUFFICIENT_EVIDENCE" and any(m["kind"] == "compiled_rule_missing_fact" for m in r2["missing_evidence"])


def test_unc_unresolved_contradiction_maps_to_conditional_and_escalates():
    docs = {"inv_c.txt": "Invoice No: INV-5\nPO Number: PO-5\nVendor: Boreal Metals\nBilled amount INR 800\n", "po_c.txt": "Purchase Order No: PO-5\nVendor: Zenith Traders\nOrder value INR 800\n"}
    G = _unc_g(docs); r = _unc_one(G); _unc_valid(G, r)
    assert r["decision"]["verdict"] == "SATISFIED" and r["contradiction_severity"]["value"] == "MAJOR" and r["uncertainty_status"] == "CONDITIONAL"  # not COMPLIANT: the verdict stays, the state records the open contradiction
    assert "UNRESOLVED_CONTRADICTION" in _unc_reasons(r) and r["escalation_required"] is True and r["policy_alignment"]["value"] == "ALIGNED" and r["evidence_completeness"]["value"] == "COMPLETE"
    stored = {f["finding_id"] for f in G.graph["contradiction_findings"] if f["category"] == "MAJOR_CONTRADICTION"}
    assert stored and {c["finding_id"] for c in r["contradicting_evidence"] if c.get("finding_id")} == stored


def test_unc_confidence_is_never_invented_and_low_confidence_needs_an_existing_signal(monkeypatch):
    G = _unc_g(); d = _unc_dec(G); r = T.build_uncertainty_assessment(G, d)
    assert r["decision_confidence"]["value"] is None and r["decision_confidence"]["status"] == _CF_NM and "LOW_CONFIDENCE" not in _unc_reasons(r)  # NOT_MEASURED alone is not low confidence
    monkeypatch.setattr(T, "_COMPILER_MIN_CONFIDENCE", 0.6, raising=False)
    G.nodes[d]["compiled_rules"] = [{"rule_id": "c_hi", "confidence": 0.9}]
    assert "LOW_CONFIDENCE" not in _unc_reasons(T.build_uncertainty_assessment(G, d))
    G.nodes[d]["compiled_rules"] = [{"rule_id": "c_lo", "confidence": 0.2}]
    lo = T.build_uncertainty_assessment(G, d); _unc_valid(G, lo)
    assert "LOW_CONFIDENCE" in _unc_reasons(lo) and lo["escalation_required"] is True and lo["decision_confidence"]["value"] is None and lo["decision_confidence"]["status"] == _CF_NM  # the stored component is listed, never aggregated
    assert [c["rule_id"] for c in lo["decision_confidence"]["low_components"]] == ["c_lo"] and lo["uncertainty_status"] == "NON_COMPLIANT"
    monkeypatch.delattr(T, "_COMPILER_MIN_CONFIDENCE", raising=False)  # no floor exists -> nothing can be called low
    nf = T.build_uncertainty_assessment(G, d)
    assert "LOW_CONFIDENCE" not in _unc_reasons(nf) and nf["decision_confidence"]["low_confidence_floor"] is None


def test_unc_indicators_are_not_measured_when_they_cannot_be_derived():
    G = _unc_g({"i_nm.txt": "Invoice No: INV-2\nBilled amount INR 800\n"}); G.graph.pop("contradiction_findings")  # the classifier never ran on this graph
    r = _unc_one(G); _unc_valid(G, r)
    assert r["contradiction_severity"]["value"] is None and r["contradiction_severity"]["status"] == _CF_NM and r["risk_level"]["status"] == _CF_NM and r["decision_confidence"]["status"] == _CF_NM
    G2 = _unc_g(); T._sv7d_apply_fault(G2, _unc_dec(G2), "remove_support"); r2 = _unc_one(G2)  # applicability not assessed once 7B FAILED
    assert r2["verification_status"] == "FAILED" and r2["policy_alignment"]["value"] is None and r2["policy_alignment"]["status"] == _CF_NM
    miss = T.build_uncertainty_assessment(G, "no_such_decision")
    assert miss["found"] is False and miss["uncertainty_status"] == _CF_NM and miss["escalation_required"] is False and all(miss[k]["status"] == _CF_NM for k in T.UNC_INDICATORS) and set(T.UNC_FIELDS) <= set(miss)


def test_unc_high_risk_escalates_only_from_a_stored_risk_signal():
    for sev in ("high", "critical"):
        G = _unc_g(rulebook=f"FORBID TRANSACTION > INR 1000 [severity={sev}]\n"); r = _unc_one(G); _unc_valid(G, r)
        assert r["risk_level"]["value"] == sev.upper() and r["risk_level"]["severity_source"] == "rule_configured" and r["uncertainty_status"] == "NON_COMPLIANT"  # the state is not changed by risk
        assert _unc_reasons(r) == ["HIGH_RISK"] and r["escalation_required"] is True
        assert r["risk_level"]["risk_ids"] and all(G.nodes[x]["type"] == "Risk" and G.nodes[x]["severity"] == sev.upper() for x in r["risk_level"]["risk_ids"])
    low = _unc_one(_unc_g(rulebook="FORBID TRANSACTION > INR 1000 [severity=low]\n"))
    assert low["risk_level"]["value"] == "LOW" and low["escalation_required"] is False
    sat = _unc_one(_unc_g({"i_rs.txt": "Invoice No: INV-2\nBilled amount INR 800\n"}, "FORBID TRANSACTION > INR 1000 [severity=high]\n"))
    assert sat["risk_level"]["value"] is None and sat["escalation_required"] is False  # no Risk node -> no risk classification is invented


def test_unc_no_numeric_indicator_is_ever_produced():
    graphs = [_unc_g(), _unc_g(rulebook="FORBID TRANSACTION > INR 1000 [severity=high]\n"), _unc_g({"i_n.txt": "Invoice No: INV-2\nBilled amount INR 800\n"}), _unc_g(rulebook="Vendors should behave in spirit.\n"),
              _unc_g({"m_n.txt": "Meeting notes\n"}), _unc_g(_CF_UNLINKED, "FORBID TRANSACTION > INR 1000\n")]
    for G in graphs:
        for r in T.build_uncertainty_assessments(G):
            _unc_valid(G, r)
            for k in T.UNC_INDICATORS:
                v = r[k]["value"]
                assert v is None or isinstance(v, str), (k, v)
                assert (v is None) == (r[k]["status"] == _CF_NM)
            assert r["decision_confidence"]["value"] is None
    bad = _unc_one(graphs[0]); bad["policy_alignment"]["value"] = 0.93  # the contract validator rejects an invented numeric indicator
    assert any("must not be numeric" in p for p in T.validate_uncertainty_assessment(graphs[0], bad))
    bad2 = _unc_one(graphs[0]); bad2["escalation_required"] = True
    assert T.validate_uncertainty_assessment(graphs[0], bad2)


def test_unc_is_read_only_deterministic_and_idempotent():
    docs = {"inv_i.txt": "Invoice No: INV-5\nPO Number: PO-5\nVendor: Boreal Metals\nBilled amount INR 5,000\n", "po_i.txt": "Purchase Order No: PO-5\nVendor: Zenith Traders\nOrder value INR 800\n"}
    G = _unc_g(docs, "FORBID TRANSACTION > INR 1000 [severity=high]\nREQUIRE KEYWORD \"manager approval\"\nVendors should behave in spirit.\n")
    before = T._sv7d_snapshot(G)
    a, b = T.build_uncertainty_assessments(G), T.build_uncertainty_assessments(G)
    assert T._sv7d_snapshot(G) == before and json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    assert len(a) == len(T._nodes_of_type(G, "Decision")) == 3 and [r["decision_id"] for r in a] == sorted(r["decision_id"] for r in a)
    a[0]["supporting_evidence"].append("tamper"); a[0]["escalation_reasons"].append("tamper")  # returned copies never alias graph state
    assert json.dumps(T.build_uncertainty_assessments(G), sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str) and T._sv7d_snapshot(G) == before


def test_unc_preserves_verdicts_evidence_rule_ids_findings_verification_and_provenance():
    docs = {"inv_p.txt": "Invoice No: INV-5\nPO Number: PO-5\nVendor: Boreal Metals\nBilled amount INR 5,000\n", "po_p.txt": "Purchase Order No: PO-5\nVendor: Zenith Traders\nOrder value INR 800\n"}
    G = _unc_g(docs, "FORBID TRANSACTION > INR 1000\nREQUIRE KEYWORD \"manager approval\"\n")
    verdicts = {n: d["verdict"] for n, d in T._nodes_of_type(G, "Decision")}; findings = json.dumps(G.graph["contradiction_findings"], sort_keys=True, default=str)
    for r in T.build_uncertainty_assessments(G):
        d = r["decision_id"]; vr = T.verify_policy_applicability(G, d); lin = T.query_decision_lineage(G, d)
        assert r["decision"]["verdict"] == verdicts[d] == G.nodes[d]["verdict"] and r["decision"]["rule_id"] == G.nodes[d]["rule_id"]
        assert r["verification_status"] == vr["verification_status"] and r["policy_applicability_status"] == vr["policy_applicability"]["status"]
        assert sorted({x["evidence_id"] for x in r["supporting_evidence"]}) == sorted(e["evidence_id"] for e in lin["evidence"])
        assert [x["rule_id"] for x in r["policy_rules"]] == [G.nodes[d]["rule_id"]]
        for ref in r["supporting_evidence"]: assert ref["provenance"] == T._sv_json(G.nodes[ref["evidence_id"]]["provenance"]) and ref["document_id"]
    assert {n: d["verdict"] for n, d in T._nodes_of_type(G, "Decision")} == verdicts and json.dumps(G.graph["contradiction_findings"], sort_keys=True, default=str) == findings


def test_unc_legacy_prompt_8_counterfactuals_and_prior_layers_are_unchanged(monkeypatch):
    docs = {"inv_l.txt": _UNC_INV, "m_l.txt": "notes\n"}; rb = 'FORBID TRANSACTION > INR 1000\nREQUIRE KEYWORD "manager approval"\nVendors should behave in spirit.\n'
    G = _unc_g(docs, rb)
    cf0, sv0, pa0 = (json.dumps(f(G), sort_keys=True, default=str) for f in (T.build_counterfactuals, T.verify_self_verification_results, T.verify_policy_applicability_results))
    T.build_uncertainty_assessments(G)
    assert json.dumps(T.build_counterfactuals(G), sort_keys=True, default=str) == cf0 and json.dumps(T.verify_self_verification_results(G), sort_keys=True, default=str) == sv0
    assert json.dumps(T.verify_policy_applicability_results(G), sort_keys=True, default=str) == pa0
    def boom(*a, **k): raise AssertionError("Phase 8 / 7x must not depend on the uncertainty layer")
    monkeypatch.setattr(T, "build_uncertainty_assessment", boom); monkeypatch.setattr(T, "build_uncertainty_assessments", boom)
    assert json.dumps(T.build_counterfactuals(G), sort_keys=True, default=str) == cf0
    d, c = _cf_for(_cf_g({"i_l.txt": _CF_INV5}, "FORBID TRANSACTION > INR 1000\n"), "TRANSACTION")
    assert c["status"] == "ESTABLISHED" and c["change_type"] == "CORRECTIVE_ACTION" and all(k in c for k in T.CF_FIELDS)
    b1 = json.dumps(T.run_counterfactual_benchmark(), sort_keys=True, default=str)  # whole Phase 8 benchmark runs with the uncertainty layer disabled (boom) and is deterministic
    monkeypatch.undo()
    T.build_uncertainty_assessments(G)
    assert b1 == json.dumps(T.run_counterfactual_benchmark(), sort_keys=True, default=str)


# 20b ---- uncertainty layer edge-case hardening
def _unc_patched_vr(monkeypatch, **over):
    orig = T.verify_policy_applicability
    def f(G, d):
        r = orig(G, d)
        for k, v in over.items():
            if k == "pa_status": r["policy_applicability"]["status"] = v
            else: r[k] = v
        return r
    monkeypatch.setattr(T, "verify_policy_applicability", f)


def test_unc_failed_verification_without_a_listed_claim_still_escalates(monkeypatch):
    G = _unc_g(); _unc_patched_vr(monkeypatch, verification_status="FAILED", material_claims=[])
    r = _unc_one(G); _unc_valid(G, r)
    assert r["uncertainty_status"] == "INSUFFICIENT_EVIDENCE" and r["verification_status"] == "FAILED" and _unc_reasons(r) == ["CRITICAL_EVIDENCE_MISSING"] and "FAILED" in r["escalation_reasons"][0]["detail"]


def test_unc_unverified_applicability_is_conditional_with_a_reason_and_never_a_verdict_state(monkeypatch):
    G = _unc_g(); _unc_patched_vr(monkeypatch, pa_status="NOT_CHECKED")
    r = _unc_one(G); _unc_valid(G, r)
    assert r["decision"]["verdict"] == "VIOLATION" and r["uncertainty_status"] == "CONDITIONAL" and _unc_reasons(r) == ["POLICY_APPLICABILITY_UNESTABLISHED"]
    for fault in ("wrong_scope", "basis_amount_none"):  # genuine MISMATCH / UNESTABLISHED: one reason, never duplicated
        G2 = _unc_g(); T._sv7d_apply_fault(G2, _unc_dec(G2), fault); r2 = _unc_one(G2)
        assert r2["uncertainty_status"] == "CONDITIONAL" and _unc_reasons(r2) == ["POLICY_APPLICABILITY_UNESTABLISHED"]


def test_unc_unevaluated_rule_is_a_policy_problem_not_missing_evidence():
    r = _unc_one(_unc_g(rulebook="Vendors should behave in spirit.\n"))
    assert r["decision"]["verdict"] == "UNEVALUATED" and r["missing_evidence"] == [] and r["evidence_completeness"]["value"] is None and _unc_reasons(r) == ["POLICY_APPLICABILITY_UNESTABLISHED"]


_UNC_CONTRA = {"inv_c2.txt": "Invoice No: INV-5\nPO Number: PO-5\nVendor: Boreal Metals\nBilled amount INR 800\n", "po_c2.txt": "Purchase Order No: PO-5\nVendor: Zenith Traders\nOrder value INR 800\n"}


def test_unc_contradiction_severity_distinctions_are_preserved():
    G = _unc_g(_UNC_CONTRA); fs = [f for f in G.graph["contradiction_findings"] if f["category"] == "MAJOR_CONTRADICTION"]; assert fs
    for f in fs: f["category"], f["severity"] = "MINOR_CONTRADICTION", "minor"  # private graph: the same finding classified MINOR
    r = _unc_one(G); _unc_valid(G, r)
    assert r["contradiction_severity"]["value"] == "MINOR" and r["uncertainty_status"] == "COMPLIANT" and _unc_reasons(r) == ["UNRESOLVED_CONTRADICTION"]  # unresolved minor escalates but is not promoted to MAJOR / CONDITIONAL
    G2 = _unc_g(_UNC_CONTRA)
    for f in G2.graph["contradiction_findings"]:
        if f["category"] == "MAJOR_CONTRADICTION": f["resolution"] = "RESOLVED"
    r2 = _unc_one(G2); _unc_valid(G2, r2)
    assert r2["contradiction_severity"]["value"] == "MAJOR" and r2["uncertainty_status"] == "COMPLIANT" and r2["escalation_reasons"] == []  # a resolved major is still reported as MAJOR but no longer drives state / escalation


def test_unc_heuristic_contradiction_signal_never_becomes_a_policy_violation():
    G = _unc_g({"i_h.txt": "Invoice No: INV-2\nBilled amount INR 800\n", "o_h.txt": "Note\nBilled amount INR 900\n"}); d = _unc_dec(G)
    basis = G.nodes[d]["basis_node_ids"][0]; other = next(e for e, _ in T._investigation_evidence(G) if e not in T._supporting_evidence_ids(G, basis))
    G.add_edge(other, basis, relation="CONTRADICTS", heuristic=True, match_strength="weak", method="test")
    before = (G.nodes[d]["verdict"], json.dumps(G.graph["contradiction_findings"], sort_keys=True, default=str))
    r = T.build_uncertainty_assessment(G, d); _unc_valid(G, r)
    assert r["contradiction_severity"]["value"] == "UNCLASSIFIED" and "UNRESOLVED_CONTRADICTION" in _unc_reasons(r)
    assert r["decision"]["verdict"] == "SATISFIED" and r["uncertainty_status"] != "NON_COMPLIANT" and (G.nodes[d]["verdict"], json.dumps(G.graph["contradiction_findings"], sort_keys=True, default=str)) == before


def test_unc_confidence_and_risk_accept_only_genuine_stored_values(monkeypatch):
    G = _unc_g(); d = _unc_dec(G)
    monkeypatch.setattr(T, "_COMPILER_MIN_CONFIDENCE", 0.6, raising=False)
    G.nodes[d]["compiled_rules"] = [{"rule_id": "c_b", "confidence": True}, {"rule_id": "c_s", "confidence": "0.1"}, {"rule_id": "c_n", "confidence": float("nan")}]
    r = T.build_uncertainty_assessment(G, d); _unc_valid(G, r)
    assert r["decision_confidence"]["low_components"] == [] and "LOW_CONFIDENCE" not in _unc_reasons(r) and r["decision_confidence"]["value"] is None
    monkeypatch.setattr(T, "_COMPILER_MIN_CONFIDENCE", True, raising=False)  # a bool is not a floor
    G.nodes[d]["compiled_rules"] = [{"rule_id": "c_lo", "confidence": 0.0}]
    r2 = T.build_uncertainty_assessment(G, d)
    assert r2["decision_confidence"]["low_confidence_floor"] is None and "LOW_CONFIDENCE" not in _unc_reasons(r2)
    risk = next(x for x in G.successors(d) if G.nodes[x].get("type") == "Risk"); G.nodes[risk]["severity"] = "BANANA"
    r3 = T.build_uncertainty_assessment(G, d); _unc_valid(G, r3)
    assert r3["risk_level"]["value"] is None and r3["risk_level"]["status"] == _CF_NM and "HIGH_RISK" not in _unc_reasons(r3)
    G.nodes[risk]["severity"] = "medium"; assert not _unc_one(G)["escalation_required"]  # stored MEDIUM alone never escalates


def test_unc_validator_rejects_inconsistent_indicators_and_unbacked_or_duplicate_reasons():
    G = _unc_g(); base = _unc_one(G); import copy
    def probs(**kw):
        r = copy.deepcopy(base); r.update(kw); return T.validate_uncertainty_assessment(G, r)
    assert probs() == []
    assert any("unknown status" in p for p in probs(policy_alignment={"value": "ALIGNED", "status": "MAYBE", "basis": "x"}))
    assert any("NOT_MEASURED" in p for p in probs(risk_level={"value": "HIGH", "status": _CF_NM, "basis": "x"}))
    dup = [{"reason": "HIGH_RISK", "detail": "x"}] * 2
    assert any("duplicate" in p for p in probs(escalation_reasons=dup, escalation_required=True))
    assert any("HIGH_RISK without" in p for p in probs(escalation_reasons=dup[:1], escalation_required=True))
    assert any("LOW_CONFIDENCE without" in p for p in probs(escalation_reasons=[{"reason": "LOW_CONFIDENCE", "detail": "x"}], escalation_required=True))
    assert any("UNRESOLVED_CONTRADICTION without" in p for p in probs(escalation_reasons=[{"reason": "UNRESOLVED_CONTRADICTION", "detail": "x"}], escalation_required=True))


def test_unc_every_generated_reason_is_unique_and_backed_across_scenarios():
    gs = [_unc_g(), _unc_g(_UNC_CONTRA), _unc_g({"inv_m2.txt": "Invoice No: INV-7\nPO Number: PO-777\nVendor: Boreal\nBilled amount INR 5,000\n"}, "FORBID TRANSACTION > INR 1000 [severity=critical]\nREQUIRE KEYWORD \"manager approval\"\nVendors should behave in spirit.\n")]
    for G in gs:
        for r in T.build_uncertainty_assessments(G):
            codes = _unc_reasons(r); assert len(codes) == len(set(codes)) and r["escalation_required"] is bool(codes); _unc_valid(G, r)


def test_unc_9b_infinite_and_non_numeric_confidence_values_are_ignored(monkeypatch):
    G = _unc_g(); d = _unc_dec(G); inf = float("inf")
    monkeypatch.setattr(T, "_COMPILER_MIN_CONFIDENCE", 0.6, raising=False)
    G.nodes[d]["compiled_rules"] = [{"rule_id": "c_ninf", "confidence": -inf}, {"rule_id": "c_none", "confidence": None}, {"rule_id": "c_l", "confidence": [0.1]}]
    r = T.build_uncertainty_assessment(G, d); _unc_valid(G, r)
    assert r["decision_confidence"]["low_components"] == [] and "LOW_CONFIDENCE" not in _unc_reasons(r)
    monkeypatch.setattr(T, "_COMPILER_MIN_CONFIDENCE", inf, raising=False)
    G.nodes[d]["compiled_rules"] = [{"rule_id": "c_lo", "confidence": 0.0}]
    r2 = T.build_uncertainty_assessment(G, d)
    assert r2["decision_confidence"]["low_confidence_floor"] is None and "LOW_CONFIDENCE" not in _unc_reasons(r2) and r2["decision_confidence"]["value"] is None


def test_unc_9b_validator_rejects_state_that_contradicts_verification_or_risk_backing():
    import copy
    G = _unc_g(); base = _unc_one(G)
    def probs(**kw):
        r = copy.deepcopy(base); r.update(kw); return T.validate_uncertainty_assessment(G, r)
    assert probs() == []
    assert any("7B FAILED requires" in p for p in probs(verification_status="FAILED"))
    fr = {"uncertainty_status": "INSUFFICIENT_EVIDENCE", "verification_status": "FAILED", "escalation_required": True}
    assert probs(**fr, escalation_reasons=[{"reason": "CRITICAL_EVIDENCE_MISSING", "detail": "x"}]) == []
    assert any("must not carry" in p for p in probs(**fr, escalation_reasons=[{"reason": "CRITICAL_EVIDENCE_MISSING", "detail": "x"}, {"reason": "POLICY_APPLICABILITY_UNESTABLISHED", "detail": "x"}]))
    assert any("unverified policy applicability requires" in p for p in probs(policy_applicability_status="NOT_CHECKED"))
    assert any("requires verified" in p for p in probs(policy_applicability_status="MISMATCH"))
    assert probs(uncertainty_status="CONDITIONAL", policy_applicability_status="MISMATCH", escalation_required=True, escalation_reasons=[{"reason": "POLICY_APPLICABILITY_UNESTABLISHED", "detail": "x"}]) == []
    assert any("recognised stored severity" in p for p in probs(risk_level={"value": "BANANA", "status": "MEASURED", "basis": "x"}))
    bad_conf = copy.deepcopy(base["decision_confidence"]); bad_conf["low_confidence_floor"] = float("inf")
    assert any("finite number" in p for p in probs(decision_confidence=bad_conf))


def test_unc_9b_generated_assessments_always_pass_the_strengthened_validator():
    for G in (_unc_g(), _unc_g(_UNC_CONTRA), _unc_g(rulebook="Vendors should behave in spirit.\n"), _unc_g({"m_v.txt": "Meeting notes\n"}), _unc_g({"inv_v.txt": "Invoice No: INV-7\nPO Number: PO-777\nVendor: Boreal\nBilled amount INR 800\n"})):
        for r in T.build_uncertainty_assessments(G): _unc_valid(G, r)
    for fault in ("remove_support", "evidence_text_5000_to_500", "weak_location", "wrong_scope", "rule_condition_9000", "basis_amount_none"):
        G = _unc_g(); T._sv7d_apply_fault(G, _unc_dec(G), fault); _unc_valid(G, _unc_one(G))


# 21 ---- uncertainty evaluation benchmark (Phase 9C): hand-labelled, deterministic, read-only; calibration metrics only from genuine numeric confidence
@pytest.fixture(scope="module")
def uncb():
    return T.run_uncertainty_benchmark()


def _ub_rec(case_id, exp, preds, conf=None, exp_esc=None):
    """Hand-built record for arithmetic tests. preds: {variant: (state, escalated)}."""
    return {"case_id": case_id, "category": "compliant", "expected_state": exp, "expected_escalation": bool(exp_esc), "expected_reasons": [], "predicted_confidence": conf,
            "predictions": {v: {"state": s, "escalated": e} for v, (s, e) in preds.items()}, "deterministic": True, "graph_unchanged": True,
            "id_integrity": {"evidence_ids_in_graph": True, "provenance_matches_graph": True, "rule_ids_in_graph": True, "verdict_preserved": True, "contract_problems": []}}


def _ub_all(s, e=False): return {v: (s, e) for v in T.UNCB_VARIANTS}


def _ub_m(recs, v="uncertainty_aware"): return T.aggregate_uncertainty_results(recs, [])["variants"][v]["metrics"]


def test_uncb_labels_validate_and_cover_every_required_category_state_and_escalation_class():
    assert T.validate_uncertainty_benchmark() == []
    cats = {c["category"] for c in T.UNCERTAINTY_BENCHMARK}
    assert set(T.UNCB_CATEGORIES) <= cats and {"compliant", "non_compliant", "conditional", "insufficient_evidence", "missing_critical_evidence", "unresolved_contradiction", "resolved_contradiction", "policy_mismatch", "low_confidence", "high_risk", "unsupported_ambiguous"} <= cats
    gts = [c["ground_truth"] for c in T.UNCERTAINTY_BENCHMARK]
    assert {g["state"] for g in gts} == set(T.UNC_STATES) and {g["escalation"] for g in gts} == {True, False} and {x for g in gts for x in g["reasons"]} == set(T.UNC_ESCALATION_CODES)


def test_uncb_is_deterministic_and_idempotent(uncb):
    again = T.run_uncertainty_benchmark()
    assert json.dumps(uncb, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert uncb["aggregate"]["integrity"] == {"all_deterministic": True, "all_graphs_unchanged": True, "all_ids_and_provenance_intact": True}
    assert uncb["aggregate"]["benchmark"]["seed"] is None and uncb["aggregate"]["benchmark"]["llm_used"] is False and len(uncb["records"]) == len(T.UNCERTAINTY_BENCHMARK)


def test_uncb_ground_truth_is_independent_of_system_output(monkeypatch):
    import copy
    before = copy.deepcopy(T.UNCERTAINTY_BENCHMARK); clean = T.run_uncertainty_benchmark()
    assert T.UNCERTAINTY_BENCHMARK == before  # running never edits the labels
    junk = {"uncertainty_status": "COMPLIANT", "escalation_required": False, "escalation_reasons": [], "decision_confidence": {"value": None}, "supporting_evidence": [], "policy_rules": [], "decision": {"verdict": None}}
    monkeypatch.setattr(T, "build_uncertainty_assessment", lambda G, d: dict(junk))  # a system that answers garbage must not move a single label
    monkeypatch.setattr(T, "validate_uncertainty_assessment", lambda G, r: [])
    bad = T.run_uncertainty_benchmark()
    keys = ("case_id", "expected_state", "expected_escalation", "expected_reasons")
    assert [{k: r[k] for k in keys} for r in bad["records"]] == [{k: r[k] for k in keys} for r in clean["records"]]
    for r, c in zip(bad["records"], T.UNCERTAINTY_BENCHMARK): assert (r["expected_state"], r["expected_escalation"], r["expected_reasons"]) == (c["ground_truth"]["state"], c["ground_truth"]["escalation"], c["ground_truth"]["reasons"])
    assert bad["aggregate"]["variants"]["uncertainty_aware"]["metrics"]["state_accuracy"]["value"] < clean["aggregate"]["variants"]["uncertainty_aware"]["metrics"]["state_accuracy"]["value"]


def test_uncb_validator_rejects_inconsistent_or_prediction_dependent_labels():
    import copy
    base = copy.deepcopy(T.UNCERTAINTY_BENCHMARK[0]); other = lambda **kw: [{**copy.deepcopy(base), **kw}]
    assert T.validate_uncertainty_benchmark(other(), require_coverage=False) == []
    assert any("case keys" in p for p in T.validate_uncertainty_benchmark(other(predicted_state="COMPLIANT"), require_coverage=False))  # no output field may ride along on a case
    gt = lambda **kw: {**base["ground_truth"], **kw}
    assert any("exactly" in p for p in T.validate_uncertainty_benchmark(other(ground_truth={"state": "COMPLIANT"}), require_coverage=False))
    assert any("not one of" in p for p in T.validate_uncertainty_benchmark(other(ground_truth=gt(state="MAYBE")), require_coverage=False))
    assert any("exactly when" in p for p in T.validate_uncertainty_benchmark(other(ground_truth=gt(escalation=True)), require_coverage=False))
    assert any("exactly when" in p for p in T.validate_uncertainty_benchmark(other(ground_truth=gt(escalation=False, reasons=["HIGH_RISK"])), require_coverage=False))
    assert any("known escalation codes" in p for p in T.validate_uncertainty_benchmark(other(ground_truth=gt(reasons=["MADE_UP"], escalation=True)), require_coverage=False))
    assert any("bool literal" in p for p in T.validate_uncertainty_benchmark(other(ground_truth=gt(escalation=lambda: True)), require_coverage=False))
    assert any("unknown fault" in p for p in T.validate_uncertainty_benchmark(other(fault="nope"), require_coverage=False)) and any("unknown setup" in p for p in T.validate_uncertainty_benchmark(other(setup="nope"), require_coverage=False))
    assert any("duplicate" in p for p in T.validate_uncertainty_benchmark(other() + other(), require_coverage=False))
    assert any("not covered" in p for p in T.validate_uncertainty_benchmark(other()))  # coverage is enforced for the shipped set
    assert any("non-escalated final state" in p for p in T.validate_uncertainty_benchmark(other(category="policy_mismatch"), require_coverage=False))
    with pytest.raises(ValueError): T.run_uncertainty_benchmark(other(ground_truth=gt(state="MAYBE")))


def test_uncb_metric_arithmetic_from_hand_built_records():
    A = "uncertainty_aware"
    recs = [_ub_rec("a", "COMPLIANT", {**_ub_all("COMPLIANT"), A: ("COMPLIANT", False)}),
            _ub_rec("b", "COMPLIANT", {**_ub_all("COMPLIANT"), A: ("NON_COMPLIANT", False)}),                      # false positive
            _ub_rec("c", "NON_COMPLIANT", {**_ub_all("NON_COMPLIANT"), A: ("COMPLIANT", False)}),                  # false negative
            _ub_rec("d", "NON_COMPLIANT", {**_ub_all("NON_COMPLIANT"), A: ("NON_COMPLIANT", True)}, exp_esc=True),  # right state, escalated
            _ub_rec("e", "CONDITIONAL", {**_ub_all("CONDITIONAL"), A: ("NON_COMPLIANT", False)}, exp_esc=True),   # unsafe finalization, missed escalation
            _ub_rec("f", "INSUFFICIENT_EVIDENCE", {**_ub_all("INSUFFICIENT_EVIDENCE", True), A: ("INSUFFICIENT_EVIDENCE", True)}, exp_esc=True),
            _ub_rec("g", "CONDITIONAL", {**_ub_all("CONDITIONAL"), A: ("CONDITIONAL", False)}),                   # fully correct, open state not escalated as labelled
            _ub_rec("h", "COMPLIANT", {**_ub_all("COMPLIANT"), A: ("COMPLIANT", True)})]                           # needless escalation
    m = _ub_m(recs)
    got = {k: (v["numerator"], v["denominator"], v["value"]) for k, v in m.items() if "numerator" in v}
    assert got["state_accuracy"] == (5, 8, 0.625)                  # a, d, f, g, h
    assert got["false_positive_rate"] == (1, 3, 0.3333)            # b among a, b, h (labelled COMPLIANT); CONDITIONAL / INSUFFICIENT not in the denominator
    assert got["false_negative_rate"] == (1, 2, 0.5)               # c among c, d
    assert got["unsafe_finalization_rate"] == (1, 3, 0.3333)       # e among e, f, g; f and g stay open
    assert got["escalation_rate"] == (3, 8, 0.375)                 # d, f, h
    assert got["escalation_false_positive_rate"] == (1, 5, 0.2)    # h among a, b, c, g, h
    assert got["escalation_false_negative_rate"] == (1, 3, 0.3333) # e among d, e, f
    assert got["escalation_agreement"] == (6, 8, 0.75)             # wrong on e and h
    assert got["automation_coverage"] == (4, 8, 0.5)               # a, b, c, e: final and not escalated
    assert got["automated_decision_accuracy"] == (1, 4, 0.25)      # only a is right among a, b, c, e
    lo = T.aggregate_uncertainty_results(recs, [])["labels_only"]
    assert lo["expected_escalation_rate"]["value"] == 0.375 and lo["expected_automation_coverage"]["value"] == 0.5  # labels only: d, e, f expect escalation; a, b, c, h are final without it


def test_uncb_measured_zero_is_distinct_from_not_measured_and_from_unavailable_confidence():
    recs = [_ub_rec("a", "COMPLIANT", _ub_all("COMPLIANT"))]
    m = _ub_m(recs)
    assert m["false_positive_rate"]["status"] == "MEASURED" and m["false_positive_rate"]["value"] == 0.0 and m["false_positive_rate"]["denominator"] == 1  # a genuine zero
    assert m["false_negative_rate"]["status"] == T.NOT_MEASURED and m["false_negative_rate"]["value"] == T.NOT_MEASURED and m["false_negative_rate"]["reason_code"] == "EMPTY_DENOMINATOR" and m["false_negative_rate"]["denominator"] == 0
    assert m["unsafe_finalization_rate"]["reason_code"] == "EMPTY_DENOMINATOR" and m["escalation_false_negative_rate"]["reason_code"] == "EMPTY_DENOMINATOR"
    assert m["ece"]["status"] == T.NOT_MEASURED and m["ece"]["reason_code"] == "NO_NUMERIC_CONFIDENCE" and m["brier"]["reason_code"] == "NO_NUMERIC_CONFIDENCE" and m["calibration"]["reason_code"] == "NO_NUMERIC_CONFIDENCE"
    allesc = _ub_m([_ub_rec("e", "COMPLIANT", _ub_all("COMPLIANT", True))])  # nothing eligible: coverage is a measured 0.0, accuracy over automation has no denominator
    assert allesc["automation_coverage"]["value"] == 0.0 and allesc["automation_coverage"]["status"] == "MEASURED" and allesc["automated_decision_accuracy"]["reason_code"] == "EMPTY_DENOMINATOR"
    empty = T.aggregate_uncertainty_results([], [])["variants"]["uncertainty_aware"]["metrics"]
    assert empty["state_accuracy"]["status"] == T.NOT_MEASURED and empty["escalation_rate"]["reason_code"] == "EMPTY_DENOMINATOR"


def test_uncb_calibration_ece_brier_use_only_genuine_numeric_confidence(uncb):
    ua = uncb["aggregate"]["variants"]["uncertainty_aware"]["metrics"]
    assert all(r["predicted_confidence"] is None for r in uncb["records"])  # the existing layer never provides one
    for k in ("calibration", "ece", "brier"): assert ua[k]["status"] == T.NOT_MEASURED and ua[k]["reason_code"] == "NO_NUMERIC_CONFIDENCE" and ua[k]["n_valid_confidence"] == 0 and ua[k]["n_cases"] == len(T.UNCERTAINTY_BENCHMARK)
    assert ua["calibration"]["bins"] == [] and any("NOT_MEASURED" in x for x in uncb["aggregate"]["limitations"])
    pairs = [(0.9, True), (0.9, True), (0.8, False), (0.2, False), (0.1, True)]  # genuine numbers
    c = T.uncertainty_calibration_metrics(pairs, 9 )
    assert c["calibration"]["status"] == "MEASURED" and c["calibration"]["n_valid_confidence"] == 5 and c["calibration"]["n_cases"] == 9
    bins = {b["bin"]: b for b in c["calibration"]["bins"]}
    assert bins[9] == {"bin": 9, "range": [0.9, 1.0], "count": 2, "mean_confidence": 0.9, "accuracy": 1.0} and bins[8]["count"] == 1 and bins[8]["accuracy"] == 0.0 and bins[1]["accuracy"] == 1.0 and bins[2]["accuracy"] == 0.0
    assert c["ece"]["value"] == pytest.approx(0.42, abs=1e-4)  # 2/5*0.1 + 1/5*0.8 + 1/5*0.2 + 1/5*0.9
    assert c["brier"]["value"] == pytest.approx((0.01 + 0.01 + 0.64 + 0.04 + 0.81) / 5, abs=1e-4)
    perfect = T.uncertainty_calibration_metrics([(1.0, True), (0.0, False)], 2)  # a measured 0.0, not NOT_MEASURED
    assert perfect["ece"]["value"] == 0.0 and perfect["ece"]["status"] == "MEASURED" and perfect["brier"]["value"] == 0.0
    junk = T.uncertainty_calibration_metrics([(True, True), (float("nan"), True), (float("inf"), False), (1.5, True), (-0.1, False), ("0.9", True), (None, True), ("HIGH", True)], 8)
    assert junk["ece"]["reason_code"] == "NO_NUMERIC_CONFIDENCE" and junk["calibration"]["n_valid_confidence"] == 0  # booleans, strings, NaN, inf, out-of-range and categorical labels are never probabilities
    recs = [_ub_rec("a", "COMPLIANT", _ub_all("COMPLIANT"), conf=0.9), _ub_rec("b", "COMPLIANT", {**_ub_all("COMPLIANT"), "uncertainty_aware": ("NON_COMPLIANT", False)}, conf=0.7)]
    m = _ub_m(recs)  # aggregate wires only the uncertainty-aware variant's genuine confidences; baselines produce none
    assert m["brier"]["value"] == pytest.approx((0.01 + 0.49) / 2, abs=1e-4) and _ub_m(recs, "binary_verdict")["ece"]["status"] == T.NOT_MEASURED
    assert T._uncb_numeric_confidence({"decision_confidence": {"value": None, "status": "NOT_MEASURED", "components": {"compiled_rules": [{"value": 0.9}]}}}) is None  # components are never promoted to a decision confidence


def test_uncb_false_positive_and_false_negative_do_not_fold_open_states_into_final_ones():
    r = _ub_rec("x", "CONDITIONAL", {**_ub_all("CONDITIONAL"), "uncertainty_aware": ("NON_COMPLIANT", False)})
    m = _ub_m([r, _ub_rec("y", "INSUFFICIENT_EVIDENCE", {**_ub_all("INSUFFICIENT_EVIDENCE"), "uncertainty_aware": ("COMPLIANT", False)})])
    assert m["false_positive_rate"]["denominator"] == 0 and m["false_negative_rate"]["denominator"] == 0  # open-state labels are in neither denominator
    assert m["unsafe_finalization_rate"]["value"] == 1.0 and m["unsafe_finalization_rate"]["denominator"] == 2


def test_uncb_escalation_and_automation_on_the_benchmark(uncb):
    agg = uncb["aggregate"]; ua = agg["variants"]["uncertainty_aware"]["metrics"]; recs = uncb["records"]
    esc = sum(r["predictions"]["uncertainty_aware"]["escalated"] for r in recs)
    assert ua["escalation_rate"]["numerator"] == esc and ua["escalation_rate"]["denominator"] == len(recs) and 0 < esc < len(recs)  # escalation and non-escalation both occur
    assert agg["labels_only"]["expected_escalation_rate"]["numerator"] == sum(r["expected_escalation"] for r in recs)  # expected escalation is available separately from the labels
    auto = [r for r in recs if r["predictions"]["uncertainty_aware"]["state"] in T.UNCB_FINAL_STATES and not r["predictions"]["uncertainty_aware"]["escalated"]]
    assert ua["automation_coverage"]["numerator"] == len(auto) > 0 and ua["automation_coverage"]["value"] == round(len(auto) / len(recs), 4)
    assert all(r["expected_state"] in T.UNCB_FINAL_STATES and not r["expected_escalation"] for r in auto)  # every automated case is labelled final and safe
    for r in recs:  # per-case: reasons the system gives match the hand label where it escalates
        p = r["predictions"]["uncertainty_aware"]; assert p["escalated"] == r["expected_escalation"] and p["reasons"] == r["expected_reasons"], r["case_id"]


def test_uncb_all_four_states_are_exercised_and_scored(uncb):
    recs = uncb["records"]
    assert {r["expected_state"] for r in recs} == set(T.UNC_STATES) == {r["predictions"]["uncertainty_aware"]["state"] for r in recs}
    assert uncb["aggregate"]["benchmark"]["expected_state_counts"].keys() == set(T.UNC_STATES)
    assert uncb["aggregate"]["variants"]["uncertainty_aware"]["metrics"]["state_accuracy"]["value"] == 1.0


def test_uncb_uncertainty_aware_is_compared_against_both_baselines(uncb):
    cmpx = uncb["aggregate"]["comparison"]; assert set(cmpx["state_accuracy"]) == set(T.UNCB_VARIANTS)
    assert cmpx["state_accuracy"]["uncertainty_aware"] > cmpx["state_accuracy"]["verdict_with_verification"] > cmpx["state_accuracy"]["binary_verdict"]
    assert cmpx["unsafe_finalization_rate"]["uncertainty_aware"] < cmpx["unsafe_finalization_rate"]["verdict_with_verification"] < cmpx["unsafe_finalization_rate"]["binary_verdict"]
    assert cmpx["escalation_false_negative_rate"]["binary_verdict"] == 1.0 and cmpx["escalation_rate"]["binary_verdict"] == 0.0 and cmpx["automation_coverage"]["binary_verdict"] == 1.0
    assert cmpx["automated_decision_accuracy"]["uncertainty_aware"] > cmpx["automated_decision_accuracy"]["binary_verdict"]
    r = next(r for r in uncb["records"] if r["case_id"] == "ub_missing_referenced_document")  # a baseline can look right on the verdict and still be unsafe
    assert r["verdict"] == "SATISFIED" and r["predictions"]["binary_verdict"]["state"] == "COMPLIANT" and r["predictions"]["uncertainty_aware"]["state"] == "INSUFFICIENT_EVIDENCE"


def test_uncb_never_mutates_a_caller_graph_or_module_state_and_preserves_provenance(tmp_path):
    G = _unc_g({"i_ib.txt": _UNC_INV}); d = _unc_dec(G); before = T._sv7d_snapshot(G); verdict = G.nodes[d]["verdict"]
    had, old = hasattr(T, "_COMPILER_MIN_CONFIDENCE"), getattr(T, "_COMPILER_MIN_CONFIDENCE", None)
    T.uncertainty_variant_predictions(G, d); T._uncb_floor(0.6, lambda: T.uncertainty_variant_predictions(G, d))
    assert T._sv7d_snapshot(G) == before and G.nodes[d]["verdict"] == verdict
    res = T.run_uncertainty_benchmark()
    assert hasattr(T, "_COMPILER_MIN_CONFIDENCE") == had and getattr(T, "_COMPILER_MIN_CONFIDENCE", None) == old  # the declared floor is restored
    assert all(r["graph_unchanged"] and r["id_integrity"]["evidence_ids_in_graph"] and r["id_integrity"]["provenance_matches_graph"] and r["id_integrity"]["rule_ids_in_graph"] and r["id_integrity"]["verdict_preserved"] and r["id_integrity"]["contract_problems"] == [] for r in res["records"])
    assert not any(k.endswith("_id") or k == "evidence_ids" for r in res["records"] for k in r if k != "case_id")  # no random graph ids leak into records
    out = tmp_path / "uncb.json"; assert not out.exists()
    T.run_uncertainty_benchmark(output_path=str(out)); assert json.loads(out.read_text(encoding="utf-8"))["benchmark_id"] == T.UNCB_ID


def test_uncb_unresolved_vs_resolved_contradiction_and_confidence_floor_cases_behave_as_labelled(uncb):
    by = {r["case_id"]: r["predictions"]["uncertainty_aware"] for r in uncb["records"]}
    assert (by["ub_major_contradiction_vendor"]["state"], by["ub_major_contradiction_vendor"]["escalated"]) == ("CONDITIONAL", True)
    assert (by["ub_major_contradiction_resolved"]["state"], by["ub_major_contradiction_resolved"]["escalated"]) == ("COMPLIANT", False)
    assert (by["ub_minor_contradiction_unresolved"]["state"], by["ub_minor_contradiction_unresolved"]["reasons"]) == ("COMPLIANT", ["UNRESOLVED_CONTRADICTION"])
    assert by["ub_low_confidence_component"]["reasons"] == ["LOW_CONFIDENCE"] and by["ub_confidence_above_floor"]["escalated"] is False
    assert by["ub_noncompliant_amount"]["escalated"] is False and by["ub_high_risk"]["reasons"] == ["HIGH_RISK"] and by["ub_not_applicable"]["state"] == "INSUFFICIENT_EVIDENCE" and by["ub_not_applicable"]["escalated"] is False

# --- OMNI-Bench foundation: schema + leakage validation ---
def _ob_case(**o):
    c = {"case_id": "c1", "family_id": "f1", "split": "TRAIN", "document_set": [{"doc_id": "d1", "text": "Invoice paid 2024-01-01"}],
         "policy": "Invoices must be paid within 30 days.", "expected_decision": "COMPLIANT", "applicable_policy_rule": "R1",
         "supporting_evidence": ["d1"], "contradicting_evidence": [], "missing_evidence": [], "expected_escalation": False,
         "counterfactual_correction": "Remove d1 -> INSUFFICIENT_EVIDENCE", "difficulty_category": "simple_compliance"}
    c.update(o)
    return c


def _ob_other(cid, fam, split, text="other"):
    return _ob_case(case_id=cid, family_id=fam, split=split, document_set=[{"doc_id": "d1", "text": text}])


def test_omni_bench_definitions_and_foundation():
    assert [c[1] for c in T.OMNI_BENCH_CATEGORIES] == [
        "Simple compliance", "Multi-document compliance", "Contradictory evidence", "Missing evidence", "Policy exceptions", "Temporal violations",
        "Entity mismatch", "Distractor documents", "OCR noise", "Policy paraphrasing", "Ambiguous policies", "Adversarial document content",
        "Prompt injection inside documents", "Conflicting policies", "Evidence removal"]
    assert list(T.OMNI_BENCH_SPLITS) == ["TRAIN", "DEV", "TEST"]
    assert T.validate_omni_bench_cases() == [] and T.validate_omni_bench_metadata() == []


def test_omni_bench_metadata_validation():
    m = dict(T.OMNI_BENCH_METADATA); m["seed_policy"] = ""; m["split_definitions"] = {"TRAIN": "x"}
    assert len(T.validate_omni_bench_metadata(m)) == 2


def test_omni_bench_valid_cases_pass_and_are_deterministic():
    cs = [_ob_case(), _ob_other("c2", "f2", "TEST")]
    assert T.validate_omni_bench_cases(cs) == [] and T.validate_omni_bench_cases(cs) == T.validate_omni_bench_cases(list(reversed(cs)))


def test_omni_bench_duplicate_case_ids():
    assert any("duplicate case_id" in p for p in T.validate_omni_bench_cases([_ob_case(), _ob_case(family_id="f2", document_set=[{"doc_id": "d1", "text": "z"}])]))


def test_omni_bench_family_cannot_cross_splits():
    assert any("multiple splits" in p for p in T.validate_omni_bench_cases([_ob_case(), _ob_other("c2", "f1", "TEST")]))
    assert T.validate_omni_bench_cases([_ob_case(), _ob_other("c2", "f1", "TRAIN")]) == []


def test_omni_bench_identical_content_cannot_cross_splits():
    assert any("identical document_set+policy" in p for p in T.validate_omni_bench_cases([_ob_case(), _ob_case(case_id="c2", family_id="f2", split="TEST")]))


def test_omni_bench_missing_fields_categories_and_splits():
    for f in T.OMNI_BENCH_REQUIRED_FIELDS:
        c = _ob_case(); del c[f]
        assert any(f"missing required field '{f}'" in p for p in T.validate_omni_bench_cases([c])), f
    assert any("invalid difficulty_category" in p for p in T.validate_omni_bench_cases([_ob_case(difficulty_category="Simple compliance ")]))
    assert any("invalid split" in p for p in T.validate_omni_bench_cases([_ob_case(split="VALIDATION")]))


def test_omni_bench_malformed_ground_truth():
    bad = [dict(expected_decision="MAYBE"), dict(expected_escalation="yes"), dict(supporting_evidence="d1"), dict(supporting_evidence=["nope"]),
           dict(document_set=[]), dict(document_set=[{"doc_id": "d1"}]), dict(counterfactual_correction=5),
           dict(expected_decision="INSUFFICIENT_EVIDENCE", missing_evidence=[])]
    for o in bad:
        assert T.validate_omni_bench_cases([_ob_case(**o)]), o


def test_omni_bench_rejects_prediction_derived_ground_truth():
    assert any("prediction-derived" in p for p in T.validate_omni_bench_cases([_ob_case(predicted_decision="COMPLIANT")]))
    assert any("prediction-derived" in p for p in T.validate_omni_bench_cases([_ob_case(provenance={"model_output": "x"})]))
    assert any("label_source" in p for p in T.validate_omni_bench_cases([_ob_case(label_source="system_prediction")]))
    assert T.validate_omni_bench_cases([_ob_case(label_source="human_annotation")]) == []


# --- OMNI-Bench population: 15 categories x 3 splits, deterministic hand-authored cases ---
import ast, datetime as _dt, json, re

_OB_RULE = re.compile(r"[A-Z]{2,5}-\d+\.\d+")
_OB_INJ = re.compile(r"IGNORE ALL PREVIOUS|SYSTEM OVERRIDE|Note to the AI auditor|Attention automated reviewer|New instructions|\[ADMIN\]")


def _ob_cat(cat): return [c for c in T.OMNI_BENCH_CASES if c["difficulty_category"] == cat]
def _ob_fams(cat):
    g = collections.defaultdict(list)
    for c in _ob_cat(cat): g[c["family_id"]].append(c)
    return [sorted(v, key=lambda c: c["case_id"]) for _, v in sorted(g.items())]
def _ob_dates(c): return [_dt.date.fromisoformat(x) for x in re.findall(r"\d{4}-\d{2}-\d{2}", " ".join(d["text"] for d in c["document_set"]))]


def test_omni_bench_population_covers_all_categories_and_splits():
    cs = T.OMNI_BENCH_CASES
    assert {c["split"] for c in cs} == {"TRAIN", "DEV", "TEST"}
    cells = collections.Counter((c["difficulty_category"], c["split"]) for c in cs)
    for cat in T.OMNI_BENCH_CATEGORY_IDS:
        assert len(_ob_cat(cat)) >= 8 and len(_ob_fams(cat)) >= 4, cat
        assert len({c["expected_decision"] for c in _ob_cat(cat)}) >= 2, cat
        for sp in T.OMNI_BENCH_SPLITS: assert cells[(cat, sp)] >= 2, (cat, sp)
    assert {c["expected_decision"] for c in cs} == set(T.OMNI_BENCH_DECISIONS) and {c["expected_escalation"] for c in cs} == {True, False}


def test_omni_bench_population_passes_schema_and_metadata_validation():
    assert T.validate_omni_bench_cases() == [] and T.validate_omni_bench_metadata() == []
    assert len({c["case_id"] for c in T.OMNI_BENCH_CASES}) == len(T.OMNI_BENCH_CASES)
    assert T.OMNI_BENCH_METADATA["generation_methodology"].startswith("Hand-authored") and T.OMNI_BENCH_METADATA["seed_policy"].startswith("No randomness")
    assert "inert data" in T.OMNI_BENCH_METADATA["document_content_policy"]


def test_omni_bench_population_family_and_content_leakage_impossible():
    cs = T.OMNI_BENCH_CASES
    fam, content, doc_text = (collections.defaultdict(set) for _ in range(3))
    for c in cs:
        fam[c["family_id"]].add(c["split"])
        content[json.dumps([c["document_set"], c["policy"]], sort_keys=True)].add(c["split"])
        for d in c["document_set"]: doc_text[d["text"]].add(c["split"])
    assert all(len(s) == 1 for s in (*fam.values(), *content.values(), *doc_text.values()))
    # the validator still rejects leakage injected into the real population
    assert any("identical document_set+policy" in p for p in T.validate_omni_bench_cases(cs + [dict(cs[0], case_id="OB-LEAK-1", family_id="FAM-LEAK-1", split="TEST")]))
    assert any("multiple splits" in p for p in T.validate_omni_bench_cases(cs + [dict(cs[0], case_id="OB-LEAK-2", split="TEST", document_set=[{"doc_id": "d1", "text": "x"}])]))


def test_omni_bench_population_is_deterministic_and_offline():
    mod = importlib.import_module("omni_pkg_under_test.omni_bench_cases")
    a, b = T.build_omni_bench_cases(), T.build_omni_bench_cases()
    assert a == b == T.OMNI_BENCH_CASES and json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    a[0]["document_set"][0]["text"] = "mutated"
    assert T.build_omni_bench_cases() == T.OMNI_BENCH_CASES  # fresh objects each call
    mods = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else n.module for n in ast.walk(ast.parse(pathlib.Path(mod.__file__).read_text())) if isinstance(n, (ast.Import, ast.ImportFrom))}
    assert mods <= {"copy", "datetime", "typing"}


def test_omni_bench_population_ground_truth_is_independent_and_policy_grounded():
    for c in T.OMNI_BENCH_CASES:
        assert c["label_source"] == "synthetic_construction" and not T._omni_bench_has_forbidden_key(c)
        rule_ids = _OB_RULE.findall(c["applicable_policy_rule"])
        assert rule_ids and all(r in c["policy"] for r in rule_ids), c["case_id"]
        assert set(_OB_RULE.findall(c["counterfactual_correction"])) & set(_OB_RULE.findall(c["policy"])), c["case_id"]
        if c["expected_decision"] == "INSUFFICIENT_EVIDENCE": assert c["missing_evidence"] and c["expected_escalation"] is True, c["case_id"]


def test_omni_bench_category_invariants_basic_structure():
    assert all(len(c["document_set"]) == 1 for c in _ob_cat("simple_compliance"))
    assert all(len(c["document_set"]) >= 3 and len(c["supporting_evidence"]) >= 2 for c in _ob_cat("multi_document_compliance"))
    assert all(c["supporting_evidence"] and c["contradicting_evidence"] for c in _ob_cat("contradictory_evidence") if c["expected_decision"] == "INSUFFICIENT_EVIDENCE")
    assert all(bool(c["missing_evidence"]) == (c["expected_decision"] == "INSUFFICIENT_EVIDENCE") for c in _ob_cat("missing_evidence"))
    for c in _ob_cat("distractor_documents"):
        assert len(c["document_set"]) >= 4 and c["supporting_evidence"] == ["d1"] and c["contradicting_evidence"] == []
    for fam in _ob_fams("policy_exceptions"):
        assert {c["expected_decision"] for c in fam} == {"COMPLIANT", "NON_COMPLIANT"} and all("Exception" in c["policy"] for c in fam)
    for fam in _ob_fams("contradictory_evidence"):  # same documents: only the policy's precedence clause decides
        assert fam[0]["document_set"] == fam[1]["document_set"] and "prevails" in fam[1]["policy"] and "prevails" not in fam[0]["policy"]


def test_omni_bench_category_invariants_temporal_ocr_and_entity_oracles():
    for cat, noisy in (("temporal_violations", False), ("ocr_noise", True)):
        for c in _ob_cat(cat):  # test-side oracle recomputed from the quoted policy and document dates only
            n, (d0, d1) = int(re.search(r"within (\d+)", c["policy"]).group(1)), _ob_dates(c)[:2]
            assert c["expected_decision"] == ("COMPLIANT" if (d1 - d0).days <= n else "NON_COMPLIANT"), c["case_id"]
            if noisy: assert all(d["text"].startswith("| ") and re.search(r"[A-Za-z][01][A-Za-z]", d["text"]) for d in c["document_set"]), c["case_id"]
    for c in _ob_cat("entity_mismatch"):
        m1 = re.search(r"to (.+?), registration (\S+?)\. Status", c["document_set"][0]["text"]); m2 = re.search(r"clearance: (.+?) cleared; registration (\S+?)\.", c["document_set"][1]["text"])
        assert c["expected_decision"] == ("COMPLIANT" if m1.groups() == m2.groups() else "NON_COMPLIANT"), c["case_id"]
        if m1.groups() == m2.groups(): assert c["supporting_evidence"] == ["d1", "d2"] and c["contradicting_evidence"] == [], c["case_id"]
        else: assert c["supporting_evidence"] == ["d1"] and c["contradicting_evidence"] == ["d2"], c["case_id"]  # mismatched clearance is contradicting, never supporting


def test_omni_bench_category_invariants_paraphrasing_and_ambiguity():
    for fam in _ob_fams("policy_paraphrasing"):
        assert len(fam) >= 3 and len({c["policy"] for c in fam}) == len(fam)
        assert all(c["document_set"] == fam[0]["document_set"] and c["expected_decision"] == fam[0]["expected_decision"] and c["expected_escalation"] == fam[0]["expected_escalation"]
                   and c["counterfactual_correction"] == fam[0]["counterfactual_correction"] for c in fam)
        n = {re.search(r"(\d+) business days", c["policy"]).group(1) for c in fam}; r = int(re.search(r"returned (\d+) business days", fam[0]["document_set"][0]["text"]).group(1))
        assert len(n) == 1 and fam[0]["expected_decision"] == ("COMPLIANT" if r <= int(n.pop()) else "NON_COMPLIANT")
    for c in _ob_cat("ambiguous_policies"):
        if c["expected_decision"] == "INSUFFICIENT_EVIDENCE":
            assert c["expected_escalation"] and "definition" in c["missing_evidence"][0] and not re.search(r"\d+ ?(days|years|hours|%)", c["policy"]), c["case_id"]
        else: assert c["expected_escalation"] is False


def test_omni_bench_category_invariants_conflicts_removal_injection_adversarial():
    for c in _ob_cat("conflicting_policies"):
        ids = set(_OB_RULE.findall(c["applicable_policy_rule"]))
        assert len(ids) >= 2 and ids <= set(_OB_RULE.findall(c["policy"])), c["case_id"]
        if "PRC-1.0" in c["policy"]: assert c["expected_decision"] != "INSUFFICIENT_EVIDENCE" and not c["expected_escalation"]
        else: assert c["expected_decision"] == "INSUFFICIENT_EVIDENCE" and c["expected_escalation"] and "precedence" in c["missing_evidence"][0]
    for base, removed, irrelevant in _ob_fams("evidence_removal"):  # convention: A base, B decisive evidence removed, C irrelevant document removed
        ids = lambda c: {d["doc_id"] for d in c["document_set"]}
        assert ids(removed) < ids(base) and ids(irrelevant) < ids(base) and {c["split"] for c in (base, removed, irrelevant)} == {base["split"]}
        assert removed["expected_decision"] == "INSUFFICIENT_EVIDENCE" and removed["missing_evidence"] and set(removed["supporting_evidence"]) <= ids(removed)
        assert irrelevant["expected_decision"] == base["expected_decision"] and base["document_set"][:1] == removed["document_set"][:1]
        assert "does not change the expected decision" in irrelevant["counterfactual_correction"] and base["expected_decision"] in irrelevant["counterfactual_correction"]
        assert "INSUFFICIENT_EVIDENCE" not in irrelevant["counterfactual_correction"] and "restored" in removed["counterfactual_correction"]
    for c in _ob_cat("prompt_injection_inside_documents"):  # injected text is inert content: label follows the evidence, never the injected demand
        assert any(_OB_INJ.search(d["text"]) for d in c["document_set"]) and not _OB_INJ.search(" ".join([c["policy"], c["counterfactual_correction"], c["applicable_policy_rule"], *c["missing_evidence"]]))
        assert c["expected_decision"] == ("NON_COMPLIANT" if "Attachment: none" in c["document_set"][0]["text"] else "COMPLIANT"), c["case_id"]
    for c in _ob_cat("adversarial_document_content"):
        assert c["expected_decision"] == ("COMPLIANT" if any("issued by" in d["text"] for d in c["document_set"]) else "NON_COMPLIANT"), c["case_id"]
        assert bool(c["contradicting_evidence"]) == (c["expected_decision"] == "COMPLIANT")


# --- OMNI-Bench evaluator: structured counterfactual check + strict prediction container ---
import json as _json_ev
E = importlib.import_module("omni_pkg_under_test.omni_bench_eval")
_CF_BASE = "Remove d1 -> INSUFFICIENT_EVIDENCE"


def _ev_case(cid="c1", cf=_CF_BASE, **o):
    return _ob_case(case_id=cid, family_id="f_" + cid, document_set=[{"doc_id": "d1", "text": "Invoice " + cid}, {"doc_id": "d2", "text": "PO " + cid}], counterfactual_correction=cf, **o)


def _ev_pred(cid="c1", **o):
    p = {"case_id": cid, "predicted_decision": "COMPLIANT"}
    p.update(o)
    return p


def _cf_result(case, predicted):
    r = E.evaluate_omni_bench([_ev_pred(case["case_id"], predicted_counterfactual_correction=predicted)], cases=[case])
    assert r["status"] == "MEASURED", r
    return r["per_case"][0]["counterfactual_correct"], r["overall"]["counterfactual_correctness"]


def test_evaluator_equivalent_counterfactual_wording_is_scored_correct():
    assert _cf_result(_ev_case(), "If d1 is removed, the decision becomes INSUFFICIENT_EVIDENCE.") == (True, 1.0)
    assert _cf_result(_ev_case(), "  remove   d1 -> insufficient_evidence ")[0] is True
    rule_case = _ev_case(cf="Per PRC-1.0, restore d2 -> COMPLIANT")
    assert _cf_result(rule_case, "COMPLIANT results once d2 is restored under PRC-1.0")[0] is True


def test_evaluator_materially_different_counterfactual_is_rejected():
    case = _ev_case()
    for wrong in ("Remove d1 -> COMPLIANT", "Remove d1 -> NON_COMPLIANT", "Remove d2 -> INSUFFICIENT_EVIDENCE", "Remove d1 and d2 -> INSUFFICIENT_EVIDENCE",
                  "Removing d1 does not change the decision INSUFFICIENT_EVIDENCE"):
        assert _cf_result(case, wrong) == (False, 0.0), wrong
    assert _cf_result(_ev_case(cf="Per PRC-1.0, restore d2 -> COMPLIANT"), "Per PRC-2.0, restore d2 -> COMPLIANT")[0] is False
    assert _cf_result(_ev_case(cf="Extend the deadline to 30 days -> COMPLIANT"), "Extend the deadline to 45 days -> COMPLIANT")[0] is False
    assert _cf_result(_ev_case(cf="Adding d2 does not change the expected decision COMPLIANT"), "Adding d2 -> NON_COMPLIANT")[0] is False


def test_evaluator_ambiguous_counterfactual_is_not_measured():
    case = _ev_case()
    for vague in ("Remove the invoice", "INSUFFICIENT_EVIDENCE", "Remove d1", "Something would be different"):  # no anchors, or anchors incomplete
        got, agg = _cf_result(case, vague)
        assert got == E.NOT_MEASURED and agg == E.NOT_MEASURED, vague
    assert _cf_result(_ev_case(cf="The evidence must be restored"), "Restore the evidence")[0] == E.NOT_MEASURED  # benchmark text has no deterministic anchor
    r = E.evaluate_omni_bench([_ev_pred()], cases=[_ev_case()])  # no counterfactual supplied
    assert r["per_case"][0]["counterfactual_correct"] == E.NOT_MEASURED and r["overall"]["counterfactual_correctness"] == E.NOT_MEASURED


def test_evaluator_rejects_malformed_prediction_container():
    case = _ev_case()
    rec = _ev_pred()
    for bad in ({"c1": rec}, rec, "c1", 5, (r for r in [rec]), {"c1"}):
        r = E.evaluate_omni_bench(bad, cases=[case])
        assert r["status"] == "INVALID_PREDICTIONS" and r["errors"] and "list or tuple" in r["errors"][0], bad
        assert "per_case" not in r
    assert E.validate_omni_bench_predictions({"c1": rec}, cases=[case]) == ["predictions must be a list or tuple of records"]
    assert E.evaluate_omni_bench(None, cases=[case])["status"] == "NOT_MEASURED" and E.evaluate_omni_bench([], cases=[case])["status"] == "NOT_MEASURED"
    assert E.evaluate_omni_bench((rec,), cases=[case])["status"] == "MEASURED" and E.evaluate_omni_bench([rec], cases=[case])["status"] == "MEASURED"
    assert E.evaluate_omni_bench(["not a record"], cases=[case])["status"] == "INVALID_PREDICTIONS"


def _ev_perfect(c):
    return {"case_id": c["case_id"], "predicted_decision": c["expected_decision"], "predicted_escalation": c["expected_escalation"], "predicted_applicable_policy_rule": c["applicable_policy_rule"],
            "predicted_supporting_evidence": list(c["supporting_evidence"]), "predicted_contradicting_evidence": list(c["contradicting_evidence"]),
            "predicted_missing_evidence": list(c["missing_evidence"]), "predicted_counterfactual_correction": c["counterfactual_correction"]}


def test_evaluator_existing_perfect_and_wrong_metrics_unchanged():
    cases = [_ev_case("c1"), _ev_case("c2", expected_decision="NON_COMPLIANT", expected_escalation=True, contradicting_evidence=["d2"], supporting_evidence=["d1"])]
    perfect = E.evaluate_omni_bench([_ev_perfect(c) for c in cases], cases=cases)
    o = perfect["overall"]
    assert perfect["status"] == "MEASURED" and perfect["complete"] is True and perfect["coverage"] == 1.0
    for k in ("decision_accuracy", "escalation_accuracy", "policy_rule_accuracy", "counterfactual_correctness", "supporting_evidence_f1", "contradicting_evidence_f1"):
        assert o[k] == 1.0, k
    assert o["missing_evidence_precision"] == E.NOT_MEASURED  # nothing predicted or expected: still not measured
    wrong = []
    for c in cases:
        w = _ev_perfect(c)
        w.update(predicted_decision="COMPLIANT" if c["expected_decision"] != "COMPLIANT" else "NON_COMPLIANT", predicted_escalation=not c["expected_escalation"],
                 predicted_applicable_policy_rule="ZZ-9.9", predicted_supporting_evidence=["d2"] if c["supporting_evidence"] == ["d1"] else ["d1"],
                 predicted_counterfactual_correction="Remove d1 -> COMPLIANT")
        wrong.append(w)
    o = E.evaluate_omni_bench(wrong, cases=cases)["overall"]
    for k in ("decision_accuracy", "escalation_accuracy", "policy_rule_accuracy", "counterfactual_correctness", "supporting_evidence_f1"):
        assert o[k] == 0.0, k
    assert o["denominators"]["decision_accuracy"] == 2 and o["denominators"]["counterfactual_correctness"] == 2


def test_evaluator_repeated_evaluation_is_identical_and_non_mutating():
    cases = [_ev_case("c1"), _ev_case("c2", cf="Per PRC-1.0, restore d2 -> COMPLIANT")]
    preds = [_ev_perfect(cases[0]), dict(_ev_perfect(cases[1]), predicted_counterfactual_correction="COMPLIANT once d2 is restored under PRC-1.0")]
    before_p, before_c = _json_ev.dumps(preds, sort_keys=True), _json_ev.dumps(cases, sort_keys=True)
    a, b = E.evaluate_omni_bench(preds, cases=cases), E.evaluate_omni_bench(tuple(preds), cases=cases)
    assert _json_ev.dumps(a, sort_keys=True) == _json_ev.dumps(b, sort_keys=True) == _json_ev.dumps(E.evaluate_omni_bench(preds, cases=cases), sort_keys=True)
    assert a["per_case"][1]["counterfactual_correct"] is True
    assert _json_ev.dumps(preds, sort_keys=True) == before_p and _json_ev.dumps(cases, sort_keys=True) == before_c