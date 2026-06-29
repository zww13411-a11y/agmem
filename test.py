#!/usr/bin/env python3
"""End-to-end test and demo for AgMem v2.

Runs the full pipeline:
  1. Create an in-memory SQLite database.
  2. Write several memory nodes with diverse content and types.
  3. Create explicit links between them.
  4. Run retrieval queries and inspect scored paths.
  5. Verify weight dynamics (traversal updates).
  6. Verify entity extraction on varied input.
  7. Verify link pruning after decay.
"""

from __future__ import annotations

import time
from pprint import pprint

from agmem import (
    AgMemDB,
    extract_entities,
    on_traverse,
    daily_decay,
    retrieve,
    write_memory,
    RecallTier,
    walk_graph,
    score_path,
    run_inspection,
    apply_prune,
    InspectionReport,
    run_consolidation,
    rebuild_fts,
    ConsolidationReport,
    generate_profile,
    generate_summary_line,
)
from agmem.core import ScoredPath


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def test_entity_extraction() -> None:
    section("Entity Extraction (EN)")

    cases = [
        # (input, expected_substrings)
        ("I like Python and Rust", ["Python", "Rust"]),
        ("I prefer PostgreSQL over MySQL", ["PostgreSQL", "MySQL"]),
        ("the FastAPI framework is great for APIs", ["FastAPI"]),
        ("I use Redis for caching", ["Redis"]),
        ("My favorite editor is Neovim", ["Neovim"]),
        ("Known for strong consistency guarantees", ["strong consistency"]),
        ("", []),
        ("just random words here", []),
    ]

    for text, expected in cases:
        entities = extract_entities(text)
        ok = all(any(e.lower().startswith(exp.lower()) for e in entities) for exp in expected)
        status = "OK" if ok or (not expected and not entities) else "MISMATCH"
        print(f"  [{status}] {text!r}  →  {entities}")
        if not ok and expected:
            print(f"         Expected any of: {expected}")


def test_entity_extraction_zh() -> None:
    section("木行 — Entity Extraction (ZH + mixed)")

    cases_zh = [
        ("我喜欢Python和Rust", ["Python", "Rust"]),
        ("我偏好PostgreSQL", ["PostgreSQL"]),
        ("Docker很好用", ["Docker"]),
        ("大模型的记忆系统架构设计", ["记忆系统"]),
        ("那个缩放的问题", ["缩放"]),
        ("", []),
    ]

    for text, expected in cases_zh:
        entities = extract_entities(text)
        # Check that at least one expected entity is present
        ok = any(exp.lower() in " ".join(e.lower() for e in entities) for exp in expected)
        status = "OK" if ok or (not expected and not entities) else "MISMATCH"
        print(f"  [{status}] {text!r}  →  {entities}")
        if not ok and expected:
            print(f"         Expected: {expected}")

    # Test synonym normalization
    section("Synonym Normalization")
    synonym_cases = [
        ("I use postgresql", "PostgreSQL"),
        ("pg database", "PostgreSQL"),
        ("数据库设计", "DB"),
        ("redis cache", "Redis"),
        ("记忆系统", "memory"),
    ]
    for text, expected_canonical in synonym_cases:
        entities = extract_entities(text)
        ok = any(expected_canonical.lower() in e.lower() for e in entities)
        status = "OK" if ok else "MISMATCH"
        print(f"  [{status}] {text!r}  →  {entities}")

    # Test implicit references
    section("Implicit References")
    implicit_cases = [
        "那个五行映射的问题",
        "刚才说的LLM架构",
        "上次的AgMem方案",
    ]
    for text in implicit_cases:
        entities = extract_entities(text)
        print(f"  {text!r}  →  {entities}")
        assert len(entities) >= 1, f"Expected >= 1 entity for {text}"


def test_write_and_retrieve() -> None:
    section("Write & Retrieve Pipeline (in-memory)")

    db = AgMemDB(":memory:")

    # --- Write memories ---
    memories = [
        ("I prefer Python for data science work", "preference"),
        ("FastAPI is great for building REST APIs", "fact"),
        ("Use PostgreSQL for production databases", "fact"),
        ("Redis works well for caching layers", "fact"),
        ("Prefer async/await over callbacks", "preference"),
        ("SQLite is suitable for local development", "fact"),
        ("I like using Docker for reproducible environments", "preference"),
    ]

    created: list = []
    for content, ntype in memories:
        node = write_memory(db, content, node_type=ntype)
        created.append(node)
        print(f"  Created node [{node.id[:8]}…]  type={node.node_type}  entities={node.entities}")

    # --- Create explicit links ---
    print()
    links = [
        (created[0].id, created[1].id, "example_of"),       # Python → FastAPI
        (created[1].id, created[2].id, "related_to"),       # FastAPI → PostgreSQL
        (created[2].id, created[3].id, "extends"),           # PostgreSQL → Redis
        (created[4].id, created[5].id, "related_to"),       # async/await → SQLite
        (created[6].id, created[1].id, "inspired_by"),       # Docker → FastAPI
    ]
    for src, tgt, rel in links:
        link = db.create_link(src, tgt, rel)
        print(f"  Linked  {src[:8]}… → {tgt[:8]}…  ({rel})")

    # --- Query: "Python data science" ---
    print("\n  Query 1: \"Python for data science\"")
    results: list[ScoredPath] = retrieve(db, "Python for data science", top_k=3)
    for i, path in enumerate(results):
        print(f"    #{i+1}  score={path.score:.4f}  path={[n.id[:8] + ':' + n.content[:40] for n in path.nodes]}")

    # --- Query: "databases and caching" ---
    print("\n  Query 2: \"databases and caching\"")
    results = retrieve(db, "databases and caching", top_k=3)
    for i, path in enumerate(results):
        print(f"    #{i+1}  score={path.score:.4f}  path={[n.id[:8] + ':' + n.content[:40] for n in path.nodes]}")

    # --- Query: "containers and deployment" ---
    print("\n  Query 3: \"containers and deployment\"")
    results = retrieve(db, "containers and deployment", top_k=3)
    for i, path in enumerate(results):
        print(f"    #{i+1}  score={path.score:.4f}  path={[n.id[:8] + ':' + n.content[:40] for n in path.nodes]}")


def test_graph_walk() -> None:
    section("Graph Walk & Path Scoring (in-memory)")

    db = AgMemDB(":memory:")

    n1 = db.create_node("n1", "Python is great", "fact", entities=["Python"])
    n2 = db.create_node("n2", "FastAPI is a Python framework", "fact", entities=["FastAPI", "Python"])
    n3 = db.create_node("n3", "REST APIs need routing", "fact", entities=["REST APIs"])

    l1 = db.create_link("n1", "n2", "related_to", 0.8)
    l2 = db.create_link("n2", "n3", "related_to", 0.6)

    paths = walk_graph(db, "n1", max_hops=2)
    print(f"  Walk from n1 ({n1.content[:30]}):")
    for p in paths:
        chain = " → ".join(f"{n.id}({n.content[:20]})" for n in p.nodes)
        print(f"    score={p.score:.4f}  {chain}")

    # Score a specific path manually.
    manual_score = score_path([n1, n2, n3], [l1, l2])
    print(f"\n  Manual score n1→n2→n3: {manual_score:.4f}")
    # Formula: entry.strength × ∏(target.strength × link.weight) × hop_decay^hops
    # n1=0.5, n2=0.5, n3=0.5, l1=0.8, l2=0.6, hop_decay=0.85
    # = 0.5 * 0.8 * 0.5 * 0.85 * 0.6 * 0.5 * 0.85 = 0.04335
    expected = 0.5 * 0.8 * 0.5 * 0.85 * 0.6 * 0.5 * 0.85
    print(f"  Expected ≈ {expected:.4f}")
    assert abs(manual_score - expected) < 1e-4, f"Score mismatch: {manual_score} != {expected}"


def test_weight_dynamics() -> None:
    section("Weight Dynamics (traversal updates & decay)")

    db = AgMemDB(":memory:")

    n1 = db.create_node("n1", "Test node A", "fact", entities=["test"])
    n2 = db.create_node("n2", "Test node B", "fact", entities=["test"])
    link = db.create_link("n1", "n2", "related_to", 0.5)

    print(f"  Initial weight: {link.weight:.4f}")

    # Neutral traversal.
    prev_weight = link.weight
    on_traverse(db, link.id, outcome_signal=0.0)
    link = db.get_link(link.id)
    assert link is not None
    assert link.weight > prev_weight, "Neutral should increase weight"
    print(f"  After neutral:  {link.weight:.4f}  (> {prev_weight:.4f})")

    # Positive feedback.
    prev_weight = link.weight
    on_traverse(db, link.id, outcome_signal=1.0)
    link = db.get_link(link.id)
    assert link is not None
    assert link.weight > prev_weight, "Positive should increase weight"
    print(f"  After positive: {link.weight:.4f}  (> {prev_weight:.4f})")

    # Negative feedback.
    prev_weight = link.weight
    on_traverse(db, link.id, outcome_signal=-1.0)
    link = db.get_link(link.id)
    assert link is not None
    assert link.weight < prev_weight, "Negative should decrease weight"
    print(f"  After negative: {link.weight:.4f}  (< {prev_weight:.4f})")

    # --- Decay test ---
    section("Decay & Pruning")
    db2 = AgMemDB(":memory:")

    # Create a low-decay node and a link.
    nn1 = db2.create_node(
        "nn1", "Perishable memory", "temp", strength=0.3, decay_rate=0.10
    )
    nn2 = db2.create_node("nn2", "Stable memory", "fact", strength=0.8, decay_rate=0.01)
    l1 = db2.create_link("nn1", "nn2", "related_to", 0.5)

    # Manually set last_traversed to 30 days ago.
    past = int(time.time()) - 30 * 86400
    db2._conn.execute(
        "UPDATE links SET last_traversed = ? WHERE id = ?", (past, l1.id)
    )
    db2._conn.commit()

    pruned = daily_decay(db2)
    link_after = db2.get_link(l1.id)

    print(f"  Link weight before: 0.5")
    print(f"  Link after 30-day decay: {'PRUNED' if link_after is None else f'{link_after.weight:.6f}'}")

    # The link should have been pruned: 0.5 * (0.9)**30 ≈ 0.5 * 0.042 ≈ 0.021 < 0.2
    assert link_after is None, f"Expected link to be pruned, but weight is {link_after.weight}"
    print(f"  Pruned: {pruned == 1} (expected True)")


def test_persistence() -> None:
    section("Persistence (file-backed SQLite)")

    import os, tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = AgMemDB(db_path)
        write_memory(db, "Python is my favorite", "preference")
        write_memory(db, "FastAPI is a Python web framework", "fact")
        db.close()

        # Re-open and verify.
        db2 = AgMemDB(db_path)
        assert db2.node_count() == 2
        nodes = db2.get_all_nodes()
        print(f"  Nodes persisted: {len(nodes)}")
        for n in nodes:
            print(f"    {n.id[:8]}…  {n.content}")
        db2.close()
    finally:
        os.unlink(db_path)


def test_regression_empty_inputs() -> None:
    section("Edge Cases / Empty Inputs")

    db = AgMemDB(":memory:")

    # Retrieve with empty query → fallback to strength-sorted.
    results = retrieve(db, "", top_k=3)
    assert isinstance(results, list)
    print(f"  Empty query → {len(results)} results (expected 0)")

    # Write with minimal content.
    node = write_memory(db, "   ", node_type="fact")
    assert node is not None
    print(f"  Whitespace-only content → node created, 0 entities")

    # Retrieve with gibberish — should fall back to strength-sorted nodes.
    results = retrieve(db, "xyzzxq bzzzt", top_k=3)
    assert isinstance(results, list)
    print(f"  Gibberish query → {len(results)} results (fallback, no entities found)")

    # Link to non-existent target → should raise.
    try:
        db.create_link("nonexistent", "also-nonexistent", "related_to")
        print("  Link to non-existent nodes → DID NOT RAISE (unexpected)")
    except ValueError:
        print(f"  Link to non-existent nodes → ValueError (expected)")


def test_hermes_tools() -> None:
    section("Hermes-Compat Tools")

    db = AgMemDB(":memory:")
    from agmem.retrieval import memory_recall, memory_write, memory_link

    mem = memory_write(
        db, "I like distributed systems and Raft consensus", "preference"
    )
    print(f"  memory_write → id={mem['id'][:8]}…  type={mem['node_type']}  entities={mem['entities']}")

    mem2 = memory_write(
        db, "Raft is great for fault tolerance", "fact",
        link_to=[mem["id"]],
    )
    print(f"  memory_write (with link_to) → id={mem2['id'][:8]}…")

    mem3 = memory_write(db, "Redis is good for caching", "fact")
    result = memory_link(db, mem3["id"], mem2["id"], "example_of")
    print(f"  memory_link → {result['source_id'][:8]}… → {result['target_id'][:8]}…")

    results = memory_recall(db, "distributed consensus", top_k=3)
    print(f"\n  memory_recall('distributed consensus') → {len(results)} results:")
    for r in results:
        print(f"    score={r['score']:.4f}  path={r['path']}")


def test_inspection_pass() -> None:
    section("金行 — Inspection Pass")

    db = AgMemDB(":memory:")

    # Create a mix of healthy and stale nodes.
    write_memory(db, "Python is my favorite language", "preference", strength=0.8)
    write_memory(db, "FastAPI is a Python web framework", "fact", strength=0.7)
    write_memory(db, "old experiment that didn't work", "fact", strength=0.15)
    write_memory(db, "failed approach using wrong library", "process", strength=0.2)

    # Mark some as old/cold via surgery.
    import time
    past = int(time.time()) - 60 * 86400  # 60 days ago
    db._conn.execute(
        "UPDATE nodes SET created_at = ?, last_accessed = ?, access_count = 0 "
        "WHERE content LIKE ? OR content LIKE ?",
        (past, past, "%old%", "%failed%"),
    )
    db._conn.execute(
        "UPDATE nodes SET outcome = 'FAILURE' WHERE content LIKE ?",
        ("%failed%",),
    )
    db._conn.commit()

    # Run inspection (dry run).
    report = run_inspection(db, dry_run=True)
    report.print()

    assert report.inspected_count >= 2, f"Expected >= 2 inspected, got {report.inspected_count}"
    assert len(report.prune) >= 1, f"Expected >= 1 prune, got {report.prune}"

    # Run wet and verify verdicts are written.
    report2 = run_inspection(db, dry_run=False)
    for nid in report2.prune:
        node = db.get_node(nid)
        assert node is not None
        assert node.inspection_verdict == "prune"
        print(f"  ✅ Verdict written: {nid[:12]}… → {node.inspection_verdict}")

    # Test apply_prune dry-run (should print but not delete).
    dry = apply_prune(db, report2, confirm=True)
    assert dry == 0, "Dry-run should return 0"
    print(f"  ✅ apply_prune dry-run: {len(report2.prune)} candidates listed, 0 deleted")

    # Test apply_prune wet.
    deleted = apply_prune(db, report2, confirm=False)
    assert deleted == len(report2.prune), f"Expected {len(report2.prune)} deleted, got {deleted}"
    print(f"  ✅ apply_prune wet: {deleted} nodes deleted")

    # Verify they're gone.
    for nid in report2.prune:
        assert db.get_node(nid) is None, f"Node {nid} should be gone"
    print(f"  ✅ All pruned nodes confirmed deleted")


def test_consolidation() -> None:
    section("土行 — Consolidation Pass")

    db = AgMemDB(":memory:")

    # Create duplicate-ish nodes with explicit shared entities.
    n1 = db.create_node("n1", "Python is good for data science work",
                         "preference", strength=0.7,
                         entities=["Python", "data science"])
    n2 = db.create_node("n2", "Python is great for data science work",
                         "preference", strength=0.5,
                         entities=["Python", "data science"])
    n3 = db.create_node("n3", "FastAPI is a Python web framework",
                         "fact", strength=0.6,
                         entities=["FastAPI", "Python"])
    n4 = db.create_node("n4", "Redis works well for caching",
                         "fact", strength=0.8,
                         entities=["Redis", "cache"])
    # Link n1 → n3 so we can test link transfer
    db.create_link("n1", "n3", "related_to", 0.5)

    print(f"  Before: {db.node_count()} nodes, {db.link_count()} links")

    # Dry run
    report = run_consolidation(db, dry_run=True)
    print(f"  Dry-run: {report.nodes_merged} duplicates found")

    # Wet run
    report = run_consolidation(db, dry_run=False)
    report.print()

    assert report.nodes_merged >= 1, f"Expected >= 1 merge, got {report.nodes_merged}"
    print(f"  After:  {db.node_count()} nodes (merged {report.nodes_merged})")
    assert report.nodes_deleted >= 1

    # Verify FTS5 still works after consolidation
    fts_results = db.fts5_search("python", limit=10)
    print(f"  FTS5 'python': {len(fts_results)} results")
    assert len(fts_results) >= 1

    # Verify link was transferred
    links = db.get_links(node_id="n1")
    assert any(l.relation == "related_to" for l in links), "Links should survive merge"
    print(f"  ✅ Link transfer: n1 → n3 survived consolidation")


def test_fts5_search() -> None:
    section("土行 — FTS5 Full-Text Search")

    db = AgMemDB(":memory:")
    write_memory(db, "Python is great for data science", "fact")
    write_memory(db, "FastAPI is a Python web framework", "fact")
    write_memory(db, "Redis is a caching layer", "fact")
    write_memory(db, "PostgreSQL is a relational database", "fact")
    write_memory(db, "Docker containers for reproducible environments", "fact")

    # Simple term search
    results = db.fts5_search("python", limit=5)
    assert len(results) == 2, f"Expected 2, got {len(results)}: {[r.content for r in results]}"
    print(f"  'python' → {len(results)} results ✅")

    # Prefix wildcard
    results = db.fts5_search("data*", limit=5)
    assert len(results) >= 1
    print(f"  'data*' → {len(results)} results ✅")

    # No match
    results = db.fts5_search("nonexistent_xyz", limit=5)
    assert len(results) == 0
    print(f"  'nonexistent_xyz' → 0 results ✅")

    # Empty query
    results = db.fts5_search("", limit=5)
    assert len(results) == 0
    print(f"  '' → 0 results ✅")

    # Rebuild FTS
    count = rebuild_fts(db)
    assert count == 5
    print(f"  rebuild_fts → {count} nodes re-indexed ✅")


def test_tiered_recall() -> None:
    section("火行 — Tiered Recall (温/烟/燃/炎)")

    db = AgMemDB(":memory:")

    # Create a chain: n1 → n2 → n3 → n4 (3 hops)
    n1 = db.create_node("n1", "Python is a programming language", "fact",
                         strength=0.8, entities=["Python"])
    n2 = db.create_node("n2", "FastAPI is built with Python", "fact",
                         strength=0.7, entities=["FastAPI", "Python"])
    n3 = db.create_node("n3", "REST APIs need routing logic", "fact",
                         strength=0.6, entities=["REST API"])
    n4 = db.create_node("n4", "PostgreSQL stores API data", "fact",
                         strength=0.9, entities=["PostgreSQL"])
    n5 = db.create_node("n5", "Redis caches are fast", "fact",
                         strength=0.4, entities=["Redis", "cache"])

    db.create_link("n1", "n2", "related_to", 0.8)
    db.create_link("n2", "n3", "related_to", 0.7)
    db.create_link("n3", "n4", "related_to", 0.6)
    db.create_link("n1", "n5", "related_to", 0.3)

    # 温 tier: FTS5 only, 0 hops
    warm_results = retrieve(db, "python", tier=RecallTier.WARM, top_k=5)
    print(f"  温 'python': {len(warm_results)} results")
    for p in warm_results:
        assert len(p.nodes) == 1, f"温 should be 0 hops, got {len(p.nodes)}"
    assert len(warm_results) >= 1
    print(f"    ✅ All paths are single-node")

    # 烟 tier: 1 hop max
    smoke_results = retrieve(db, "python", tier=RecallTier.SMOKE, top_k=5)
    print(f"  烟 'python': {len(smoke_results)} results")
    for p in smoke_results:
        assert len(p.nodes) <= 2, f"烟 max 1 hop, got {len(p.nodes)}"
    assert len(smoke_results) >= 1
    print(f"    ✅ All paths ≤ 2 nodes")

    # 燃 tier: 3 hops (default) — should find multi-hop paths
    burn_results = retrieve(db, "python", tier=RecallTier.BURN, top_k=10)
    print(f"  燃 'python': {len(burn_results)} results")
    found_long = any(len(p.nodes) >= 3 for p in burn_results)
    assert found_long, "BURN tier should find ≥3-hop paths with min_path_score=0.05"
    print(f"    ✅ Long paths (≥3 nodes): {found_long}")

    # 炎 tier: deep dive
    inferno_results = retrieve(db, "python", tier=RecallTier.INFERNO, top_k=20)
    print(f"  炎 'python': {len(inferno_results)} results")
    assert len(inferno_results) >= len(burn_results), \
        f"炎 should find >= 燃 ({len(inferno_results)} vs {len(burn_results)})"
    print(f"    ✅ Inferno deeper than burn")

    # Default tier should be 燃
    default_results = retrieve(db, "python", top_k=5)
    assert len(default_results) >= 1
    print(f"  Default tier: {len(default_results)} results ✅")

    print(f"  ✅ All tier tests passed")


def test_profile() -> None:
    section("水行 — Profile Generation")

    db = AgMemDB(":memory:")

    # Empty DB → empty profile.
    profile = generate_profile(db)
    assert profile == "", f"Empty DB should give empty profile, got {profile!r}"
    print(f"  Empty DB → '' ✅")

    # Add some high-confidence preferences.
    write_memory(db, "prefers Python for backend work", "preference", strength=0.9)
    write_memory(db, "likes FastAPI over Django", "preference", strength=0.85)
    write_memory(db, "uses PostgreSQL for production", "fact", strength=0.8)
    write_memory(db, "Redis is used for caching", "fact", strength=0.75)
    write_memory(db, "decided to use async/await pattern", "decision", strength=0.7)
    write_memory(db, "tried a random library once", "fact", strength=0.2)

    profile = generate_profile(db)
    print(f"  Profile generated ({len(profile)} chars):")
    print(f"  {profile[:200]}...")
    assert len(profile) > 0, "Profile should not be empty"
    assert "Preferences" in profile, "Should include preferences"
    assert "Python" in profile, "Should include known entities"
    assert "tried a random library" not in profile, "Should exclude low-strength"
    print(f"  ✅ Profile structure correct")

    # Summary line
    summary = generate_summary_line(db)
    print(f"  Summary: {summary}")
    assert "Profile" in summary
    print(f"  ✅ Summary line generated")


def main() -> None:
    print("AgMem v2 — End-to-End Test Suite")
    print(f"  {__file__}")
    print(f"  Python {__import__('sys').version}")

    test_entity_extraction()
    test_entity_extraction_zh()
    test_graph_walk()
    test_weight_dynamics()
    test_write_and_retrieve()
    test_persistence()
    test_regression_empty_inputs()
    test_hermes_tools()
    test_inspection_pass()
    test_consolidation()
    test_fts5_search()
    test_tiered_recall()
    test_profile()

    print(f"\n{'=' * 72}")
    print("  ALL TESTS PASSED")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
