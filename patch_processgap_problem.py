"""
Replaces the problem statement in ProcessGap app.py with
industry-research-backed language. No other code is touched.

Sources used:
  IDC Research      — companies lose 20-30% of annual revenue to inefficiencies
  Crebos / McKinsey — $250K-$600K per mid-sized company annually (2024)
  Kroolo research   — 52.8% of business leaders cite long-term bottlenecks as
                      their single biggest barrier to growth
  Appian (2023)     — traditional process analysis is "lengthy, expensive,
                      and often subjective"

Usage: python patch_processgap_problem.py
Run from the same folder as your app.py.
"""

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# ── CHANGE 1: Docstring ───────────────────────────────────────────────────────
OLD_DOCSTRING_BLOCK = """\
  Process mining tools (Celonis, UiPath) cost $80K–$500K to implement.
  Management consultants charge $200K to produce the same analysis.
  Companies with 50–5,000 employees have nothing affordable."""

NEW_DOCSTRING_BLOCK = """\
  IDC research found that companies lose 20–30% of annual revenue to process
  inefficiencies. For mid-sized companies, that amounts to $250K–$600K per year
  in unrealised value lost to rework, waiting, and non-value-added steps
  (Crebos / McKinsey, 2024). 52.8% of business leaders report that unresolved
  long-term bottlenecks are their single biggest barrier to growth.

  The core problem: companies know their processes are slow but cannot identify
  which specific step costs the most, or whether fixing it is worth the investment.
  Traditional process analysis is expensive and slow to deploy. Mid-market
  operations teams have no affordable structured tool for this."""

# ── CHANGE 2: Overview module problem card ────────────────────────────────────
OLD_PROBLEM_CARD = """\
    Every company has broken processes. Most know which ones feel slow.
    What nobody knows is <strong>which specific step costs the most money</strong>
    and what fixing it is actually worth. Process mining tools (Celonis, UiPath)
    cost $80K–$500K to implement. Management consultants charge $200K for the
    same analysis. Companies between $5M and $500M in revenue have nothing.
    <br><br>
    The result: the same bottlenecks persist for years because nobody
    has quantified them well enough to justify the fix."""

NEW_PROBLEM_CARD = """\
    IDC research found that companies lose 20–30% of annual revenue to process
    inefficiencies every year. For a mid-sized organisation, that translates to
    $250K–$600K in unrealised value annually — lost to rework, waiting, and steps
    that generate no value for the customer (Crebos / McKinsey, 2024).
    <br><br>
    The problem is not that operations leaders don't know their processes are slow.
    They do. The problem is that <strong>nobody has quantified which specific step
    costs the most money</strong> — or what fixing it is actually worth. Without
    that number, there is no business case to approve the fix.
    <br><br>
    Traditional process analysis relies on consultants or enterprise software:
    slow, expensive, and built for large organisations. Mid-market teams fall back
    on spreadsheets and gut feel. The result: 52.8% of business leaders say
    unresolved long-term bottlenecks are their single biggest barrier to growth."""

# ── Apply ─────────────────────────────────────────────────────────────────────
changes = [
    (OLD_DOCSTRING_BLOCK,  NEW_DOCSTRING_BLOCK,  "Docstring problem statement"),
    (OLD_PROBLEM_CARD,     NEW_PROBLEM_CARD,     "Overview module problem card"),
]

for old, new, label in changes:
    if old in content:
        content = content.replace(old, new)
        print(f"[OK]   {label}")
    else:
        print(f"[WARN] Not found — check manually: {label}")
        print(f"       First 80 chars of target: {repr(old[:80])}")

# ── Verify no Celonis references remain ──────────────────────────────────────
remaining = content.lower().count("celonis")
print(f"\nPost-patch: 'celonis' mentions remaining = {remaining}  (target: 0)")

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done. app.py updated.")
