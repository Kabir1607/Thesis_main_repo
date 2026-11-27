import json, os
from collections import OrderedDict
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

RD = os.path.join(os.path.dirname(__file__), "results")
runs = [json.loads(l) for l in open(os.path.join(RD, "svcomp_runs.jsonl")) if l.strip()]
summary = json.load(open(os.path.join(RD, "svcomp_summary.json")))
meta = summary["metadata"]

methods = ["plain_mc", "adaptive_stratified", "pse_exact", "dise_betting"]
mname = {"plain_mc": "Plain MC", "adaptive_stratified": "Adaptive Stratified MC",
         "pse_exact": "PSE-exact", "dise_betting": "DiSE"}
mcolor = {"plain_mc": "5B7DA3", "adaptive_stratified": "3E8E7E",
          "pse_exact": "C9742E", "dise_betting": "2E7D32"}

progs = OrderedDict()
for r in runs:
    progs.setdefault(r["program"], {})[r["method"]] = r
plist = sorted(progs.keys(), key=str.lower)
N = len(plist)

FONT = "Arial"
DARK = "1F4E78"
HDR = Font(name=FONT, bold=True, color="FFFFFF", size=10)
HDR2 = Font(name=FONT, bold=True, color="FFFFFF", size=9)
BODY = Font(name=FONT, size=10)
BOLD = Font(name=FONT, bold=True, size=10)
TITLE = Font(name=FONT, bold=True, size=13, color=DARK)
NOTE = Font(name=FONT, size=9, italic=True, color="595959")
darkfill = PatternFill("solid", start_color=DARK)
thin = Side(style="thin", color="BFBFBF")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)
CTR = Alignment(horizontal="center", vertical="center", wrap_text=True)
LFT = Alignment(horizontal="left", vertical="center")

wb = Workbook()

# ---------------- Per-Program sheet ----------------
pp = wb.active
pp.title = "Per-Program"
fields = [("mu_hat", "Mu-hat", "0.000000"), ("half_width", "Half-Width", "0.000000"),
          ("samples", "Samples", "0"), ("wall_s", "Wall (s)", "0.000"),
          ("covered", "Covered", "0")]
# header row 1: groups
pp.cell(row=1, column=1, value="Program")
pp.cell(row=1, column=2, value="n_nondets")
pp.cell(row=1, column=3, value="Mu (MC truth)")
for c in range(1, 4):
    pp.merge_cells(start_row=1, start_column=c, end_row=2, end_column=c)
col = 4
for m in methods:
    pp.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + 4)
    gc = pp.cell(row=1, column=col, value=mname[m])
    gc.fill = PatternFill("solid", start_color=mcolor[m])
    gc.font = HDR
    gc.alignment = CTR
    for j, (_, lbl, _) in enumerate(fields):
        hc = pp.cell(row=2, column=col + j, value=lbl)
        hc.fill = PatternFill("solid", start_color=mcolor[m])
        hc.font = HDR2
        hc.alignment = CTR
        hc.border = BORDER
    col += 5
for c in range(1, 4):
    pp.cell(row=1, column=c).fill = darkfill
    pp.cell(row=1, column=c).font = HDR
    pp.cell(row=1, column=c).alignment = CTR
    pp.cell(row=1, column=c).border = BORDER

# data
for i, prog in enumerate(plist):
    row = 3 + i
    rec = progs[prog]
    any_r = next(iter(rec.values()))
    pp.cell(row=row, column=1, value=prog).font = BODY
    pp.cell(row=row, column=2, value=any_r.get("n_nondets")).font = BODY
    mt = pp.cell(row=row, column=3, value=any_r.get("mu_truth"))
    mt.font = BODY
    mt.number_format = "0.000000"
    col = 4
    for m in methods:
        r = rec[m]
        vals = [r.get("mu_hat"), r.get("half_width"), r.get("samples"),
                r.get("wall_s"), 1 if r.get("interval_contains_truth") else 0]
        for j, v in enumerate(vals):
            cc = pp.cell(row=row, column=col + j, value=v)
            cc.font = BODY
            cc.number_format = fields[j][2]
            cc.alignment = CTR
        col += 5
    pp.cell(row=row, column=1).alignment = LFT
    pp.cell(row=row, column=2).alignment = CTR

last = 2 + N
for r in range(1, last + 1):
    for c in range(1, 24):
        pp.cell(row=r, column=c).border = BORDER
pp.freeze_panes = "B3"
pp.column_dimensions["A"].width = 34
pp.column_dimensions["B"].width = 10
pp.column_dimensions["C"].width = 13
for c in range(4, 24):
    pp.column_dimensions[get_column_letter(c)].width = 11

# half-width / samples / wall / covered column letters per method
HW = {m: get_column_letter(5 + 5 * k) for k, m in enumerate(methods)}
SA = {m: get_column_letter(6 + 5 * k) for k, m in enumerate(methods)}
WA = {m: get_column_letter(7 + 5 * k) for k, m in enumerate(methods)}
CV = {m: get_column_letter(8 + 5 * k) for k, m in enumerate(methods)}
rng = lambda L: f"'Per-Program'!{L}3:{L}{last}"

# ---------------- Summary sheet ----------------
sm = wb.create_sheet("Summary", 0)
sm["A1"] = "DiSE — SV-COMP ReachSafety-Loops Evaluation"
sm["A1"].font = TITLE
sm.merge_cells("A1:E1")

mrows = [
    ("Benchmark suite", "SV-COMP ReachSafety-Loops"),
    ("Programs evaluated", N),
    ("Sample budget", meta.get("budget")),
    ("Confidence (1 - delta)", 1 - meta.get("delta", 0)),
    ("Random seed", meta.get("seed")),
    ("Per-program timeout (s)", meta.get("per_program_timeout")),
    ("Per-cell SMT budget (s)", meta.get("per_cell_seconds")),
    ("MC ground-truth budget", meta.get("mc_truth_budget")),
]
r = 3
for k, v in mrows:
    sm.cell(row=r, column=1, value=k).font = BOLD
    c = sm.cell(row=r, column=2, value=v)
    c.font = BODY
    if isinstance(v, float):
        c.number_format = "0.00"
    r += 1
r += 1

# per-method table
sm.cell(row=r, column=1, value="Per-Method Aggregates").font = Font(name=FONT, bold=True, size=11, color=DARK)
r += 1
hdr_row = r
sm.cell(row=r, column=1, value="Metric")
for k, m in enumerate(methods):
    sm.cell(row=r, column=2 + k, value=mname[m])
for c in range(1, 6):
    cell = sm.cell(row=hdr_row, column=c)
    cell.fill = darkfill
    cell.font = HDR
    cell.alignment = CTR
    cell.border = BORDER
r += 1

metrics = [
    ("Programs", lambda m: f"=COUNT({rng(HW[m])})", "0"),
    ("Coverage (empirical)", lambda m: f"=AVERAGE({rng(CV[m])})", "0.0%"),
    ("Median half-width", lambda m: f"=MEDIAN({rng(HW[m])})", "0.000000"),
    ("Mean half-width", lambda m: f"=AVERAGE({rng(HW[m])})", "0.000000"),
    ("P90 half-width", lambda m: f"=PERCENTILE({rng(HW[m])},0.9)", "0.000000"),
    ("Min half-width", lambda m: f"=MIN({rng(HW[m])})", "0.000000"),
    ("Max half-width", lambda m: f"=MAX({rng(HW[m])})", "0.000000"),
    ("Exact certificates (HW = 0)", lambda m: f"=COUNTIF({rng(HW[m])},0)", "0"),
    ("Fraction exact (HW = 0)", lambda m: f"=COUNTIF({rng(HW[m])},0)/COUNT({rng(HW[m])})", "0.0%"),
    ("Median samples", lambda m: f"=MEDIAN({rng(SA[m])})", "0"),
    ("Total samples", lambda m: f"=SUM({rng(SA[m])})", "#,##0"),
    ("Total wall time (s)", lambda m: f"=SUM({rng(WA[m])})", "0.0"),
    ("Mean wall time (s)", lambda m: f"=AVERAGE({rng(WA[m])})", "0.000"),
]
for label, fn, fmt in metrics:
    sm.cell(row=r, column=1, value=label).font = BOLD
    sm.cell(row=r, column=1).border = BORDER
    for k, m in enumerate(methods):
        cc = sm.cell(row=r, column=2 + k, value=fn(m))
        cc.font = Font(name=FONT, size=10, bold=(m == "dise_betting"))
        cc.number_format = fmt
        cc.alignment = CTR
        cc.border = BORDER
    r += 1
r += 1

# pairwise table
sm.cell(row=r, column=1, value="Pairwise: DiSE vs. Baseline (half-width comparison)").font = Font(name=FONT, bold=True, size=11, color=DARK)
r += 1
ph = r
for c, lbl in enumerate(["Comparison", "DiSE narrower", "Tie", "Baseline narrower", "DiSE exact (HW=0)"]):
    cell = sm.cell(row=ph, column=1 + c, value=lbl)
    cell.fill = darkfill
    cell.font = HDR
    cell.alignment = CTR
    cell.border = BORDER
r += 1
TOL = "1E-9"
dh = rng(HW["dise_betting"])
for m in ["plain_mc", "adaptive_stratified", "pse_exact"]:
    oh = rng(HW[m])
    sm.cell(row=r, column=1, value=f"DiSE vs. {mname[m]}")
    sm.cell(row=r, column=2, value=f"=SUMPRODUCT(({dh}<{oh}-{TOL})*1)")
    sm.cell(row=r, column=3, value=f"=SUMPRODUCT((ABS({dh}-{oh})<={TOL})*1)")
    sm.cell(row=r, column=4, value=f"=SUMPRODUCT(({dh}>{oh}+{TOL})*1)")
    sm.cell(row=r, column=5, value=f"=SUMPRODUCT(({dh}=0)*1)")
    for c in range(1, 6):
        cell = sm.cell(row=r, column=c)
        cell.font = BOLD if c == 1 else BODY
        cell.alignment = LFT if c == 1 else CTR
        cell.border = BORDER
        if c > 1:
            cell.number_format = "0"
    r += 1
r += 1
sm.cell(row=r, column=1, value="Half-width is the certified interval half-width; smaller is better. "
        "Coverage is the empirical fraction of programs whose interval contained the MC ground truth. "
        "Pairwise ties are counted within a 1e-9 floating-point tolerance.").font = NOTE
sm.merge_cells(start_row=r, start_column=1, end_row=r, end_column=5)
r += 1
sm.cell(row=r, column=1, value="Source: experiments/results/svcomp_runs.jsonl (seed 0). One row per (method, program) in the All-Runs sheet.").font = NOTE
sm.merge_cells(start_row=r, start_column=1, end_row=r, end_column=5)

sm.column_dimensions["A"].width = 34
for c in "BCDE":
    sm.column_dimensions[c].width = 21
sm.freeze_panes = "A2"

# ---------------- All-Runs sheet ----------------
ar = wb.create_sheet("All-Runs")
cols = ["Program", "Method", "n_nondets", "Mu (MC truth)", "Mu-hat",
        "Interval Lo", "Interval Hi", "Half-Width", "Samples", "Wall (s)",
        "Covered", "Terminated", "W_open", "n_leaves", "n_open",
        "Hits", "Closed-T", "Closed-F", "Enum. Paths", "Hit Time Limit"]
for c, h in enumerate(cols, 1):
    cell = ar.cell(row=1, column=c, value=h)
    cell.fill = darkfill
    cell.font = HDR2
    cell.alignment = CTR
    cell.border = BORDER
row = 2
for prog in plist:
    for m in methods:
        r = progs[prog][m]
        e = r.get("extras", {})
        iv = r.get("interval", [None, None])
        vals = [prog, mname[m], r.get("n_nondets"), r.get("mu_truth"), r.get("mu_hat"),
                iv[0], iv[1], r.get("half_width"), r.get("samples"), r.get("wall_s"),
                1 if r.get("interval_contains_truth") else 0,
                e.get("terminated"), e.get("W_open"), e.get("n_leaves"),
                e.get("n_open", e.get("n_open_leaves")), e.get("hits"),
                e.get("n_closed_true"), e.get("n_closed_false"),
                e.get("enumerated_paths"), e.get("hit_time_limit")]
        for c, v in enumerate(vals, 1):
            cc = ar.cell(row=row, column=c, value=v)
            cc.font = BODY
            cc.border = BORDER
            if c in (4, 5, 6, 7, 8, 13):
                cc.number_format = "0.000000"
            elif c == 10:
                cc.number_format = "0.000"
            elif c in (3, 9, 11, 14, 15, 16, 17, 18, 19):
                cc.number_format = "0"
                cc.alignment = CTR
        row += 1
ar.freeze_panes = "A2"
ar.column_dimensions["A"].width = 34
ar.column_dimensions["B"].width = 20
ar.column_dimensions["L"].width = 16
for c in [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20]:
    ar.column_dimensions[get_column_letter(c)].width = 12

wb.calculation.fullCalcOnLoad = True
out = os.path.join(RD, "svcomp_numbers.xlsx")
wb.save(out)
print("wrote", out, "| programs:", N, "| runs:", len(runs))
