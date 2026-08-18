import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
                                  PageBreak, HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

RESULTS = Path(__file__).resolve().parent.parent / "results"
CHARTS = RESULTS / "charts"
OUT = RESULTS / "RecWM_REN_Investor_Report.pdf"

NAVY = colors.HexColor("#0b0b0b")
VIOLET = colors.HexColor("#4a3aa7")
BLUE = colors.HexColor("#2a78d6")
GREY = colors.HexColor("#52514e")
LIGHT_GREY = colors.HexColor("#e3e2dc")
GREEN = colors.HexColor("#008300")
RED = colors.HexColor("#e34948")
SURFACE = colors.HexColor("#fcfcfb")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("CoverTitle", fontSize=28, leading=34, textColor=NAVY, fontName="Helvetica-Bold", spaceAfter=6))
styles.add(ParagraphStyle("CoverSub", fontSize=14, leading=19, textColor=GREY, fontName="Helvetica", spaceAfter=4))
styles.add(ParagraphStyle("H1", fontSize=17, leading=21, textColor=NAVY, fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8))
styles.add(ParagraphStyle("H2", fontSize=12.5, leading=16, textColor=VIOLET, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", fontSize=10, leading=14.5, textColor=NAVY, fontName="Helvetica", spaceAfter=6))
styles.add(ParagraphStyle("BodySmall", fontSize=8.5, leading=12, textColor=GREY, fontName="Helvetica"))
styles.add(ParagraphStyle("Caption", fontSize=9, leading=12, textColor=GREY, fontName="Helvetica-Oblique",
                            spaceAfter=14, alignment=TA_CENTER))
styles.add(ParagraphStyle("MyBullet", fontSize=10, leading=14.5, textColor=NAVY, fontName="Helvetica",
                            leftIndent=14, bulletIndent=0, spaceAfter=4))
styles.add(ParagraphStyle("StatNum", fontSize=22, leading=24, textColor=VIOLET, fontName="Helvetica-Bold", alignment=TA_CENTER))
styles.add(ParagraphStyle("StatLabel", fontSize=8.5, leading=11, textColor=GREY, fontName="Helvetica", alignment=TA_CENTER))

story = []
_target = [story]  # top of stack = where flowables currently append


def _cur(): return _target[-1]


def start_section(): _target.append([])
def end_section():
    section = _target.pop()
    # Keep the heading+intro+chart of a section glued together so a
    # heading never gets stranded alone at the bottom of a page while its
    # chart flows to the next (a real layout bug caught while proofing
    # this PDF, not a hypothetical one).
    _target[-1].append(KeepTogether(section))


def h1(text): _cur().append(Paragraph(text, styles["H1"]))
def h2(text): _cur().append(Paragraph(text, styles["H2"]))
def body(text): _cur().append(Paragraph(text, styles["Body"]))
def bullet(text): _cur().append(Paragraph(f"&bull;&nbsp;&nbsp;{text}", styles["MyBullet"]))
def hr(): _cur().append(HRFlowable(width="100%", thickness=0.75, color=LIGHT_GREY, spaceBefore=8, spaceAfter=8))
def pagebreak(): _cur().append(PageBreak())
def chart(name, caption, width=6.6):
    img = Image(str(CHARTS / name), width=width * inch, height=width * inch * 0.53)
    _cur().append(img)
    _cur().append(Paragraph(caption, styles["Caption"]))


# =====================================================================
# COVER
# =====================================================================
story.append(Spacer(1, 1.6 * inch))
story.append(Paragraph("RecWM / REN", styles["CoverTitle"]))
story.append(Paragraph("Reflexive Equilibrium Networks — Independent Implementation &amp; Evaluation", styles["CoverSub"]))
story.append(Spacer(1, 0.35 * inch))
story.append(HRFlowable(width="40%", thickness=2, color=VIOLET, spaceAfter=14, hAlign="LEFT"))
story.append(Paragraph(
    "A from-scratch, independently built and tested implementation of the ten inventions described in the "
    "RecWM thesis, benchmarked against real market data and six standard market models. Every figure in this "
    "document is measured, not a design target — including the ones that are unfavorable.",
    styles["Body"]))
story.append(Spacer(1, 2.6 * inch))
story.append(Paragraph("Prepared for investor review &middot; August 2026", styles["BodySmall"]))
story.append(Paragraph("Full technical report, code, and raw data available on request.", styles["BodySmall"]))
story.append(PageBreak())

# =====================================================================
# EXECUTIVE SUMMARY
# =====================================================================
h1("Executive Summary")
body("This report independently implemented and empirically tested all ten inventions in the RecWM/REN "
     "architecture — real code, real market data (16 liquid instruments, 2015&ndash;2026), real benchmarks. "
     "The goal was to determine, with evidence rather than assertion, which parts of the architecture work as "
     "claimed and which do not.")

# stat row
stat_data = [
    [Paragraph("2,618", styles["StatNum"]), Paragraph("REAL", styles["StatNum"]),
     Paragraph("2,316&times;", styles["StatNum"]), Paragraph("0 / 5", styles["StatNum"])],
    [Paragraph("real trading days<br/>tested end-to-end", styles["StatLabel"]),
     Paragraph("market data throughout<br/>&mdash; no synthetic substitutes", styles["StatLabel"]),
     Paragraph("verified speedup from<br/>Composition Algebra", styles["StatLabel"]),
     Paragraph("trained seeds beat<br/>simple buy-and-hold", styles["StatLabel"])],
]
t = Table(stat_data, colWidths=[1.6 * inch] * 4)
t.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, 0), 6), ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
    ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white),
]))
story.append(Spacer(1, 6))
story.append(t)
story.append(Spacer(1, 10))

h2("What holds up")
bullet("The core mathematics — the resolvent influence kernel, the DEQ fixed-point solver, Anderson "
       "acceleration, implicit-function-theorem gradients — is correctly implemented and verified against "
       "known ground truth, not just asserted to work.")
bullet("<b>Composition Algebra</b> is the strongest result: composing pre-solved scenarios approximates a full "
       "re-solve to within 2.7% error at a real, measured 2,316&times; speedup, across 100 independent tests.")
bullet("<b>Adversarial Defense</b> shows real, statistically significant signal: a crowding-risk indicator "
       "(p&nbsp;=&nbsp;2&times;10<super rise=3 size=7>-5</super>) and an 8.1&times; reduction in a standard execution-detectability metric.")
bullet("A real market-impact model answers a question the original thesis never computed: an $800M fund's own "
       "footprint is already ~24bps &mdash; a concrete number for sizing decisions.")

h2("What does not, yet")
bullet("<b>No configuration tested &mdash; trained or untrained, 25 seeds total &mdash; produced a Sharpe ratio "
       "that beats simple buy-and-hold</b> on the real 2024&ndash;2026 held-out test period, with statistically "
       "robust confidence intervals confirming this is not a fluke of one time window.")
bullet("REN ranks in the middle of six standard baseline models &mdash; ahead of naive mean-reversion and ridge "
       "regression, behind buy-and-hold, risk parity, 60/40, and a plain random forest.")
bullet("The architecture's headline unifying claim (spectral radius as a free crisis early-warning signal) "
       "moved in the wrong direction on real historical crashes when untrained, and only recovered the correct "
       "direction after being explicitly trained toward that exact target &mdash; i.e. it is a trainable "
       "feature, not a free byproduct as originally framed.")

story.append(PageBreak())

# =====================================================================
# PERFORMANCE VS MARKET MODELS
# =====================================================================
start_section()
h1("Performance vs. Standard Market Models")
body("Six real, independently implemented baseline strategies were run on identical real data, identical "
     "test period, and identical transaction-cost assumptions as every REN configuration, so the comparison is "
     "fair. Error bars are 90% block-bootstrap confidence intervals (2,000 resamples), which test whether an "
     "apparent edge is statistically robust or could be explained by the ordinary noise in one time window.")
chart("01_sharpe_comparison.png",
      "Sharpe ratio, real held-out test period (2024-07 to 2026-08, 524 trading days). Buy-and-hold and risk "
      "parity's intervals sit entirely above zero; REN's best result does not.")
end_section()

start_section()
h1("Cumulative Performance")
body("The same story in dollar terms. REN's best trained configuration is profitable in absolute terms but "
     "tracks well below both simple benchmarks throughout the test period; the untrained baseline spends much "
     "of the period in negative territory.")
chart("02_equity_curves.png", "Cumulative return, real held-out test period.")
end_section()

pagebreak()

start_section()
h1("Robustness Across Market Conditions")
body("Splitting the same test period by real market-volatility regime (calm / normal / turbulent, by SPY "
     "realized volatility) tests whether a model's performance is consistent or concentrated in one regime.")
chart("03_regime_breakdown.png",
      "Sharpe ratio by volatility regime. Note the untrained REN variant swings from the best performer "
      "(normal regime) to sharply negative (turbulent regime) &mdash; the opposite pattern from the trained "
      "variant, evidence this behavior is idiosyncratic to a specific run, not a structural property of the "
      "architecture.")
end_section()

start_section()
h1("Seed-to-Seed Reliability")
body("Every trained and untrained configuration tested (25 random seeds total) against the real held-out "
     "period. This is the single clearest picture of reliability: no seed, trained or untrained, reaches the "
     "buy-and-hold benchmark.")
chart("05_seed_distribution.png",
      "Held-out Sharpe ratio, every seed tested. The dashed line marks simple buy-and-hold's Sharpe (1.61) for reference.")
end_section()

pagebreak()

# =====================================================================
# SEISMOGRAPH FINDING
# =====================================================================
start_section()
h1("Case Study: The Crisis-Detection Claim")
body("The architecture's central unifying idea is that one number &mdash; the spectral radius of the model's "
     "own Jacobian &mdash; should rise toward a critical threshold as a real crisis approaches, giving early "
     "warning \"for free.\" This was tested directly against six real historical stress events, including the "
     "COVID-19 crash.")
chart("04_seismograph_covid.png",
      "The same diagnostic signal, real COVID-19 crash window (Feb&ndash;Mar 2020), before and after targeted "
      "training. Untrained: moves in the wrong direction. Trained toward the target: recovers the correct "
      "direction, out-of-sample.")
body("<b>The honest reading:</b> as originally specified &mdash; a signal that falls out of the equilibrium "
     "computation with no extra work &mdash; the claim did not hold on real data (p&nbsp;=&nbsp;6&times;10<super rise=3 size=7>-19</super> "
     "in the wrong direction). When the model was explicitly trained toward real forward volatility as a target, "
     "the correct direction reappeared on data the training never saw (p&nbsp;=&nbsp;0.0016). That is a real, "
     "useful, trainable feature &mdash; but a materially different and more modest claim than \"free\" crisis "
     "detection.")
end_section()

pagebreak()

# =====================================================================
# METHODOLOGY & INTEGRITY
# =====================================================================
h1("Methodology &amp; Data Integrity")
h2("What is real")
bullet("All market data: 16 liquid instruments (equities, sectors, rates, gold, oil, dollar, volatility), "
       "2015&ndash;2026, sourced live, not simulated.")
bullet("All latency, convergence, and accuracy numbers: measured on real hardware running the actual code, "
       "not restated design targets.")
bullet("All backtests: real chronological train/validation/test splits with no lookahead; hyperparameters "
       "selected on validation only, test set touched exactly once per configuration.")
bullet("Three real bugs were found and fixed during this work (a gradient-computation bug, a train/test "
       "leakage risk in an early hyperparameter sweep, and a data-labeling bug affecting the very last day of "
       "any test window) &mdash; each is disclosed with its measured before/after impact in the full technical report.")

h2("What is a disclosed modeling choice")
bullet("No public dataset of real trading-desk beliefs exists anywhere. The five \"agent types\" driving the "
       "model are explicit, documented functions of real price/volume data &mdash; not fabricated data "
       "presented as if it were observed.")
bullet("This implementation is a faithful, independently-engineered, smaller-scale version of the core "
       "mathematical ideas in the original specification (which describes a larger reference architecture, "
       "including a Sylvester-recurrence belief representation and a larger preconditioned solver not "
       "reproduced here). The underlying mechanisms tested &mdash; fixed-point equilibria, resolvent kernels, "
       "spectral-radius monitoring &mdash; are the same class of object; the specific engineering is not a "
       "byte-for-byte reproduction.")

hr()
body("<i>Full technical report (REPORT.md, ~30 pages across six rounds of testing and fixes), all source code, "
     "and every raw result file referenced above are available in full for technical due diligence.</i>",
     )

doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                          topMargin=0.85 * inch, bottomMargin=0.85 * inch,
                          leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                          title="RecWM / REN — Investor Report", author="Independent Implementation Project")
doc.build(story)
print("Saved:", OUT)
