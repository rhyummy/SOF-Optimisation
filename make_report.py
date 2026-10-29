import os, re
import numpy as np, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

P = r"C:\Users\Rhythm Garg\OneDrive\Desktop\SOC_Assessment10"
OUT = os.path.join(P, "outputs")
FIG = os.path.join(OUT, "figures")
pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", r"C:\Windows\Fonts\ariali.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["Normal"], fontName="Arial", fontSize=10, leading=14, spaceAfter=6)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8.5, leading=11)
CAP = ParagraphStyle("c", parent=BODY, fontName="Arial-Italic", fontSize=9, alignment=1, spaceBefore=2, spaceAfter=10)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Arial-Bold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1f3b63"))
H2 = ParagraphStyle("h2", parent=BODY, fontName="Arial-Bold", fontSize=11.5, leading=15, spaceBefore=6, spaceAfter=4)
TITLE = ParagraphStyle("t", parent=BODY, fontName="Arial-Bold", fontSize=18, leading=23, alignment=1, spaceAfter=8)
CENTER = ParagraphStyle("ce", parent=BODY, alignment=1)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=10.5, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Arial-Bold")

W = A4[0] - 4 * cm


def table(rows, widths=None, header=True, bold_rows=(), font=8.5):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) or i in bold_rows else CELL) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f1")))
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff4d6")))
    t.setStyle(TableStyle(st))
    return t


def fig(name, caption, width=W):
    from PIL import Image as PI
    path = os.path.join(FIG, name)
    w, h = PI.open(path).size
    return KeepTogether([Image(path, width=width, height=width * h / w), Paragraph(caption, CAP)])


f4 = lambda x: "-" if pd.isna(x) else f"{x:.4f}"
f2 = lambda x: "-" if pd.isna(x) else f"{x:.2f}"

# ---------------- readings ----------------
res = pd.read_csv(os.path.join(OUT, "results_table.csv"), index_col=0)
per = pd.read_csv(os.path.join(OUT, "per_trip_rmse.csv"))
pred = pd.read_csv(os.path.join(OUT, "predictions_test.csv"))
log = open(os.path.join(OUT, "run_log.txt"), encoding="utf-8", errors="replace").read().splitlines()
sysinfo = [l.split(":", 1) for l in log[1:8] if ":" in l]
trip_rows = [l.split() for l in log if re.match(r"^\s*\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+(train|test)\s*$", l)]
epochs = [re.findall(r"[\d.]+e?-?\d*", l) for l in log if l.startswith("Epoch")]
grab = lambda pat: next(re.search(pat, l).groups() for l in log if re.search(pat, l))
k_cc, cap = grab(r"gain k = ([\d.]+) %/Ah \(effective capacity ([\d.]+) Ah\)")
s1 = grab(r"stage 1.*train-CV RMSE ([\d.]+)%.*test RMSE ([\d.]+)%")
s2 = grab(r"stage 2.*train-CV RMSE ([\d.]+)%.*test RMSE ([\d.]+)%")
vbase = grab(r"physics estimate alone.*: ([\d.]+)%")[0]
nparam, rfield = grab(r"Trainable parameters: ([\d,]+) \| receptive field: (\d+)")
nwin = grab(r"fit: ([\d,]+)\s+val: ([\d,]+)\s+test: ([\d,]+)")
desc_start = next(i for i, l in enumerate(log) if l.strip().startswith("Voltage") and "Current" in l)
desc = [l.split() for l in log[desc_start + 1:desc_start + 9]]

tcn_name, phys_name = res.index[0], res.index[1]
m_tcn = res.loc[tcn_name]


def trip_stats(col_err):
    g = pred.groupby("trip")[col_err]
    return pd.DataFrame({"RMSE": g.apply(lambda e: np.sqrt(np.mean(e ** 2))), "MAE": g.apply(lambda e: e.abs().mean()),
                         "MAX": g.apply(lambda e: e.abs().max()), "Bias": g.mean()})


ts_tcn, ts_rf = trip_stats("TCN_error"), trip_stats("RF_error")
r2 = 1 - np.sum(pred.TCN_error ** 2) / np.sum((pred.actual_SOC - pred.actual_SOC.mean()) ** 2)
r2_rf = 1 - np.sum(pred.RF_error ** 2) / np.sum((pred.actual_SOC - pred.actual_SOC.mean()) ** 2)
within = {th: 100 * np.mean(pred.TCN_error.abs() <= th) for th in (1, 2, 5, 10)}
within_rf = {th: 100 * np.mean(pred.RF_error.abs() <= th) for th in (1, 2, 5, 10)}

# ---------------- document ----------------
story = []
story += [Spacer(1, 1.2 * cm),
          Paragraph("State of Charge Estimation for the BMW i3 using a Physics-Informed Temporal Convolutional Network", TITLE),
          Spacer(1, 4),
          Paragraph("<b>Name:</b> RHYTHM GARG &nbsp;&nbsp;|&nbsp;&nbsp; <b>Reg. No.:</b> 25BBS0153", CENTER),
          Paragraph("Assessment 10 – SOC estimation", CENTER), Spacer(1, 10)]
story.append(table([
    ["Base paper", "M. H. Sulaiman and Z. Mustaffa, “State of charge estimation for electric vehicles using random forest”, "
                   "<i>Green Energy and Intelligent Transportation</i>, 3 (2024) 100177."],
    ["Dataset", "“Battery and Heating Data in Real Driving Cycles” (BMW i3, 60 Ah pack), IEEE DataPort, doi:10.21227/6jr9-5235"],
    ["Proposed model", "Physics-informed TCN: coulomb counting + temperature-aware voltage model + causal fusion, "
                       "with a TCN residual correction"],
    ["Code", "github.com/rhyummy/SOF-Optimisation (soc_tcn.py, soc_tcn.ipynb)"],
], widths=[3.2 * cm, W - 3.2 * cm], header=False))
story.append(Spacer(1, 12))

story.append(Paragraph("Summary of results", H2))
story.append(table([
    ["Metric (test trips #61–#70)", "Proposed TCN", "RF 25 trees (re-impl.)", "RF 25 trees (paper)", "Improvement vs paper"],
    ["RMSE (%)", f4(m_tcn["RMSE (%)"]), f4(res.iloc[2]["RMSE (%)"]), f4(res.iloc[3]["RMSE (%)"]),
     f"{100 * (1 - m_tcn['RMSE (%)'] / res.iloc[3]['RMSE (%)']):.1f} % lower"],
    ["MAE (%)", f4(m_tcn["MAE (%)"]), f4(res.iloc[2]["MAE (%)"]), f4(res.iloc[3]["MAE (%)"]),
     f"{100 * (1 - m_tcn['MAE (%)'] / res.iloc[3]['MAE (%)']):.1f} % lower"],
    ["MAX (%)", f4(m_tcn["MAX (%)"]), f4(res.iloc[2]["MAX (%)"]), f4(res.iloc[3]["MAX (%)"]),
     f"{100 * (1 - m_tcn['MAX (%)'] / res.iloc[3]['MAX (%)']):.1f} % lower"],
    ["STD. DEV. (%)", f4(m_tcn["STD. DEV. (%)"]), f4(res.iloc[2]["STD. DEV. (%)"]), f4(res.iloc[3]["STD. DEV. (%)"]),
     f"{100 * (1 - m_tcn['STD. DEV. (%)'] / res.iloc[3]['STD. DEV. (%)']):.1f} % lower"],
    ["R²", f"{r2:.4f}", f"{r2_rf:.4f}", "-", ""],
], widths=[4.2 * cm, 2.9 * cm, 3.4 * cm, 3.2 * cm, W - 13.7 * cm]))
story.append(Spacer(1, 6))
story.append(Paragraph(
    f"The proposed model reaches an RMSE of <b>{m_tcn['RMSE (%)']:.4f} %</b> on the 118,974 test instances, below the "
    f"3.6 % target and lower than both the re-implemented Random Forest ({res.iloc[2]['RMSE (%)']:.4f} %) and the best "
    f"result reported in the base paper ({res.iloc[3]['RMSE (%)']:.4f} %).", BODY))
story.append(PageBreak())

# 1 objective
story.append(Paragraph("1. Objective", H1))
story.append(Paragraph(
    "Estimate the battery State of Charge (SOC) of a BMW i3 from the four signals used in the base paper — battery voltage, "
    "battery current, battery temperature and ambient temperature — using a Temporal Convolutional Network, and compare "
    "it with the paper's best Random Forest (25 trees). The target is the SOC estimated by the manufacturer "
    "(<i>SoC [%]</i> column). Following the paper, trips #1–#60 are used for training and trips #61–#70 for testing; "
    "metrics are RMSE, MAE, maximum absolute error, standard deviation of the error and computation time.", BODY))

# 2 dataset
story.append(Paragraph("2. Dataset and preprocessing", H1))
n_tr = sum(int(r[1]) for r in trip_rows if r[4] == "train")
n_te = sum(int(r[1]) for r in trip_rows if r[4] == "test")
story.append(Paragraph(
    "The data are supplied as two CSV files (training: trips 1–60, testing: trips 61–70), sampled at 10 Hz. Trip "
    "boundaries are recovered from resets of the time column. Preprocessing follows the base paper:", BODY))
for b in ["Inputs: voltage (V), current (A), battery temperature (°C), ambient temperature (°C); output: SOC (%).",
          "Rows with missing values or physically inconsistent readings (V ≤ 0, SOC outside 0–100 %, temperatures out of "
          "range) are removed.",
          "Noisy records: rows where SOC jumps by more than 5 % between consecutive samples are removed (one logging "
          "glitch at the end of trip 46, where SOC drops from 35 % to 0 %).",
          "Min–max normalisation is fitted on the training trips only."]:
    story.append(Paragraph("• " + b, BODY))
story.append(table([
    ["Split", "Trips", "Instances (after cleaning)"],
    ["Training", "#1 – #60", f"{n_tr:,}"],
    ["  of which: fitting / early-stopping validation", "#1 – #54 / #55 – #60", f"{nwin[0]} / {nwin[1]}"],
    ["Testing", "#61 – #70", f"{n_te:,}"],
    ["Total", "70", f"{n_tr + n_te:,}"],
], widths=[7 * cm, 4 * cm, W - 11 * cm]))
story.append(Spacer(1, 8))
story.append(Paragraph("Descriptive statistics of all cleaned data:", BODY))
story.append(table([["Statistic", "Voltage (V)", "Current (A)", "Battery temp. (°C)", "Ambient temp. (°C)", "SOC (%)"]]
                   + [r for r in desc], widths=[2.6 * cm] + [(W - 2.6 * cm) / 5] * 5))
story.append(Spacer(1, 6))
story.append(fig("01_training_trip_signals.png", "Figure 1. Preprocessed input signals and SOC of training trip #1 (x-axis: sample index at 10 Hz).",
                 width=W * 0.6))

# 3 method
story.append(PageBreak())
story.append(Paragraph("3. Proposed method: physics-informed TCN", H1))
story.append(Paragraph(
    "A first analysis showed that SOC changes within a trip are almost perfectly explained by the integrated current: "
    "if the true SOC at the start of each test trip is known, coulomb counting alone tracks the test trips with an RMSE of "
    "about 0.05 %. The error of data-driven models therefore comes mostly from not knowing each trip's starting SOC. "
    "The proposed model combines physics-based features with a TCN. Every quantity is computed causally "
    "(from past and present samples only) from the four measured signals.", BODY))
steps = [
    ("Coulomb counting", f"ΔSOC(t) = k · ∫ I dt since the start of the trip. The gain k is fitted on the training trips: "
                         f"k = {k_cc} %/Ah, i.e. an effective capacity of {cap} Ah."),
    ("Stage 1 – voltage model", "Polynomial ridge regression of SOC on voltage (up to cubic), battery temperature and "
                                "current history (instantaneous, 10 s, 60 s and 300 s means of I and |I|), with "
                                "temperature and voltage interaction terms. It models the open-circuit-voltage curve, its "
                                "temperature shift and the IR / polarisation drop."),
    ("Stage 2 – fusion", "Within each trip, the offset (stage-1 estimate − ΔSOC) changes slowly. It is averaged with causal "
                         "exponential averages (half-lives 30 s, 2 min, 10 min, 50 min; plain and weighted towards "
                         "low-current samples) and added back to ΔSOC. A second ridge model blends these estimates."),
    ("Out-of-fold features", "On the training trips, stages 1 and 2 are computed with 6-fold trip-wise cross-validation, "
                             "so the TCN is trained on estimates whose errors are as large as on unseen trips."),
    ("TCN residual correction", "A causal TCN reads the last 120 samples of seven channels (V, I, T<sub>batt</sub>, "
                                "T<sub>amb</sub>, fused estimate, stage-1 estimate, ΔSOC) and outputs "
                                "SOC = fused estimate + 0.03 · tanh(TCN output), i.e. a correction of at most ±3 % SOC. "
                                "A 60-sample causal moving average is applied to the output."),
]
story.append(table([["Step", "Description"]] + [[a, b] for a, b in steps], widths=[3.8 * cm, W - 3.8 * cm]))
story.append(Spacer(1, 8))
story.append(Paragraph("TCN configuration and training settings", H2))
story.append(table([
    ["Parameter", "Value", "Parameter", "Value"],
    ["Residual blocks", "5", "Optimiser", "AdamW, lr 1e-3, weight decay 0.01"],
    ["Channels / kernel size", "32 / 3", "Loss", "Huber (δ = 0.02)"],
    ["Dilations", "1, 2, 4, 8, 16", "LR schedule", "ReduceLROnPlateau (factor 0.5, patience 2)"],
    ["Receptive field", f"{rfield} samples", "Early stopping", "patience 3, max 8 epochs, on trips #55–#60"],
    ["Input window", "120 samples (12 s)", "Batch size", "512"],
    ["Dropout", "0.2 + BatchNorm", "Windows per epoch", "every 40th training window"],
    ["Trainable parameters", nparam, "Random seed", "42"],
], widths=[3.6 * cm, 3.4 * cm, 3.2 * cm, W - 10.2 * cm]))
story.append(Spacer(1, 8))
story.append(Paragraph("Reference model", H2))
story.append(Paragraph(
    "Random Forest with 25 trees (the paper's best configuration), min_samples_leaf = 5, trained on the four normalised "
    "inputs of all training instances (scikit-learn).", BODY))
story.append(Paragraph("Experimental setup", H2))
story.append(table([["Item", "Value"]] + [[a.strip(), b.strip()] for a, b in sysinfo] +
                   [["Libraries", "PyTorch, scikit-learn, pandas, NumPy, Matplotlib"]],
                   widths=[4 * cm, W - 4 * cm]))

# 4 results
story.append(PageBreak())
story.append(Paragraph("4. Results", H1))
story.append(Paragraph("4.1 Physics-feature stages", H2))
story.append(table([
    ["Stage", "Training trips, cross-validated RMSE (%)", "Test RMSE (%)"],
    ["Stage 1 – voltage model", s1[0], s1[1]],
    ["Stage 2 – fusion with coulomb counting", s2[0], s2[1]],
], widths=[6.5 * cm, 6 * cm, W - 12.5 * cm]))
story.append(Spacer(1, 8))
story.append(Paragraph("4.2 TCN training log", H2))
story.append(Paragraph(f"Validation RMSE of the fused estimate before any TCN correction: <b>{vbase} %</b>. "
                       f"The model from the epoch with the lowest validation RMSE is kept.", BODY))
best_ep = int(np.argmin([float(e[2]) for e in epochs]))
story.append(table([["Epoch", "Training RMSE (%)", "Validation RMSE (%)", "Learning rate"]] +
                   [[e[0], e[1], e[2], e[3]] for e in epochs], widths=[2.5 * cm, 4.2 * cm, 4.2 * cm, W - 10.9 * cm],
                   bold_rows=(best_ep + 1,)))
story.append(Paragraph("Early stopping after epoch %s; best epoch highlighted." % epochs[-1][0], SMALL))
story.append(fig("02_learning_curve.png", "Figure 2. TCN learning curve (training and validation RMSE per epoch).",
                 width=W * 0.7))

story.append(Paragraph("4.3 Performance on the test data (118,974 instances)", H2))
story.append(table([["Model", "RMSE (%)", "MAE (%)", "MAX (%)", "STD. DEV. (%)", "Train (s)", "Test (s)", "Total (s)"]] +
                   [[i, f4(r["RMSE (%)"]), f4(r["MAE (%)"]), f4(r["MAX (%)"]), f4(r["STD. DEV. (%)"]),
                     f2(r["Training time (s)"]), f2(r["Testing time (s)"]), f2(r["Total computation time (s)"])]
                    for i, r in res.iterrows()],
                   widths=[4.6 * cm] + [(W - 4.6 * cm) / 7] * 7, bold_rows=(1,)))
story.append(Paragraph("Times measured on a laptop CPU (no GPU); the paper's time is taken from the paper. "
                       "The proposed model's training time includes building the physics features.", SMALL))
story.append(Spacer(1, 6))
story.append(table([["Share of test instances with |error| ≤", "1 %", "2 %", "5 %", "10 %"],
                    ["Proposed TCN"] + [f"{within[t]:.1f} %" for t in (1, 2, 5, 10)],
                    ["RF 25 trees (re-impl.)"] + [f"{within_rf[t]:.1f} %" for t in (1, 2, 5, 10)]],
                   widths=[6 * cm] + [(W - 6 * cm) / 4] * 4))
story.append(Spacer(1, 6))
story.append(fig("07_metrics_bar.png", "Figure 3. Error metrics on the test data for all models.", width=W * 0.8))

story.append(Paragraph("4.4 Per-trip results", H2))
rows = [["Trip", "Instances", "SOC start → end (%)", "TCN RMSE", "TCN MAE", "TCN MAX", "TCN bias", "RF RMSE"]]
tinfo = {int(r[0]): r for r in trip_rows}
for t in ts_tcn.index:
    a = ts_tcn.loc[t]
    rows.append([t, f"{int(tinfo[t][1]):,}", f"{tinfo[t][2]} → {tinfo[t][3]}", f"{a.RMSE:.3f}", f"{a.MAE:.3f}",
                 f"{a.MAX:.3f}", f"{a.Bias:+.3f}", f"{ts_rf.loc[t].RMSE:.3f}"])
story.append(table(rows, widths=[1.3 * cm, 2 * cm, 3.2 * cm] + [(W - 6.5 * cm) / 5] * 5))
story.append(Paragraph("All errors in % SOC. Bias = mean (estimated − actual).", SMALL))

story.append(PageBreak())
story.append(Paragraph("4.5 Simulation graphs", H2))
story.append(fig("03_soc_tracking_and_error_TCN.png",
                 "Figure 4. (a) Actual and estimated SOC by the proposed TCN for all test instances; "
                 "(b) estimation error with the maximum absolute error marked."))
story.append(fig("04_soc_tracking_TCN_vs_RF.png", "Figure 5. SOC tracking comparison – proposed TCN vs Random Forest."))
story.append(fig("05_soc_error_TCN_vs_RF.png", "Figure 6. SOC estimation error comparison – proposed TCN vs Random Forest."))
story.append(fig("06_error_hist_scatter_pertrip.png",
                 "Figure 7. Error distribution, actual vs estimated SOC (TCN) and RMSE per test trip."))

# 5 optimisation history
story.append(Paragraph("5. Optimisation history", H1))
story.append(Paragraph(
    "The table lists the configurations tried during optimisation. Settings were selected on the validation trips "
    "(#55–#60) or by cross-validation on the training trips, not on the test set.", BODY))
story.append(table([
    ["Configuration", "Validation RMSE (%)", "Test RMSE (%)"],
    ["TCN on the 4 raw signals only (96 channels, 180-sample window)", "-", "5.3655"],
    ["Physics stage 1 only (voltage model)", "-", s1[1]],
    ["Physics stage 2 only (fusion with coulomb counting, no TCN)", vbase, s2[1]],
    ["Physics + TCN correction up to ±10 %, Adam", "3.152", "3.988"],
    ["Physics + TCN correction up to ±3 %, AdamW 0.01 (selected)", "2.630", f"{m_tcn['RMSE (%)']:.4f}"],
    ["Physics + TCN ±3 %, current channel only", "3.634", "3.528"],
    ["Physics + TCN ±5 %, voltage and current channels, AdamW 0.05", "2.978", "3.294"],
], widths=[10 * cm, 3.4 * cm, W - 13.4 * cm], bold_rows=(5,)))
story.append(Paragraph("The selected configuration has the lowest validation RMSE. A variant with a lower test RMSE "
                       "(3.294 %) was not selected, because choosing it would mean tuning on the test set.", SMALL))

# 6 discussion
story.append(Paragraph("6. Discussion", H1))
for p in [
    "Adding coulomb counting and a physics-based voltage model is the main source of improvement: the RMSE drops from "
    "5.37 % (TCN on raw signals) to 3.49 % with the fused physics estimate. A plain TCN looking at a short window of "
    "voltage, current and temperature cannot know how much charge has been used since the trip started, and it "
    "over-fits to the 60 training trips.",
    "The TCN correction improves the validation trips (3.16 % → 2.63 %) but, on the test trips, it is slightly worse "
    f"than the fused estimate alone ({s2[1]} % vs {m_tcn['RMSE (%)']:.2f} %). Both results are below the 3.6 % target and "
    "both are reported.",
    "The largest errors occur on trip #61 (SOC between 31 % and 15 %, at the edge of the training data), trip #65 (the "
    "voltage corresponds to a higher SOC than the manufacturer label) and trip #70 (battery at about 2 °C, colder than "
    "almost all training trips). These cases show a roughly constant per-trip offset, which the voltage-based starting-SOC "
    "estimate cannot fully remove.",
    "Compared with the Random Forest, the proposed model has a much smaller maximum error (about 12 % vs 21 %) and "
    "tracks SOC smoothly, while the RF estimate is noisy because it treats every sample independently. Its main cost is "
    "computation time (minutes instead of seconds on a CPU).",
]:
    story.append(Paragraph(p, BODY))
story.append(Paragraph("7. Conclusion", H1))
story.append(Paragraph(
    f"A physics-informed TCN for SOC estimation of the BMW i3 was developed using the base paper's preprocessing and "
    f"inputs. On test trips #61–#70 it achieves RMSE {m_tcn['RMSE (%)']:.4f} %, MAE {m_tcn['MAE (%)']:.4f} %, MAX "
    f"{m_tcn['MAX (%)']:.4f} % and STD. DEV. {m_tcn['STD. DEV. (%)']:.4f} %, improving on the paper's best Random Forest "
    f"(RMSE 5.9028 %) by {100 * (1 - m_tcn['RMSE (%)'] / 5.9028):.1f} %. Combining coulomb counting with a learned "
    f"voltage model is the most effective step; better handling of very cold and very low-SOC trips is the main "
    f"remaining improvement.", BODY))

# appendix
story.append(PageBreak())
story.append(Paragraph("Appendix A. Trip summary (after cleaning)", H1))
rows = [["Trip", "Set", "Instances", "SOC start (%)", "SOC end (%)"]] + \
       [[r[0], r[4], f"{int(r[1]):,}", r[2], r[3]] for r in trip_rows]
half = (len(rows) + 1) // 2
left, right = rows[:half], [rows[0]] + rows[half:]
right += [[""] * 5] * (len(left) - len(right))
cw = [1.1 * cm, 1.2 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm]
story.append(Table([[table(left, widths=cw), table(right, widths=cw)]], colWidths=[W / 2, W / 2]))


def footer(c, d):
    c.saveState()
    c.setFont("Arial", 8)
    c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.2 * cm, "RHYTHM GARG (25BBS0153) – Physics-informed TCN for SOC estimation")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {d.page}")
    c.restoreState()


out = os.path.join(P, "SOC_TCN_Assessment10.pdf")
doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=2 * cm, title="SOC Estimation – Physics-Informed TCN", author="RHYTHM GARG")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("written", out)
