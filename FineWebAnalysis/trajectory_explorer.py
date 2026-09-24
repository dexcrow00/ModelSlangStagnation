#!/usr/bin/env python3
"""
trajectory_explorer.py — Build a self-contained interactive page for inspecting
each word's percent-of-peak trajectory one at a time.

Companion to ``plot_min_pct.py``: that script collapses each word to a single bar
(its lowest year as a percentage of its peak), which is good for ranking but hides
the *shape* that produced the number. This writes one small-multiple per word plus
a detail view -- year on x, percent of peak on y -- so trajectories with the same
minimum but different shapes (a clean rise-and-fall vs. a noisy flat line) can be
told apart by eye.

The page carries the roughness statistics alongside each curve, so a visual
impression can be checked against a number:

  minSE%   median per-year sampling SE, 1/sqrt(hits) -- the counting-noise floor
  TV       total-variation ratio, sum|dx| / (max-min); 1.0 = monotone
  2ndD     RMS second difference / SD; trend-immune roughness
  turns    direction changes in the series, with a z-score against the IID
           expectation 2(n-2)/3 +- sqrt((16n-29)/90) -- z near 0 means the curve
           wanders as much as a coin-flip series would
  R2       fit of a quadratic in year
  noise    residual scatter as a multiple of the Poisson floor; ~1 means the whole
           trajectory is consistent with a smooth curve plus counting noise

Defaults reproduce the word set in ``figures/min_pct_of_peak.png``.

Usage (run from FineWebAnalysis/):
    python trajectory_explorer.py
    python trajectory_explorer.py --min-hits 100 --max-min-pct 40
    python trajectory_explorer.py -o figures/explorer.html
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "peak_years_pct.json"
DEFAULT_SIZES = HERE / "fineweb_10BT_dump_sizes.json"
DEFAULT_OUTPUT = HERE / "figures" / "trajectory_explorer.html"
CRAWL_ID_RE = re.compile(r"CC-MAIN-(\d{4})-\d{2}")


def year_tokens(cache_path: Path) -> dict[int, int]:
    """{year: FineWeb sample token total}, summed from the per-dump size cache."""
    if not cache_path.is_file():
        sys.exit(f"Token-size cache not found: {cache_path}")
    tokens: dict[int, int] = {}
    for dump_id, tok in json.loads(cache_path.read_text(encoding="utf-8")).items():
        m = CRAWL_ID_RE.search(dump_id)
        if m:
            tokens[int(m.group(1))] = tokens.get(int(m.group(1)), 0) + int(tok)
    return tokens


def _polyfit2(x: list[float], y: list[float]) -> list[float]:
    """Least-squares quadratic, solved with normal equations (stdlib only)."""
    n = len(x)
    # Gram matrix for [1, x, x^2]
    A = [[sum(xi ** (i + j) for xi in x) for j in range(3)] for i in range(3)]
    b = [sum(y[k] * x[k] ** i for k in range(n)) for i in range(3)]
    # Gaussian elimination with partial pivoting
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            return [0.0, 0.0, sum(y) / n]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, 4):
                    M[r][k] -= f * M[c][k]
    return [M[i][3] / M[i][i] for i in range(3)]


def metrics(rate: list[float], tok_m: list[float]) -> dict:
    """Roughness and noise-floor statistics for one word's yearly rate series."""
    n = len(rate)
    peak = max(rate)
    pct = [100.0 * r / peak for r in rate]
    hits = [r * t for r, t in zip(rate, tok_m)]          # reconstructed counts
    mean = sum(pct) / n
    ss = sum((p - mean) ** 2 for p in pct)
    sd = math.sqrt(ss / (n - 1)) if ss > 0 else 0.0

    d1 = [pct[i + 1] - pct[i] for i in range(n - 1)]
    d2 = [pct[i + 2] - 2 * pct[i + 1] + pct[i] for i in range(n - 2)]
    tv = sum(abs(d) for d in d1) / (max(pct) - min(pct)) if max(pct) > min(pct) else 0.0
    rough = (math.sqrt(sum(d * d for d in d2) / len(d2)) / sd) if sd > 0 else 0.0
    turns = sum(1 for i in range(len(d1) - 1) if d1[i] * d1[i + 1] < 0)
    z_turns = (turns - 2 * (n - 2) / 3) / math.sqrt((16 * n - 29) / 90)

    # quadratic fit, on pct for R^2 and on rate for the Poisson comparison
    x = [float(i) for i in range(n)]
    c = _polyfit2(x, pct)
    resid = [pct[i] - (c[0] + c[1] * x[i] + c[2] * x[i] ** 2) for i in range(n)]
    r2 = 1 - sum(r * r for r in resid) / ss if ss > 0 else 0.0

    cr = _polyfit2(x, rate)
    fit = [cr[0] + cr[1] * x[i] + cr[2] * x[i] ** 2 for i in range(n)]
    # Poisson variance of a rate estimate comes from the FITTED mean, not the
    # observed count: an observed zero has observed variance zero and would
    # divide the ratio to infinity.
    chi2 = 0.0
    for i in range(n):
        lam = max(fit[i] * tok_m[i], 0.5)              # fitted counts, floored
        var = lam / (tok_m[i] ** 2)                    # Var of the rate
        chi2 += (rate[i] - fit[i]) ** 2 / var
    noise = math.sqrt(chi2 / (n - 3))

    nonzero = [h for h in hits if h > 0]
    nonzero.sort()
    med_hits = nonzero[len(nonzero) // 2] if nonzero else 0.0
    se = 100 / math.sqrt(med_hits) if med_hits > 0 else float("nan")

    return dict(tv=tv, rough=rough, turns=turns, z=z_turns, r2=r2, noise=noise, se=se)


def build(input_path: Path, sizes_path: Path, min_hits: int,
          max_min_pct: float | None, top: int | None) -> tuple[list[dict], dict]:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    tokens = year_tokens(sizes_path)

    words = []
    for r in records:
        pct_map = r.get("pct_of_peak_by_year")
        if not pct_map:
            sys.exit(f"{input_path} has no 'pct_of_peak_by_year' — regenerate with peak_year.py.")
        if r["total_hits"] < min_hits:
            continue
        years = sorted(int(y) for y in pct_map)
        pct = [pct_map[str(y)] for y in years]
        min_pct = min(pct)
        if max_min_pct is not None and min_pct > max_min_pct:
            continue
        rate = [r["rate_by_year"][str(y)] for y in years]
        tok_m = [tokens[y] / 1e6 for y in years]
        m = metrics(rate, tok_m)
        words.append({
            "w": r["word"],
            "peak": r["peak_year"],
            "hits": r["total_hits"],
            "pct": [round(p, 1) for p in pct],
            "rate": [round(v, 3) for v in rate],
            "k": [int(round(v * t)) for v, t in zip(rate, tok_m)],
            "minY": years[pct.index(min_pct)],
            "minP": round(min_pct, 1),
            "tv": round(m["tv"], 2),
            "rgh": round(m["rough"], 2),
            "trn": m["turns"],
            "z": round(m["z"], 1),
            "r2": round(m["r2"], 2),
            "nse": round(m["noise"], 1),
            "se": (round(m["se"], 1) if m["se"] == m["se"] else None),
        })
    if not words:
        sys.exit("No words passed the filters.")
    words.sort(key=lambda d: d["minP"])
    if top:
        words = words[:top]

    years = sorted(int(y) for y in records[0]["pct_of_peak_by_year"])
    meta = {
        "years": years,
        "tokens": {str(y): round(tokens[y] / 1e6, 1) for y in years},
        "minHits": min_hits,
        "maxMinPct": max_min_pct,
        "n": len(words),
    }
    return words, meta


TEMPLATE = r'''<title>Slang Lifecycles in FineWeb</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap">
<style>
:root{
  --ground:#eceef2; --surface:#f8f9fb; --raised:#ffffff;
  --ink:#141a22; --ink2:#4d5763; --ink3:#7b8695;
  --line:#d8dce3; --grid:#e3e7ec;
  --trace:#1668c4; --peak:#b0741c; --min:#9c3f66;
  --ghost:rgba(20,26,34,.10); --trace-soft:rgba(22,104,196,.10);
  --focus:#1668c4;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0f1319; --surface:#171d25; --raised:#1d242e;
    --ink:#e6eaf0; --ink2:#a2adbb; --ink3:#737e8c;
    --line:#262e39; --grid:#1e252e;
    --trace:#4f93de; --peak:#b8863a; --min:#c05f7e;
    --ghost:rgba(230,234,240,.11); --trace-soft:rgba(79,147,222,.13);
    --focus:#7fb2ea;
  }
}
:root[data-theme="dark"]{
  --ground:#0f1319; --surface:#171d25; --raised:#1d242e;
  --ink:#e6eaf0; --ink2:#a2adbb; --ink3:#737e8c;
  --line:#262e39; --grid:#1e252e;
  --trace:#4f93de; --peak:#b8863a; --min:#c05f7e;
  --ghost:rgba(230,234,240,.11); --trace-soft:rgba(79,147,222,.13);
  --focus:#7fb2ea;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font:400 14px/1.5 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;
}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
h1{
  font:600 19px/1.2 "IBM Plex Serif",Georgia,serif; margin:0; letter-spacing:-.01em;
}
.bar{
  position:sticky; top:env(safe-area-inset-top,0px); z-index:20;
  background:var(--surface); border-bottom:1px solid var(--line);
  padding:12px 20px; display:flex; flex-wrap:wrap; gap:12px 18px; align-items:baseline;
}
.bar .sub{color:var(--ink2); font-size:12.5px}
.controls{display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-left:auto}
label.ctl{display:flex; align-items:center; gap:6px; font-size:12px; color:var(--ink2)}
select,input[type=search]{
  font:400 12.5px/1 "IBM Plex Sans",sans-serif; color:var(--ink);
  background:var(--raised); border:1px solid var(--line); border-radius:5px;
  padding:6px 8px; min-width:0;
}
input[type=search]{width:140px}
select:focus-visible,input:focus-visible,button:focus-visible{outline:2px solid var(--focus); outline-offset:1px}
.toggle{display:flex; align-items:center; gap:6px; font-size:12px; color:var(--ink2); cursor:pointer}
main{display:grid; grid-template-columns:minmax(0,1fr) 404px; gap:22px; padding:20px; align-items:start}
.gridwrap{min-width:0}
.legend{display:flex; flex-wrap:wrap; gap:14px; font-size:11.5px; color:var(--ink2); margin:0 0 12px}
.legend span{display:flex; align-items:center; gap:5px}
.swatch{width:14px;height:0;border-top:2px solid var(--trace)}
.swatch.ghost{border-top-color:var(--ghost)}
.dot{width:8px;height:8px;border-radius:50%}
.dot.peak{background:var(--peak)}
.dot.min{background:var(--surface);border:2px solid var(--min);width:9px;height:9px}
.cards{display:grid; grid-template-columns:repeat(auto-fill,minmax(118px,1fr)); gap:8px}
.card{
  appearance:none; text-align:left; cursor:pointer; padding:7px 8px 5px;
  background:var(--surface); border:1px solid var(--line); border-radius:6px;
  color:inherit; font:inherit; display:block; min-width:0;
}
.card:hover{border-color:var(--ink3)}
.card[aria-current="true"]{border-color:var(--trace); box-shadow:inset 0 0 0 1px var(--trace); background:var(--trace-soft)}
.card .w{font-size:11.5px; font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.card .v{font-size:10px; color:var(--ink3); margin-top:1px}
.card svg{display:block; width:100%; height:auto; margin-top:3px}
aside{
  position:sticky; top:calc(env(safe-area-inset-top,0px) + 62px);
  background:var(--surface); border:1px solid var(--line); border-radius:8px;
  padding:16px; min-width:0;
}
.word{font:600 26px/1.1 "IBM Plex Serif",Georgia,serif; letter-spacing:-.015em; word-break:break-word}
.sub2{color:var(--ink2); font-size:12.5px; margin-top:3px}
.stats{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; margin:14px 0 4px;
  background:var(--line); border:1px solid var(--line); border-radius:6px; overflow:hidden}
.stat{background:var(--surface); padding:7px 8px}
.stat dt{font-size:9.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink3); margin:0}
.stat dd{margin:2px 0 0; font-size:14px; font-weight:500}
.stat dd small{font-size:10px; color:var(--ink3); font-weight:400}
.chart{width:100%; height:auto; display:block; margin-top:10px; touch-action:none}
.readout{font-size:11.5px; color:var(--ink2); min-height:1.5em; margin-top:2px}
.caption{font-size:11px; color:var(--ink3); margin-top:8px; line-height:1.45}
details{margin-top:12px; border-top:1px solid var(--line); padding-top:10px}
summary{font-size:12px; color:var(--ink2); cursor:pointer}
summary:focus-visible{outline:2px solid var(--focus)}
table{border-collapse:collapse; width:100%; margin-top:8px; font-size:11.5px}
th,td{text-align:right; padding:3px 4px; border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink3); font-weight:500; font-size:10px; letter-spacing:.04em; text-transform:uppercase}
.nav{display:flex; gap:8px; margin-top:12px}
.nav button{
  flex:1; font:500 12px/1 "IBM Plex Sans",sans-serif; color:var(--ink);
  background:var(--raised); border:1px solid var(--line); border-radius:5px;
  padding:7px 8px; cursor:pointer;
}
.nav button:hover{border-color:var(--ink3)}
.kbd{font-family:"IBM Plex Mono",monospace; font-size:10.5px; color:var(--ink3)}
.foot{padding:0 20px 28px; color:var(--ink3); font-size:11.5px; max-width:70ch}
@media (max-width:960px){
  main{grid-template-columns:minmax(0,1fr)}
  aside{position:static; order:-1}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<header class="bar">
  <div>
    <h1>Slang Lifecycles in FineWeb</h1>
    <div class="sub">__N__ words &middot; each year as a percentage of that word&rsquo;s own peak year &middot; FineWeb 10BT, RoBERTa slang sense &ge; 0.99</div>
  </div>
  <div class="controls">
    <label class="ctl">Sort
      <select id="sort">
        <option value="minP">lowest year (% of peak)</option>
        <option value="peak">peak year</option>
        <option value="hits">total hits</option>
        <option value="rgh">roughness (2nd diff)</option>
        <option value="tv">total variation</option>
        <option value="z">turning-point z</option>
        <option value="nse">noise multiple</option>
        <option value="w">alphabetical</option>
      </select>
    </label>
    <label class="ctl">Align
      <select id="align">
        <option value="cal">calendar year</option>
        <option value="peak">years from peak</option>
      </select>
    </label>
    <label class="ctl"><span class="sr">Find</span>
      <input type="search" id="q" placeholder="filter words" autocomplete="off">
    </label>
  </div>
</header>

<main>
  <div class="gridwrap">
    <p class="legend">
      <span><i class="swatch"></i>selected word</span>
      <span><i class="swatch ghost"></i>the other words</span>
      <span><i class="dot peak"></i>peak year (100%)</span>
      <span><i class="dot min"></i>lowest year</span>
      <span class="kbd">&larr; &rarr; to step through</span>
    </p>
    <div class="cards" id="cards"></div>
  </div>

  <aside>
    <div class="word" id="dWord">&nbsp;</div>
    <div class="sub2" id="dSub">&nbsp;</div>
    <dl class="stats" id="dStats"></dl>
    <svg class="chart" id="dChart" viewBox="0 0 600 320" role="img" aria-labelledby="dChartTitle"><title id="dChartTitle">Percent of peak by year</title></svg>
    <p class="readout mono" id="dRead">&nbsp;</p>
    <svg class="chart" id="dHits" viewBox="0 0 600 78" role="img" aria-labelledby="dHitsTitle"><title id="dHitsTitle">Slang-sense hits per year</title></svg>
    <p class="caption" id="dCap"></p>
    <div class="nav">
      <button type="button" id="prev">&larr; Previous</button>
      <button type="button" id="next">Next &rarr;</button>
    </div>
    <details>
      <summary>Table view</summary>
      <table id="dTable"><thead><tr><th>Year</th><th>% of peak</th><th>per M</th><th>Hits</th><th>&plusmn;SE</th></tr></thead><tbody></tbody></table>
    </details>
  </aside>
</main>

<p class="foot">Percent of peak divides each year&rsquo;s slang-sense rate (per million sample tokens) by the word&rsquo;s own best year, so the peak always reads 100%. Sampling noise scales as 1/&radic;hits, and the per-year token budget is far from flat &mdash; 2024 covers only dump <span class="mono">2024-10</span> (103.9M tokens against 1,429.7M in 2017), so its point is the least certain on every curve. A word whose <b>noise</b> multiple sits near 1 has a trajectory fully consistent with a smooth curve plus counting noise; nothing about its shape is evidence of a lifecycle.</p>

<script>
const YEARS = __YEARS__;
const TOKENS = __TOKENS__;
const W = __DATA__;
const N = YEARS.length;

const $ = (id) => document.getElementById(id);
const state = {sort:"minP", align:"cal", q:"", sel:0};
let view = W.slice();

const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (n) => n.toLocaleString("en-US");

/* ---- x mapping: calendar year, or index relative to the word's peak ---- */
function relIndex(d, i){ return i - YEARS.indexOf(d.peak); }
function xFrac(d, i){
  return state.align === "cal" ? i / (N - 1) : (relIndex(d, i) + (N - 1)) / (2 * (N - 1));
}

function sortView(){
  const q = state.q.trim().toLowerCase();
  view = W.filter((d) => !q || d.w.toLowerCase().includes(q));
  const k = state.sort;
  view.sort((a, b) => {
    if (k === "w") return a.w.localeCompare(b.w);
    if (k === "hits" || k === "peak") return b[k] - a[k] || a.w.localeCompare(b.w);
    if (k === "rgh" || k === "tv" || k === "z" || k === "nse") return b[k] - a[k] || a.w.localeCompare(b.w);
    return a.minP - b.minP || a.w.localeCompare(b.w);
  });
}

/* ---- small multiples ---- */
function spark(d){
  const w = 120, h = 40, pad = 4;
  const X = (i) => pad + xFrac(d, i) * (w - 2 * pad);
  const Y = (v) => h - pad - (v / 100) * (h - 2 * pad);
  const pts = d.pct.map((v, i) => X(i).toFixed(1) + "," + Y(v).toFixed(1)).join(" ");
  const pi = YEARS.indexOf(d.peak), mi = YEARS.indexOf(d.minY);
  return '<svg viewBox="0 0 ' + w + ' ' + h + '" aria-hidden="true">' +
    '<line x1="0" y1="' + (h - pad) + '" x2="' + w + '" y2="' + (h - pad) + '" stroke="var(--grid)" stroke-width="1"/>' +
    '<polyline points="' + pts + '" fill="none" stroke="var(--trace)" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>' +
    '<circle cx="' + X(pi).toFixed(1) + '" cy="' + Y(100).toFixed(1) + '" r="2.6" fill="var(--peak)"/>' +
    '<circle cx="' + X(mi).toFixed(1) + '" cy="' + Y(d.pct[mi]).toFixed(1) + '" r="2.4" fill="var(--surface)" stroke="var(--min)" stroke-width="1.4"/>' +
    '</svg>';
}

function renderCards(){
  $("cards").innerHTML = view.map((d, i) =>
    '<button class="card" type="button" data-i="' + i + '" aria-current="' + (i === state.sel) + '" ' +
    'title="' + esc(d.w) + ' — peak ' + d.peak + ', low ' + d.minP + '% in ' + d.minY + ', ' + fmt(d.hits) + ' hits">' +
    '<span class="w">' + esc(d.w) + '</span>' +
    '<span class="v mono">' + d.minP.toFixed(0) + '% &middot; ' + d.peak + '</span>' +
    spark(d) + '</button>').join("");
}

/* ---- detail chart ---- */
const M = {l:44, r:18, t:16, b:34}, CW = 600, CH = 320;
const PX = (d, i) => M.l + xFrac(d, i) * (CW - M.l - M.r);
const PY = (v) => CH - M.b - (v / 100) * (CH - M.t - M.b);

function renderDetail(){
  const d = view[state.sel];
  if (!d) { $("dWord").textContent = "No match"; $("dSub").textContent = ""; $("dChart").innerHTML = ""; $("dHits").innerHTML = ""; $("dStats").innerHTML = ""; $("dTable").querySelector("tbody").innerHTML = ""; $("dCap").textContent = ""; return; }

  $("dWord").textContent = d.w;
  $("dSub").innerHTML = 'peak <b class="mono">' + d.peak + '</b> &middot; low <b class="mono">' + d.minP + '%</b> in <b class="mono">' + d.minY +
    '</b> &middot; <span class="mono">' + fmt(d.hits) + '</span> slang-sense hits';

  const stats = [
    ["low % of peak", d.minP.toFixed(1), ""],
    ["median SE", d.se == null ? "&mdash;" : "&plusmn;" + d.se.toFixed(1) + "%", ""],
    ["total variation", d.tv.toFixed(2), "1.0 = monotone"],
    ["2nd-diff", d.rgh.toFixed(2), "roughness"],
    ["turns", String(d.trn), "z " + (d.z > 0 ? "+" : "") + d.z.toFixed(1)],
    ["R&sup2; quad", d.r2.toFixed(2), ""],
    ["noise", "&times;" + d.nse.toFixed(1), "vs Poisson"],
    ["peak rate", d.rate[YEARS.indexOf(d.peak)].toFixed(2), "per M"],
  ];
  $("dStats").innerHTML = stats.map(([t, v, s]) =>
    '<div class="stat"><dt>' + t + '</dt><dd class="mono">' + v + (s ? ' <small>' + s + '</small>' : '') + '</dd></div>').join("");

  /* --- main chart --- */
  let s = "";
  for (const g of [0, 25, 50, 75, 100]) {
    s += '<line x1="' + M.l + '" y1="' + PY(g) + '" x2="' + (CW - M.r) + '" y2="' + PY(g) + '" stroke="var(--grid)" stroke-width="1"/>' +
      '<text x="' + (M.l - 8) + '" y="' + (PY(g) + 4) + '" text-anchor="end" font-size="11" fill="var(--ink3)" font-family="IBM Plex Mono,monospace">' + g + '</text>';
  }
  // x ticks
  if (state.align === "cal") {
    YEARS.forEach((y, i) => {
      if (i % 2 === 0 || i === N - 1) s += '<text x="' + PX(d, i) + '" y="' + (CH - M.b + 17) + '" text-anchor="middle" font-size="11" fill="var(--ink3)" font-family="IBM Plex Mono,monospace">' + String(y).slice(2) + "'" + '</text>';
    });
  } else {
    for (let r = -10; r <= 10; r += 5) {
      const x = M.l + ((r + (N - 1)) / (2 * (N - 1))) * (CW - M.l - M.r);
      s += '<text x="' + x + '" y="' + (CH - M.b + 17) + '" text-anchor="middle" font-size="11" fill="var(--ink3)" font-family="IBM Plex Mono,monospace">' + (r === 0 ? "peak" : (r > 0 ? "+" : "") + r) + '</text>';
    }
  }
  // ensemble ghost
  for (const o of view) {
    if (o === d) continue;
    s += '<polyline points="' + o.pct.map((v, i) => PX(o, i).toFixed(1) + "," + PY(v).toFixed(1)).join(" ") +
      '" fill="none" stroke="var(--ghost)" stroke-width="1"/>';
  }
  // focused trace
  const pts = d.pct.map((v, i) => PX(d, i).toFixed(1) + "," + PY(v).toFixed(1)).join(" ");
  s += '<polyline points="' + pts + '" fill="none" stroke="var(--trace)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>';
  d.pct.forEach((v, i) => {
    s += '<circle cx="' + PX(d, i).toFixed(1) + '" cy="' + PY(v).toFixed(1) + '" r="3.2" fill="var(--trace)" stroke="var(--surface)" stroke-width="1.5"/>';
  });
  // markers
  const pi = YEARS.indexOf(d.peak), mi = YEARS.indexOf(d.minY);
  s += '<circle cx="' + PX(d, pi).toFixed(1) + '" cy="' + PY(100) + '" r="5" fill="var(--peak)" stroke="var(--surface)" stroke-width="2"/>';
  s += '<line x1="' + PX(d, mi).toFixed(1) + '" y1="' + PY(d.pct[mi]).toFixed(1) + '" x2="' + PX(d, mi).toFixed(1) + '" y2="' + PY(0) + '" stroke="var(--min)" stroke-width="1" stroke-dasharray="2 3"/>';
  s += '<circle cx="' + PX(d, mi).toFixed(1) + '" cy="' + PY(d.pct[mi]).toFixed(1) + '" r="5" fill="var(--surface)" stroke="var(--min)" stroke-width="2"/>';
  const lx = Math.min(Math.max(PX(d, mi), M.l + 30), CW - M.r - 30);
  s += '<text x="' + lx + '" y="' + Math.min(PY(d.pct[mi]) + 20, CH - M.b - 6) + '" text-anchor="middle" font-size="11" fill="var(--min)" font-family="IBM Plex Mono,monospace">' + d.minP.toFixed(0) + '%</text>';
  s += '<g id="cross"></g>';
  s += '<text x="' + M.l + '" y="' + (M.t - 3) + '" font-size="10.5" fill="var(--ink3)">% of peak</text>';
  $("dChart").innerHTML = '<title id="dChartTitle">' + esc(d.w) + ': percent of peak by year</title>' + s;

  /* --- hits strip --- */
  const HH = 78, hb = 20, kmax = Math.max(...d.k, 1);
  let hs = '<text x="' + M.l + '" y="10" font-size="10.5" fill="var(--ink3)">slang-sense hits per year</text>';
  const bw = Math.max(4, (CW - M.l - M.r) / (state.align === "cal" ? N : 2 * N) - 3);
  d.k.forEach((k, i) => {
    const h = Math.max(1, (k / kmax) * (HH - hb - 16));
    hs += '<rect x="' + (PX(d, i) - bw / 2).toFixed(1) + '" y="' + (HH - hb - h).toFixed(1) + '" width="' + bw.toFixed(1) +
      '" height="' + h.toFixed(1) + '" rx="2" fill="var(--trace)" opacity="' + (i === mi ? ".95" : ".38") + '"/>';
  });
  hs += '<text x="' + (M.l - 8) + '" y="' + (HH - hb) + '" text-anchor="end" font-size="10" fill="var(--ink3)" font-family="IBM Plex Mono,monospace">0</text>';
  hs += '<text x="' + (M.l - 8) + '" y="' + (HH - hb - (HH - hb - 16)) + '" text-anchor="end" font-size="10" fill="var(--ink3)" font-family="IBM Plex Mono,monospace">' + fmt(kmax) + '</text>';
  $("dHits").innerHTML = '<title id="dHitsTitle">' + esc(d.w) + ': slang-sense hits per year</title>' + hs;

  $("dCap").innerHTML = d.nse < 1.6
    ? 'This trajectory is within <b>' + d.nse.toFixed(1) + '&times;</b> the counting-noise floor &mdash; its shape is not evidence of a lifecycle.'
    : 'Movement is <b>' + d.nse.toFixed(1) + '&times;</b> the counting-noise floor, so the shape carries real signal.';

  $("dTable").querySelector("tbody").innerHTML = YEARS.map((y, i) => {
    const se = d.k[i] > 0 ? (100 / Math.sqrt(d.k[i])).toFixed(1) : "—";
    return '<tr><td class="mono">' + y + '</td><td class="mono">' + d.pct[i].toFixed(1) + '</td><td class="mono">' +
      d.rate[i].toFixed(3) + '</td><td class="mono">' + fmt(d.k[i]) + '</td><td class="mono">' + se + '</td></tr>';
  }).join("");

  $("dRead").innerHTML = "&nbsp;";
}

/* ---- crosshair ---- */
function hover(ev){
  const d = view[state.sel];
  if (!d) return;
  const svg = $("dChart"), r = svg.getBoundingClientRect();
  const x = ((ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left) / r.width * CW;
  let best = 0, bd = Infinity;
  d.pct.forEach((_, i) => { const dd = Math.abs(PX(d, i) - x); if (dd < bd) { bd = dd; best = i; } });
  const g = svg.querySelector("#cross");
  if (g) g.innerHTML = '<line x1="' + PX(d, best).toFixed(1) + '" y1="' + M.t + '" x2="' + PX(d, best).toFixed(1) + '" y2="' + PY(0) +
    '" stroke="var(--ink3)" stroke-width="1" stroke-dasharray="2 2"/>';
  const se = d.k[best] > 0 ? " ±" + (100 / Math.sqrt(d.k[best])).toFixed(1) + "%" : "";
  $("dRead").textContent = YEARS[best] + "  —  " + d.pct[best].toFixed(1) + "% of peak  ·  " +
    d.rate[best].toFixed(3) + "/M  ·  " + fmt(d.k[best]) + " hits" + se;
}
function unhover(){
  const g = $("dChart").querySelector("#cross"); if (g) g.innerHTML = "";
  $("dRead").innerHTML = "&nbsp;";
}

/* ---- wiring ---- */
function select(i){
  state.sel = Math.max(0, Math.min(view.length - 1, i));
  document.querySelectorAll(".card").forEach((el, k) => el.setAttribute("aria-current", String(k === state.sel)));
  renderDetail();
  const el = document.querySelector('.card[data-i="' + state.sel + '"]');
  if (el) el.scrollIntoView({block:"nearest", behavior:"smooth"});
}
function refresh(keepWord){
  const w = keepWord && view[state.sel] ? view[state.sel].w : null;
  sortView();
  const at = w ? view.findIndex((d) => d.w === w) : -1;
  state.sel = at >= 0 ? at : 0;
  renderCards(); renderDetail();
}
$("cards").addEventListener("click", (e) => {
  const b = e.target.closest(".card"); if (b) select(+b.dataset.i);
});
$("sort").addEventListener("change", (e) => { state.sort = e.target.value; refresh(true); });
$("align").addEventListener("change", (e) => { state.align = e.target.value; refresh(true); });
$("q").addEventListener("input", (e) => { state.q = e.target.value; refresh(true); });
$("prev").addEventListener("click", () => select(state.sel - 1));
$("next").addEventListener("click", () => select(state.sel + 1));
$("dChart").addEventListener("mousemove", hover);
$("dChart").addEventListener("touchmove", hover, {passive:true});
$("dChart").addEventListener("mouseleave", unhover);
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowRight" || e.key === "ArrowDown") { select(state.sel + 1); e.preventDefault(); }
  if (e.key === "ArrowLeft" || e.key === "ArrowUp") { select(state.sel - 1); e.preventDefault(); }
});
refresh(false);
</script>
'''


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                   help=f"peak_years JSON with percentages (default: {DEFAULT_INPUT.name}).")
    p.add_argument("--min-hits", type=int, default=80, metavar="N", dest="min_hits",
                   help="Skip words with fewer than N total slang-sense hits (default: 80, "
                        "matching figures/min_pct_of_peak.png).")
    p.add_argument("--max-min-pct", type=float, default=50.0, metavar="PCT", dest="max_min_pct",
                   help="Skip words whose lowest year sits above PCT%% of their own peak "
                        "(default: 50, matching figures/min_pct_of_peak.png).")
    p.add_argument("--top", type=int, metavar="N", help="Only the N words with the lowest minimum.")
    p.add_argument("--sizes-cache", type=Path, default=DEFAULT_SIZES, metavar="FILE",
                   dest="sizes_cache", help=f"Per-dump token totals (default: {DEFAULT_SIZES.name}).")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output HTML (default: {DEFAULT_OUTPUT}).")
    args = p.parse_args()

    if not args.input.is_file():
        p.error(f"Input not found: {args.input}")

    words, meta = build(args.input, args.sizes_cache, args.min_hits, args.max_min_pct, args.top)
    html = (TEMPLATE
            .replace("__YEARS__", json.dumps(meta["years"]))
            .replace("__TOKENS__", json.dumps(meta["tokens"]))
            .replace("__DATA__", json.dumps(words, ensure_ascii=True))
            .replace("__N__", str(meta["n"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Saved: {args.output}  ({meta['n']} words)")


if __name__ == "__main__":
    main()
