"""Plot revision Figures 5-11 from saved out-of-fold SHAP values; no model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import string
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd

from schema import read_main


FEATURES = [
    "BD", "PR", "MD", "ED", "ResiD", "ECED", "SID", "SI", "Mix", "FL",
    "SFA", "NDVI", "NDWI", "DSS", "DBS", "IRD", "RD", "DCBD", "DRA", "DSF",
]
LABELS = dict(zip(FEATURES, [
    "BD", "FAR", "MFD", "ED", "ResiD", "ESCFD", "SFD", "POISI", "SFSI",
    "SFL radius", "SFA", "NDVI", "NDWI", "DSS", "DBS", "ID", "RD", "DCBD", "DRA", "DSF",
]))
DATES = {
    "SV_weekday": {"label": "18 March 2025 (weekday)", "color": "#436A89", "marker": "o"},
    "SV_weekend": {"label": "22 March 2025 (weekend)", "color": "#B87260", "marker": "^"},
}
GROUPS = {
    7: ("Density", ["BD", "PR", "MD", "ResiD", "ED", "ECED", "SID"]),
    8: ("Diversity", ["SI", "Mix"]),
    9: ("Design", ["FL", "SFA", "NDVI", "NDWI"]),
    10: ("Transit", ["DSS", "DBS", "IRD", "RD"]),
    11: ("Destination", ["DCBD", "DRA", "DSF"]),
}
WIDTH_IN = 6.2
MIN_FONT_PT = 7.2
SFA_LINTHRESH = 10000.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def figure_contracts() -> dict:
    contracts = {
        5: {
            "conclusion": "Relative model attribution is concentrated in a subset of the 20 predictors on each observed date.",
            "evidence": "Paired bars show mean absolute held-out SHAP values divided by their sum within each date, for all 20 predictors.",
            "risk": "Relative attribution is conditional on the fitted model and selected predictors, and is not an effect size or causal ranking.",
            "panel_map": "One paired-bar panel, common order by the mean of the two date-specific relative shares.",
            "height_in": 6.6,
        },
        6: {
            "conclusion": "Facilities exhibit heterogeneous signed model contributions for each predictor on both observed dates.",
            "evidence": "Two matched 20-predictor beeswarm summaries retain every held-out SHAP value, with feature-value color encoded within predictor.",
            "risk": "Point density, signed SHAP and feature-value color do not establish causal effects, statistical significance or thresholds.",
            "panel_map": "a: selected weekday; b: selected weekend; common predictor order; independent SHAP x scales.",
            "height_in": 7.2,
        },
    }
    for number, (name, features) in GROUPS.items():
        columns = 3 if number in (7, 11) else 2
        rows = math.ceil(len(features) / columns)
        contracts[number] = {
            "conclusion": f"The {name.lower()} predictors show the model's facility-level attribution patterns over their observed values on two selected dates.",
            "evidence": f"One scatter panel for each of {', '.join(LABELS[f] for f in features)}, with both dates distinguished by color and marker.",
            "risk": "SHAP values are date-specific model contributions, not causal responses; no threshold, significance test, or smoothed trend is inferred.",
            "panel_map": {string.ascii_lowercase[i]: LABELS[f] for i, f in enumerate(features)},
            "height_in": 2.45 * rows + 0.65,
        }
    return {
        "backend": "Python / matplotlib",
        "archetype": "quantitative grid",
        "width_in": WIDTH_IN,
        "minimum_font_pt": MIN_FONT_PT,
        "formats": ["SVG (editable text)", "PDF (embedded TrueType text)", "PNG (600 dpi)"],
        "n_definition": "1,650 distinct facilities; each appears once in an outer test fold for each date.",
        "validation": "Five spatial outer folds using 2 km grid blocks and a 500 m training-exclusion buffer; three-fold inner tuning.",
        "dates": [d["label"] for d in DATES.values()],
        "statistics": "Descriptive model attributions; no error bars, hypothesis tests, p-values or causal estimands.",
        "integrity": "No observations are excluded; no outcome or predictor rescaling; Figure 6 color normalization and vertical packing are display-only; SFA uses an explicitly labelled symlog x axis.",
        "figures": {str(k): v for k, v in contracts.items()},
    }


def configure_style() -> None:
    global matplotlib, plt, colors, ticker, Line2D, Image, ImageDraw, FEATURE_CMAP
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors, ticker
    from matplotlib.lines import Line2D
    from PIL import Image, ImageDraw

    FEATURE_CMAP = colors.LinearSegmentedColormap.from_list(
        "feature_values", ["#3D7097", "#B7BBC3", "#B66460"]
    )
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": MIN_FONT_PT,
        "axes.labelsize": 7.6,
        "axes.titlesize": 8.0,
        "xtick.labelsize": MIN_FONT_PT,
        "ytick.labelsize": MIN_FONT_PT,
        "legend.fontsize": MIN_FONT_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_sources(root: Path, source: Path, encoding: str = "utf-8-sig", allow_unverified_input_hash: bool = False) -> tuple[pd.DataFrame, dict, dict, list]:
    data = read_main(source, encoding=encoding, expected_n=1650)
    assert set(data["FL"].unique()) == {500, 1000, 2500, 5000}
    data.index = np.arange(1, len(data) + 1)
    data.index.name = "row_id"
    shap_values, importances = {}, {}
    input_digest = sha256(source)
    manifest = root / "run_manifest.json"
    saved_digest = None
    manifest_input_key = None
    if manifest.exists():
        sources = json.loads(manifest.read_text(encoding="utf-8")).get("sources", {})
        for key in ("main_input", "0728.csv"):
            if sources.get(key, {}).get("sha256"):
                saved_digest = sources[key]["sha256"]
                manifest_input_key = key
                break
        if saved_digest is not None and saved_digest != input_digest:
            raise ValueError("Predictor input does not match the saved analysis manifest")
    if saved_digest is None and not allow_unverified_input_hash:
        raise ValueError("No compatible saved input hash; independently verify original row identity before explicitly allowing an unverified input hash")
    provenance = [{"role": "predictor data", "file_name": source.name, "sha256": input_digest,
                   "manifest_input_key": manifest_input_key,
                   "input_hash_matches_analysis_manifest": saved_digest is not None,
                   "hash_check_note": "Verified against packaged run manifest" if saved_digest else "No compatible input hash in saved analysis; verify row identity separately"}]
    reference_folds = None
    for target in DATES:
        prefix = f"spatial_2km_buffer500_{target}_XGBoost"
        values_path = root / "shap" / f"{prefix}_values.csv"
        importance_path = root / "shap" / f"{prefix}_importance.csv"
        values = pd.read_csv(values_path).set_index("row_id")
        assert values.index.is_unique and set(values.index) == set(data.index)
        values = values.reindex(data.index)
        assert np.isfinite(values[FEATURES].to_numpy()).all()
        assert set(values["fold"].unique()) == {1, 2, 3, 4, 5}
        if reference_folds is None:
            reference_folds = values["fold"].to_numpy()
        else:
            assert np.array_equal(reference_folds, values["fold"].to_numpy())
        computed = np.abs(values[FEATURES]).mean()
        saved = pd.read_csv(importance_path).set_index("feature")
        assert saved.index.is_unique
        assert np.allclose(computed.values, saved.loc[FEATURES, "mean_abs_SHAP"].values, rtol=1e-7, atol=1e-7)
        assert computed.sum() > 0, "Relative SHAP shares are undefined when every attribution is zero"
        importance = pd.DataFrame({"mean_abs_SHAP": computed, "relative_percent": computed / computed.sum() * 100})
        shap_values[target] = values
        importances[target] = importance
        for role, path in [("held-out SHAP", values_path), ("saved importance", importance_path)]:
            provenance.append({"role": role, "target": target, "relative_path": path.relative_to(root).as_posix(), "sha256": sha256(path)})
    return data, shap_values, importances, provenance


def save_figure(fig, number: int, out: Path, records: dict) -> None:
    width, height = fig.get_size_inches()
    assert width <= 6.21 and height <= 8.6
    fig.canvas.draw()
    inactive_ticks = set()
    for axes in fig.axes:
        for axis, limits in ((axes.xaxis, axes.get_xlim()), (axes.yaxis, axes.get_ylim())):
            low, high = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if tick.get_loc() < low - 1e-9 or tick.get_loc() > high + 1e-9:
                    inactive_ticks.update((id(tick.label1), id(tick.label2)))
    text_artists = [a for a in fig.findobj(matplotlib.text.Text)
                    if a.get_text() and a.get_visible() and id(a) not in inactive_ticks]
    assert all(a.get_fontsize() >= MIN_FONT_PT - 0.05 for a in text_artists)
    canvas_box = fig.bbox
    renderer = fig.canvas.get_renderer()
    outside_text = []
    for artist in text_artists:
        bounds = artist.get_window_extent(renderer=renderer)
        if bounds.x0 < -0.5 or bounds.y0 < -0.5 or bounds.x1 > canvas_box.x1 + 0.5 or bounds.y1 > canvas_box.y1 + 0.5:
            outside_text.append(artist.get_text())
    assert not outside_text, f"Figure {number} text exceeds canvas: {outside_text}"
    for legend in fig.legends:
        legend_bounds = legend.get_window_extent(renderer=renderer)
        for axes in fig.axes:
            for title in (axes.title, axes._left_title, axes._right_title):
                if title.get_text() and legend_bounds.overlaps(title.get_window_extent(renderer=renderer)):
                    raise AssertionError(f"Figure {number} legend overlaps panel title {title.get_text()}")
    paths = {}
    for extension in ("svg", "pdf", "png"):
        path = out / f"Figure{number}.{extension}"
        fig.savefig(path, dpi=600)
        paths[extension] = path.name
    image = Image.open(out / paths["png"]).convert("RGB")
    pixels = np.asarray(image)
    nonwhite = float((pixels.min(axis=2) < 245).mean())
    assert nonwhite > 0.01
    svg = (out / paths["svg"]).read_text(encoding="utf-8")
    assert "<text" in svg
    records[str(number)] = {
        "files": paths, "dimensions_in": [float(width), float(height)],
        "png_pixels": list(image.size), "png_dpi": 600,
        "nonwhite_pixel_fraction": nonwhite, "svg_text_nodes": svg.count("<text"),
        "minimum_font_pt": min(a.get_fontsize() for a in text_artists),
        "rendered_text_count": len(text_artists), "visual_inspection": "pending human/tool image inspection",
        "text_outside_canvas": outside_text,
        "legend_overlaps_panel_title": False,
        "PDF_embedded_TrueType_font": b"/FontFile2" in (out / paths["pdf"]).read_bytes(),
        "scatter_point_count": sum(len(collection.get_offsets()) for axes in fig.axes for collection in axes.collections
                                   if isinstance(collection, matplotlib.collections.PathCollection)),
    }
    plt.close(fig)


def plot_importance(importances: dict, order: list, out: Path, records: dict) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 6.6))
    y = np.arange(len(order))
    for j, (target, settings) in enumerate(DATES.items()):
        values = importances[target].loc[order, "relative_percent"].to_numpy()
        offset = -0.18 if j == 0 else 0.18
        ax.barh(y + offset, values, height=0.33, color=settings["color"], label=settings["label"])
        for yi, value in zip(y + offset, values):
            ax.text(value + 0.25, yi, f"{value:.1f}", va="center", ha="left", fontsize=MIN_FONT_PT)
    ax.set_yticks(y, [LABELS[f] for f in order])
    ax.invert_yaxis()
    ax.set_xlabel("Relative mean absolute SHAP (%)")
    largest = max(importances[t]["relative_percent"].max() for t in DATES)
    ax.set_xlim(0, math.ceil((largest + 3.2) / 5) * 5)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.10), ncol=1, handlelength=1.5)
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.08, top=0.90)
    save_figure(fig, 5, out, records)


def packed_offsets(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Vertical packing changes display position only; every SHAP x value is retained.
    bins = np.clip(np.floor((x - x.min()) / max(np.ptp(x), 1e-12) * 80).astype(int), 0, 80)
    offsets = np.zeros(len(x))
    for bin_id in np.unique(bins):
        indices = np.flatnonzero(bins == bin_id)
        rng.shuffle(indices)
        steps = np.arange(len(indices))
        signed = np.where(steps % 2 == 0, steps // 2, -(steps + 1) // 2)
        extent = max(abs(signed).max(), 1)
        offsets[indices] = signed / extent * 0.34 * min(1.0, len(indices) / 35)
    return offsets


def plot_summary(data: pd.DataFrame, values: dict, order: list, out: Path, records: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 7.2), sharey=True)
    for j, (target, settings) in enumerate(DATES.items()):
        ax = axes[j]
        rng = np.random.default_rng(1009 + j)
        for row, feature in enumerate(order):
            x = values[target][feature].to_numpy()
            raw = data[feature].to_numpy()
            feature_color = (raw - raw.min()) / max(np.ptp(raw), 1e-12)
            permutation = rng.permutation(len(x))
            y = row + packed_offsets(x, rng)
            ax.scatter(x[permutation], y[permutation], c=feature_color[permutation], cmap=FEATURE_CMAP,
                       vmin=0, vmax=1, s=3.5, alpha=0.55, linewidths=0, rasterized=True)
        ax.axvline(0, color="#999999", lw=0.55, zorder=0)
        ax.set_yticks(np.arange(len(order)), [LABELS[f] for f in order])
        ax.set_ylim(len(order) - 0.35, -0.65)
        ax.set_title(f"{string.ascii_lowercase[j]}  {settings['label']}", loc="left", fontweight="bold", pad=12)
        ax.set_xlabel("SHAP value (date-specific outcome scale)", fontsize=MIN_FONT_PT)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    cax = fig.add_axes([0.885, 0.17, 0.018, 0.59])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=FEATURE_CMAP), cax=cax)
    cb.set_ticks([0, 1], labels=["Low", "High"])
    cb.set_label("Feature value (within predictor)", fontsize=MIN_FONT_PT)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.16, right=0.855, bottom=0.085, top=0.935, wspace=0.20)
    save_figure(fig, 6, out, records)


def dependency_xlabel(feature: str) -> str:
    if feature == "FL":
        return "Assigned planning radius\n(m; assumed)"
    if feature == "SFA":
        return "SFA (recorded area; symlog)"
    if feature == "DRA":
        return "DRA (recorded scale;\nunit pending)"
    return LABELS[feature]


def plot_dependency(number: int, data: pd.DataFrame, values: dict, out: Path, records: dict) -> None:
    _, features = GROUPS[number]
    columns = 3 if number in (7, 11) else 2
    rows = math.ceil(len(features) / columns)
    height = 2.45 * rows + 0.65
    fig, axes = plt.subplots(rows, columns, figsize=(WIDTH_IN, height), squeeze=False)
    for panel, feature in enumerate(features):
        ax = axes.flat[panel]
        raw = data[feature].to_numpy()
        for target, settings in DATES.items():
            ax.scatter(raw, values[target][feature].to_numpy(), color=settings["color"],
                       marker=settings["marker"], s=7 if settings["marker"] == "^" else 6,
                       alpha=0.38, edgecolors="none", linewidths=0)
        ax.axhline(0, color="#A3A3A3", linewidth=0.5, zorder=0)
        ax.set_title(f"{string.ascii_lowercase[panel]}  {LABELS[feature]}", loc="left", fontweight="bold", pad=6)
        ax.set_xlabel(dependency_xlabel(feature), labelpad=3)
        if panel % columns == 0:
            ax.set_ylabel("SHAP value")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
        if feature == "FL":
            ax.set_xticks([500, 1000, 2500, 5000], ["500", "1,000", "2,500", "5,000"])
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_xlim(300, 5200)
        elif feature == "SFA":
            ax.set_xscale("symlog", linthresh=SFA_LINTHRESH, linscale=1.0, base=10)
            ax.set_xticks([0, 10000, 100000, 1000000], ["0", "10k", "100k", "1M"])
            ax.set_xlim(0, raw.max() * 1.09)
        elif raw.max() >= 1000:
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#777777")
    for empty in list(axes.flat)[len(features):]:
        empty.axis("off")
    handles = [Line2D([], [], color=s["color"], marker=s["marker"], ls="", markersize=4,
                      label=s["label"]) for s in DATES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 0.99), ncol=2, columnspacing=1.6)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20 if rows == 1 else 0.10,
                        top=1 - 0.54 / height, wspace=0.43, hspace=0.56)
    save_figure(fig, number, out, records)


def make_contact_sheet(out: Path) -> str:
    thumbnails = []
    for number in range(5, 12):
        im = Image.open(out / f"Figure{number}.png").convert("RGB")
        im.thumbnail((760, 880), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (800, 930), "white")
        tile.paste(im, ((800 - im.width) // 2, 38))
        ImageDraw.Draw(tile).text((18, 12), f"Figure {number}", fill="black")
        thumbnails.append(tile)
    sheet = Image.new("RGB", (800 * 3, 930 * 3), "#E9E9E9")
    for i, tile in enumerate(thumbnails):
        sheet.paste(tile, ((i % 3) * 800, (i // 3) * 930))
    name = "Figures5-11_contact_sheet.png"
    sheet.save(out / name)
    return name


def legends() -> dict:
    dates = ("The 1,650 facilities were observed on 18 March 2025 (weekday) and 22 March 2025 "
             "(weekend), not multi-day averages. ")
    heldout = "Each point is an outer-fold held-out XGBoost SHAP attribution (validation: Figure 5). "
    caution = "Values use date-specific outcome scales and model baselines, not causal or date effects."
    captions = {
        5: (
            "Figure 5. Relative importance of the 20 predictors. " + dates +
            "Contributions are held-out XGBoost SHAP values from five spatial outer folds "
            "(2 km blocks; training facilities within 500 m of any test facility excluded), "
            "with three-fold inner tuning. Relative importance is each predictor's mean absolute SHAP "
            "divided by their sum, multiplied by 100, separately for each date. Display order follows "
            "the average relative share across dates; labels are rounded to one decimal. Rankings are "
            "conditional on the model specification, not causal importance. The response is a heatmap "
            "proxy, not measured sports participation."
        ),
        6: (
            "Figure 6. Held-out SHAP summaries for the selected weekday (a) and weekend (b). " + dates + heldout +
            "Horizontal position is the unmodified signed contribution; vertical packing separates "
            "overlapping points only. Color maps each predictor's observed minimum to blue and maximum "
            "to muted red, for display only; predictors and outcomes were not normalized. All points "
            "are retained. Both panels share Figure 5's predictor order but use separate horizontal "
            "scales. " + caution
        ),
    }
    for number, (name, features) in GROUPS.items():
        mapping = "; ".join(f"({string.ascii_lowercase[i]}) {LABELS[f]}" for i, f in enumerate(features))
        extra = ""
        if number == 9:
            extra = (
                "SFL encodes assigned planning radii of 500, 1,000, 2,500 and 5,000 m; these are "
                "analysis assumptions, not observed use distances. SFA uses a base-10 symlog x axis, "
                "linear below 10,000 recorded area units; values are unaltered, including the 2,850,000 maximum. "
            )
        if number == 11:
            extra = (
                "DRA is plotted on its recorded scale (0-3,344.23); the upstream unit and possible "
                "conversion remain unconfirmed. No division by 100 was applied. "
            )
        captions[number] = (
            f"Figure {number}. {name}-predictor SHAP dependence scatterplots: {mapping}. " + dates + heldout +
            "Blue circles denote weekday and muted-red triangles weekend; every point is retained. " +
            extra + caution
        )
    for number, caption in captions.items():
        assert len(caption.split()) <= 130, f"Figure {number} caption exceeds 130 words"
    return {
        "status": "Revision figure captions for manuscript integration; not a declaration that unresolved upstream units are settled.",
        "figures": [{"figure_no": n, "png": f"Figure{n}.png", "svg": f"Figure{n}.svg", "pdf": f"Figure{n}.pdf",
                     "caption_en": captions[n], "caption_word_count": len(captions[n].split()), "n_facilities_per_date": 1650,
                     "dates": ["2025-03-18", "2025-03-22"], "causal_claim": False}
                    for n in range(5, 12)],
        "pending_author_checks": ["Confirm DRA upstream unit and scale conversion before final submission.",
                                  "Independently verify upstream GIS buffer implementation; assigned SFL planning radii are not observed use distances.",
                                  "Confirm recorded SFA area-unit definition against upstream GIS documentation.",
                                  "Use the updated method and 20-feature definitions with these new figures; do not retain old threshold claims."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Authorized 1,650-row predictor CSV in the original model row order.")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding; use gbk for the historical input.")
    parser.add_argument("--allow-unverified-input-hash", action="store_true", help="Legacy analyses only: explicitly acknowledge absent input-hash verification after independently verifying row identity.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metadata-only", action="store_true", help="Write contracts and captions without rendering.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    contract = figure_contracts()
    write_json(args.out / "figure_contract.json", contract)
    write_json(args.out / "legend_source.json", legends())
    if args.metadata_only:
        print(json.dumps({"metadata_only": True, "output": str(args.out)}))
        return
    configure_style()
    data, values, importances, provenance = load_sources(args.analysis_root, args.input, args.encoding, args.allow_unverified_input_hash)
    pooled = sum(importances[t]["relative_percent"] for t in DATES) / len(DATES)
    order = pooled.sort_values(ascending=False, kind="stable").index.tolist()
    data[FEATURES].to_csv(args.out / "source_predictors.csv", index=True, encoding="utf-8")
    for target, frame in values.items():
        frame[FEATURES + ["fold"]].to_csv(args.out / f"source_OOF_SHAP_{target}.csv", index=True, encoding="utf-8")
    importance_rows = []
    for target, table in importances.items():
        for feature, row in table.iterrows():
            importance_rows.append({"target": target, "feature": feature, "label": LABELS[feature],
                                    "mean_abs_SHAP": row["mean_abs_SHAP"], "relative_percent": row["relative_percent"]})
    pd.DataFrame(importance_rows).to_csv(args.out / "source_relative_importance.csv", index=False, encoding="utf-8")
    records = {}
    plot_importance(importances, order, args.out, records)
    plot_summary(data, values, order, args.out, records)
    for number in GROUPS:
        plot_dependency(number, data, values, args.out, records)
    contact = make_contact_sheet(args.out)
    write_json(args.out / "legend_source.json", legends())
    qa = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python / matplotlib", "matplotlib_version": matplotlib.__version__,
        "plotting_environment": {"python": platform.python_version(),
                                 **{package: version(package) for package in ("matplotlib", "numpy", "pandas", "scipy", "Pillow")}},
        "original_model_analysis_environment": json.loads((args.analysis_root / "environment.json").read_text(encoding="utf-8")),
        "model_refit_performed": False, "n_facilities": len(data), "n_features": len(FEATURES),
        "point_counts": {"Figure6": 1650 * 20 * 2,
                         **{f"Figure{n}": 1650 * len(f) * 2 for n, (_, f) in GROUPS.items()}},
        "all_predictor_and_shap_values_finite": True, "row_id_alignment_verified": True,
        "same_outer_fold_assignments_for_both_dates": True,
        "computed_mean_absolute_SHAP_matches_saved_importance": True,
        "tail_trimming": False, "predictor_jitter": False, "outcome_normalization": False,
        "DRA_scale_conversion": "none; unit pending", "SFA_axis": {"type": "symlog", "base": 10, "linthresh": SFA_LINTHRESH},
        "feature_value_ranges": {f: {"min": float(data[f].min()), "max": float(data[f].max())} for f in FEATURES},
        "provenance": provenance, "figures": records, "contact_sheet": contact,
        "pending": ["Visual inspection of all seven PNGs", "Author confirmation of DRA and SFA upstream unit definitions"],
    }
    write_json(args.out / "figure_qa.json", qa)
    print(json.dumps({"output": str(args.out), "figures": list(records), "rows": len(data), "features": len(FEATURES)}))


if __name__ == "__main__":
    main()
