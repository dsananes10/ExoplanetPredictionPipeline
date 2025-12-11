from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from matplotlib.patches import Wedge
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
TRAIN_PATH = PROC / "training_data.csv"
MODEL_XGB_PATH = PROC / "model_is_real_xgb.pkl"
VIS_DIR = ROOT / "visualizations"


def planet_color(size_class: str) -> str:
    colors = {
        "terrestrial": "#4CAF50",
        "super_earth": "#2196F3",
        "gas_dwarf": "#9C27B0",
        "neptunian": "#3F51B5",
        "gas_giant": "#FFC107",
        "unknown": "#DDDDDD",
    }
    return colors.get(str(size_class).lower(), "#DDDDDD")


def orbit_color(temp_regime: str) -> str:
    colors = {
        "F": "#00BFFF",
        "W": "#32CD32",
        "G": "#FF8C00",
        "R": "#FF0000",
        "U": "#888888",
    }
    return colors.get(str(temp_regime).upper(), "#888888")


def load_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


def load_model_bundle():
    return joblib.load(MODEL_XGB_PATH)


def predict_for_rows(
    model,
    threshold: float,
    feature_cols: List[str],
    df_rows: pd.DataFrame,
) -> pd.DataFrame:
    df = df_rows.copy()
    X = df[feature_cols]
    mask = X.notna().all(axis=1)

    proba = np.full(len(df), np.nan, dtype=float)
    pred = np.full(len(df), np.nan, dtype=float)
    if mask.any():
        X_clean = X[mask].astype(float).values
        proba_clean = model.predict_proba(X_clean)[:, 1]
        pred_clean = (proba_clean >= threshold).astype(int)
        proba[mask.values] = proba_clean
        pred[mask.values] = pred_clean

    df["model_is_real_proba"] = proba
    df["model_is_real_pred"] = pred
    return df


def disposition_upper(val) -> str:
    if not isinstance(val, str):
        return ""
    return val.upper().strip()


def format_property_block(row: pd.Series) -> str:
    name = str(row.get("kepoi_name", "UNKNOWN"))
    disp = disposition_upper(row.get("koi_disposition", ""))
    proba = row.get("model_is_real_proba", np.nan)
    pred = row.get("model_is_real_pred", np.nan)

    if disp == "CONFIRMED":
        nasa_decision = "REAL PLANET"
    elif disp == "FALSE POSITIVE":
        nasa_decision = "FALSE POSITIVE"
    elif disp == "CANDIDATE":
        nasa_decision = "CANDIDATE"
    else:
        nasa_decision = "UNKNOWN"

    if pd.isna(proba):
        prob_line = "-- model probability for this KOI is unknown --"
    else:
        prob_pct = int(round(float(proba) * 100))
        prob_line = f"-- {prob_pct}% chance the KOI is real --"

    if pd.isna(pred):
        ml_decision = "UNKNOWN"
    else:
        ml_decision = "REAL PLANET" if int(pred) == 1 else "FALSE POSITIVE"

    if nasa_decision in ("REAL PLANET", "FALSE POSITIVE") and ml_decision != "UNKNOWN":
        agreement = "AGREE" if nasa_decision == ml_decision else "DISAGREE"
    else:
        agreement = "N/A"

    lines = [
        f"KOI: {name}",
        f"NASA decision: {nasa_decision}",
        prob_line,
        f"ML decision: {ml_decision}",
        f"Agreement with NASA: {agreement}",
    ]

    show_details = (nasa_decision == "REAL PLANET") or (ml_decision == "REAL PLANET")
    if not show_details:
        return "\n".join(lines)

    type_label = row.get("combined_type_label", "")
    if not isinstance(type_label, str) or not type_label.strip():
        type_label = "Unknown world"
    lines.append(f"Planet Type: {type_label}")

    if "koi_period" in row.index and not pd.isna(row["koi_period"]):
        try:
            lines.append(f"Orbit: ~{float(row['koi_period']):.1f} days")
        except Exception:
            pass

    if "koi_teq" in row.index and not pd.isna(row["koi_teq"]):
        try:
            lines.append(f"Temperature: ~{float(row['koi_teq']):.0f} K (equilibrium)")
        except Exception:
            pass

    desc = row.get("combined_type_desc", "")
    if isinstance(desc, str) and desc.strip():
        lines.append(f"Planet Desc.: {desc.strip()}")

    return "\n".join(lines)


def create_star_system_plot(df_star: pd.DataFrame, kepid: int, out_path: Path):
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    df = df_star.copy()
    df["disp_upper"] = df["koi_disposition"].apply(disposition_upper)

    pred = df["model_is_real_pred"]
    mask = (
        ((df["disp_upper"] == "CONFIRMED") & (pred == 1.0))
        | ((df["disp_upper"] == "CANDIDATE") & (pred == 1.0))
    )
    df = df[mask]
    df = df[df["koi_period"].notna() & df["koi_prad"].notna()]

    if df.empty:
        return

    df = df.sort_values("koi_period").reset_index(drop=True)
    prad = df["koi_prad"].astype(float).values
    n_planets = len(df)

    star_r = 1.5
    cx = cy = 0.0

    spacing = 2.5
    base_orbit = star_r + 1.8
    orbit_radii = base_orbit + spacing * np.arange(n_planets)
    min_spacing = spacing

    max_pr = min(0.4 * min_spacing, 0.75 * star_r)
    min_pr = 0.25 * max_pr

    r_min, r_max = prad.min(), prad.max()
    if r_max == r_min:
        planet_r = np.full_like(prad, (min_pr + max_pr) / 2.0)
    else:
        norm = (prad - r_min) / (r_max - r_min)
        planet_r = min_pr + norm * (max_pr - min_pr)

    angles = np.linspace(0, 2 * np.pi, n_planets, endpoint=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    star = plt.Circle((cx, cy), star_r, color="#FFEFB5")
    ax.add_patch(star)
    ax.text(
        cx,
        cy - star_r - 0.4,
        f"Star {kepid}",
        color="white",
        ha="center",
        va="top",
        fontsize=12,
    )

    for (_, row), orbit_r, r, ang in zip(df.iterrows(), orbit_radii, planet_r, angles):
        temp_regime = row.get("temp_regime", None)
        o_color = orbit_color(temp_regime)

        orbit = plt.Circle(
            (cx, cy),
            orbit_r,
            color=o_color,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.add_patch(orbit)

        x = orbit_r * np.cos(ang)
        y = orbit_r * np.sin(ang)

        size_class = row.get("size_class", None)
        p_color = planet_color(size_class)

        planet = plt.Circle(
            (x, y),
            r,
            color=p_color,
            alpha=1.0,
            ec="#111111",
            lw=0.8,
        )
        ax.add_patch(planet)

        disp = disposition_upper(row.get("koi_disposition", ""))
        pred_val = row.get("model_is_real_pred", np.nan)
        if disp == "CANDIDATE" and not pd.isna(pred_val) and int(pred_val) == 1:
            halo = plt.Circle(
                (x, y),
                r * 1.15,
                fill=False,
                ec="white",
                lw=1.1,
                alpha=0.95,
            )
            ax.add_patch(halo)

        label = row.get("kepoi_name", "")
        ax.text(
            x,
            y + r + 0.18,
            str(label),
            color="white",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    max_orbit = orbit_radii.max() + max(planet_r) + 1.0
    ax.set_xlim(-max_orbit, max_orbit)
    ax.set_ylim(-max_orbit, max_orbit)
    ax.axis("off")

    size_labels = ["terrestrial", "super_earth", "gas_dwarf", "neptunian", "gas_giant"]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=planet_color(t),
            markersize=8,
        )
        for t in size_labels
    ]

    temp_labels = ["Frozen (F)", "Water (W)", "Gaseous (G)", "Roaster (R)"]
    temp_codes = ["F", "W", "G", "R"]
    temp_handles = [
        Line2D(
            [0],
            [0],
            color=orbit_color(code),
            linestyle="--",
            linewidth=2,
        )
        for code in temp_codes
    ]

    ml_halo_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="white",
        markerfacecolor="none",
        markersize=8,
        linewidth=1.1,
    )

    legend1 = ax.legend(
        size_handles,
        size_labels,
        title="Size class",
        loc="upper left",
        facecolor="#777777",
        edgecolor="white",
        fontsize=8,
        title_fontsize=9,
    )
    legend2 = ax.legend(
        temp_handles + [ml_halo_handle],
        temp_labels + ["candidate predicted REAL (ML halo)"],
        title="Orbit / temperature / ML",
        loc="upper right",
        facecolor="#777777",
        edgecolor="white",
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(legend1)
    for leg in (legend1, legend2):
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_linewidth(1.0)
        leg.get_title().set_color("white")
        for txt in leg.get_texts():
            txt.set_color("white")
    ax.add_artist(legend2)

    fig.suptitle(f"Star system for kepid {kepid}", color="white", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def handle_confirm_by_star(df: pd.DataFrame):
    kepid_str = input("\nEnter KEPID (ex: 3545478): ").strip()
    if not kepid_str.isdigit():
        print("Invalid KEPID. It must be an integer.")
        return
    kepid = int(kepid_str)

    df_star = df[df["kepid"] == kepid]
    if df_star.empty:
        print(f"No KOIs found for kepid {kepid}.")
        return

    total_koi = len(df_star)
    disp_upper = df_star["koi_disposition"].apply(disposition_upper)
    num_candidates = (disp_upper == "CANDIDATE").sum()

    print(f"\nStar: {kepid}")
    print("Distance: (not available)\n")
    print(f"KOIs: {total_koi}")
    print(f"Unconfirmed candidates: {num_candidates}\n")

    df_star = df_star.copy()
    df_star["disp_upper"] = disp_upper
    df_star["proba"] = df_star["model_is_real_proba"]

    def sort_block(block):
        return block.sort_values("proba", ascending=False)

    confirmed = sort_block(df_star[df_star["disp_upper"] == "CONFIRMED"])
    candidates = sort_block(df_star[df_star["disp_upper"] == "CANDIDATE"])
    fps = sort_block(df_star[df_star["disp_upper"] == "FALSE POSITIVE"])

    first = True
    for block in (confirmed, candidates, fps):
        if block.empty:
            continue
        for _, row in block.iterrows():
            if not first:
                print("\n" + "-" * 78 + "\n")
            print(format_property_block(row))
            first = False

    build = input("\n\nBuild the hypothetical star-system? (y/n): ").strip().lower()
    if build in ("y", "yes"):
        out_path = VIS_DIR / f"star_{kepid}_system.png"
        create_star_system_plot(df_star, kepid, out_path)

    print("\n" + "=" * 78 + "\n")


def handle_confirm_by_koi(df: pd.DataFrame):
    koi_id = input("\nEnter KOI (ex: K00353.03): ").strip()
    df_koi = df[df["kepoi_name"].astype(str).str.upper() == koi_id.upper()]
    if df_koi.empty:
        print("No exact match found for that KOI id.")
        return
    if len(df_koi) > 1:
        print(f"Warning: found {len(df_koi)} rows for this KOI; using the first.")

    row = df_koi.iloc[0]
    kepid = row.get("kepid", "UNKNOWN")

    print("\n" + format_property_block(row) + "\n")
    print("\n" + "=" * 78 + "\n")


def handle_list_stars_highest_counts(df: pd.DataFrame):
    disp = df["koi_disposition"].apply(disposition_upper)
    total_koi = df.groupby("kepid")["kepoi_name"].nunique()
    candidate_counts = (
        df[disp == "CANDIDATE"].groupby("kepid")["kepoi_name"].nunique()
    )
    candidate_counts = candidate_counts.reindex(total_koi.index).fillna(0).astype(int)

    stats = pd.DataFrame({"total_koi": total_koi, "candidate_koi": candidate_counts})

    print(
        "\nList stars with highest counts.\n"
        "Order by:\n"
        "  1. Candidate KOI count (descending)\n"
        "  2. Total KOI count (descending)\n"
    )
    choice = input("Enter 1 or 2 (default=1): ").strip()
    if choice == "2":
        stats_sorted = stats.sort_values(
            ["total_koi", "candidate_koi"], ascending=[False, False]
        )
        order_label = "total KOI count"
    else:
        stats_sorted = stats.sort_values(
            ["candidate_koi", "total_koi"], ascending=[False, False]
        )
        order_label = "candidate KOI count"

    print(f"\nTop 30 stars by {order_label}:\n")
    print(f"{'KEPID':>10} | {'Total KOIs':>10} | {'Candidate KOIs':>15}")
    print("-" * 42)
    for kepid, row in stats_sorted.head(30).iterrows():
        print(
            f"{int(kepid):>10} | {int(row['total_koi']):>10} | "
            f"{int(row['candidate_koi']):>15}"
        )

    print("\nReturning to main menu.\n" + "=" * 78 + "\n")


def handle_search_by_planet_type(df: pd.DataFrame):
    disp = df["koi_disposition"].apply(disposition_upper)
    proba = df["model_is_real_proba"]

    is_confirmed = disp == "CONFIRMED"
    is_not_fp = disp != "FALSE POSITIVE"
    is_real_model = proba.notna() & (proba >= 0.5)
    mask_real = is_not_fp & (is_confirmed | is_real_model)

    df_real = df[mask_real].copy()
    if df_real.empty:
        print("\nNo KOIs classified as real planets by the model/labels.\n")
        return

    df_real["combined_type_label"] = df_real["combined_type_label"].fillna("Unknown world")
    type_counts = df_real["combined_type_label"].value_counts()

    print("\nPlanet types (predicted REAL planets only):\n")
    type_list = []
    for i, (label, count) in enumerate(type_counts.items(), start=1):
        print(f"{i}. {label} : {count}")
        type_list.append(label)

    choice = input("\nEnter planet type NUMBER from the list above: ").strip()
    if not choice.isdigit():
        print("Invalid index.")
        return

    idx = int(choice)
    if idx < 1 or idx > len(type_list):
        print("Invalid index.")
        return

    chosen_type = type_list[idx - 1]
    df_type = df_real[df_real["combined_type_label"] == chosen_type].copy()
    df_type = df_type.sort_values("model_is_real_proba", ascending=False)

    print(f"\nUp to 30 KOIs classified as '{chosen_type}' and predicted REAL:\n")
    print(f"{'KEPID':>10} | {'KOI':>10} | {'NASA decision':>15} | {'P(real)':>8}")
    print("-" * 55)
    for _, row in df_type.head(30).iterrows():
        kepid = row.get("kepid", "UNKNOWN")
        koi = row.get("kepoi_name", "UNKNOWN")
        disp_str = disposition_upper(row.get("koi_disposition", ""))
        if disp_str == "CONFIRMED":
            nasa_decision = "REAL PLANET"
        elif disp_str == "FALSE POSITIVE":
            nasa_decision = "FALSE POSITIVE"
        elif disp_str == "CANDIDATE":
            nasa_decision = "CANDIDATE"
        else:
            nasa_decision = "UNKNOWN"
        p = row.get("model_is_real_proba", np.nan)
        prob_text = f"{p*100:.1f}%" if not pd.isna(p) else "N/A"
        if pd.notna(kepid) and str(kepid).isdigit():
            kepid_text = f"{int(kepid):>10}"
        else:
            kepid_text = f"{str(kepid):>10}"
        print(f"{kepid_text} | {str(koi):>10} | {nasa_decision:>15} | {prob_text:>8}")

    print("\nReturning to main menu.\n" + "=" * 78 + "\n")


def handle_top_candidates(df: pd.DataFrame, top_n: int = 30):
    disp = df["koi_disposition"].apply(disposition_upper)
    cand = df[disp == "CANDIDATE"].copy()
    cand = cand[cand["model_is_real_proba"].notna()]
    if cand.empty:
        print("\nNo candidates with model probabilities available.\n")
        return

    cand = cand.sort_values("model_is_real_proba", ascending=False)

    print(f"\nTop {min(top_n, len(cand))} candidate KOIs by ML confidence:\n")
    for _, row in cand.head(top_n).iterrows():
        name = row.get("kepoi_name", "UNKNOWN")
        kepid = row.get("kepid", "UNKNOWN")
        p = row.get("model_is_real_proba", np.nan)
        prob_text = f"{p*100:.1f}%" if not pd.isna(p) else "N/A"
        label = row.get("combined_type_label", "EXOPLANET")
        print(f"{name} around star {kepid}: P(real) = {prob_text}, type ~ {label}")

    print("\nReturning to main menu.\n" + "=" * 78 + "\n")


def handle_summary(df: pd.DataFrame):
    print("\nDataset summary:\n")
    print(f"Total KOI rows: {len(df)}")

    disp_counts = df["koi_disposition"].apply(disposition_upper).value_counts()
    print("\nKOI dispositions:")
    for k, v in disp_counts.items():
        print(f"  {k or 'UNKNOWN'}: {v}")

    size_counts = df["size_class"].astype(str).value_counts()
    print("\nPlanet size classes:")
    for k, v in size_counts.items():
        print(f"  {k}: {v}")

    temp_counts = df["temp_regime"].astype(str).value_counts()
    print("\nTemperature regimes (F/W/G/R):")
    for k, v in temp_counts.items():
        print(f"  {k}: {v}")

    print("\nReturning to main menu.\n" + "=" * 78 + "\n")


def main():
    df = load_data()
    bundle = load_model_bundle()
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_cols = bundle.get("features", [])

    df_full = predict_for_rows(model, threshold, feature_cols, df)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        print(
            "Enter the number corresponding to your task:\n"
            "1. Confirm KOIs by star\n"
            "2. Confirm KOIs by KOI\n"
            "3. List stars with highest KOI and candidate counts\n"
            "4. Search by planet type\n"
            "5. Show top candidate KOIs by ML confidence\n"
            "6. Summary statistics\n"
            "q. Quit"
        )
        choice = input("> ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("Exiting.")
            break
        if choice == "1":
            handle_confirm_by_star(df_full)
        elif choice == "2":
            handle_confirm_by_koi(df_full)
        elif choice == "3":
            handle_list_stars_highest_counts(df_full)
        elif choice == "4":
            handle_search_by_planet_type(df_full)
        elif choice == "5":
            handle_top_candidates(df_full, top_n=30)
        elif choice == "6":
            handle_summary(df_full)
        else:
            print("Invalid choice. Please enter 1–6 or 'q' to quit.\n")

if __name__ == "__main__":
    main()
