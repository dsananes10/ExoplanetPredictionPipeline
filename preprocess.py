from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

KOI_PATH = RAW_DIR / "koi_data.csv"
STELLAR_PATH = RAW_DIR / "stellar_data.csv"
TRAIN_PATH = PROC_DIR / "training_data.csv"


def disposition_to_is_real(disposition):
    if not isinstance(disposition, str):
        return np.nan
    d = disposition.strip().upper()
    if d == "CONFIRMED":
        return 1.0
    if d == "FALSE POSITIVE":
        return 0.0
    return np.nan


def radius_to_size_class(prad):
    if pd.isna(prad):
        return None
    r = float(prad)
    if r <= 1.5:
        return "terrestrial"
    if r <= 2.0:
        return "super_earth"
    if r <= 4.0:
        return "gas_dwarf"
    if r <= 6.0:
        return "neptunian"
    return "gas_giant"


def teq_to_regime(teq):
    if pd.isna(teq):
        return "U"
    t = float(teq)
    if t < 250.0:
        return "F"
    if t <= 450.0:
        return "W"
    if t <= 1000.0:
        return "G"
    return "R"


def combined_type_from(size_class, temp_regime):
    if not size_class or temp_regime in (None, "U"):
        return "Unknown world", "World with unknown or poorly constrained properties."

    s = size_class
    T = temp_regime

    if s == "terrestrial":
        if T == "F":
            return "COLD TERRESTRIAL", "A rocky planet in a cold region where water would be mostly ice."
        if T == "W":
            return "EARTH-LIKE WORLD", "A rocky planet in a temperature range where liquid water could exist."
        if T == "G":
            return "WARM TERRESTRIAL", "A rocky planet in a hot region where water would mostly be vapor."
        if T == "R":
            return "EXTREMELY HOT TERRESTRIAL", "A rocky planet orbiting so close to its star that conditions are extremely hot and harsh."

    if s == "super_earth":
        if T == "F":
            return "COLD SUPER-EARTH", "A super-Earth in a cold region where water would be mostly ice."
        if T == "W":
            return "TEMPERATE SUPER-EARTH", "A super-Earth in a range where liquid water might exist."
        if T == "G":
            return "WARM SUPER-EARTH", "A super-Earth in a hot region where water would mostly be vapor."
        if T == "R":
            return "EXTREMELY HOT SUPER-EARTH", "A super-Earth orbiting very close to its star with extremely hot conditions."

    if s == "gas_dwarf":
        if T == "F":
            return "COLD GAS DWARF", "A small gas-rich planet in a cold region of its system."
        if T == "W":
            return "WARM GAS DWARF", "A small gas-rich planet in a warm region where water would mostly be vapor."
        if T == "G":
            return "HOT GAS DWARF", "A hot, compact gas-rich planet orbiting close to its star."
        if T == "R":
            return "EXTREMELY HOT GAS DWARF", "A gas-dwarf experiencing extreme irradiation near its star."

    if s == "neptunian":
        if T == "F":
            return "COLD SUB-NEPTUNE", "A Neptune-like world in a cold region of its system."
        if T == "W":
            return "WARM SUB-NEPTUNE", "A Neptune-like world in a warm region where water would mostly be vapor."
        if T == "G":
            return "HOT SUB-NEPTUNE", "A hot Neptune-class planet orbiting close to its star."
        if T == "R":
            return "EXTREMELY HOT SUB-NEPTUNE", "A strongly irradiated sub-Neptune near its star."

    if s == "gas_giant":
        if T == "F":
            return "COLD GAS GIANT", "A gas giant in a cold outer region, similar to Jupiter or Saturn."
        if T == "W":
            return "WARM GAS GIANT", "A gas giant in a warm region where water would mostly be vapor."
        if T == "G":
            return "HOT GAS GIANT", "A hot gas giant orbiting relatively close to its star."
        if T == "R":
            return "HOT JUPITER", "A very hot gas giant orbiting extremely close to its star."

    return "Unknown world", "World with unknown or poorly constrained properties."


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def log10_safe(col):
        return np.where(df[col] > 0, np.log10(df[col]), np.nan)

    df["log_koi_prad"] = log10_safe("koi_prad")
    df["log_koi_period"] = log10_safe("koi_period")
    df["log_koi_teq"] = log10_safe("koi_teq")
    df["dur_over_period"] = df["koi_duration"] / df["koi_period"]
    df["depth_over_radius2"] = df["koi_depth"] / (df["koi_prad"] ** 2)
    df["duration_over_radius"] = df["koi_duration"] / df["radius"]
    return df


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    koi = pd.read_csv(KOI_PATH, comment="#")
    stellar = pd.read_csv(STELLAR_PATH, comment="#")

    stellar_cols = ["kepid", "teff", "radius", "mass", "dens"]
    stellar_subset = stellar[[c for c in stellar_cols if c in stellar.columns]]

    df = koi.merge(stellar_subset, on="kepid", how="left")

    df["is_real_planet"] = df["koi_disposition"].apply(disposition_to_is_real)
    df["size_class"] = df["koi_prad"].apply(radius_to_size_class)
    df["temp_regime"] = df["koi_teq"].apply(teq_to_regime)

    labels = df.apply(
        lambda r: combined_type_from(r["size_class"], r["temp_regime"]),
        axis=1,
        result_type="expand",
    )
    labels.columns = ["combined_type_label", "combined_type_desc"]
    df = pd.concat([df, labels], axis=1)

    df = add_engineered_features(df)

    id_cols = ["kepid", "kepoi_name", "koi_disposition"]
    phys_cols = [
        "koi_period",
        "koi_duration",
        "koi_depth",
        "koi_prad",
        "koi_model_snr",
        "koi_teq",
        "teff",
        "radius",
        "mass",
        "dens",
    ]
    eng_cols = [
        "log_koi_prad",
        "log_koi_period",
        "log_koi_teq",
        "dur_over_period",
        "depth_over_radius2",
        "duration_over_radius",
    ]
    label_cols = [
        "is_real_planet",
        "size_class",
        "temp_regime",
        "combined_type_label",
        "combined_type_desc",
    ]

    cols = [c for c in id_cols + phys_cols + eng_cols + label_cols if c in df.columns]
    df[cols].to_csv(TRAIN_PATH, index=False)

    print("Data has been processed to training_data.csv")


if __name__ == "__main__":
    main()
