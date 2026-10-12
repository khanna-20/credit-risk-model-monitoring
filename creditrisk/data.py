"""Loading the UCI Polish Companies Bankruptcy dataset (5th-year file, 1-year horizon).

Source: Tomczak, S. (2016). Polish Companies Bankruptcy [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C5F600 (CC BY 4.0)
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from .config import resolve

FEATURES = [f"X{i}" for i in range(1, 65)]
TARGET = "default_flag"
ROW_ID = "row_id"

# Short descriptions paraphrased from the UCI attribute list.
# TA = total assets, TL = total liabilities, ST = short-term, LT = long-term.
FEATURE_DESCRIPTIONS = {
    "X1": "net profit / TA", "X2": "TL / TA", "X3": "working capital / TA",
    "X4": "current assets / ST liabilities",
    "X5": "(cash + ST securities + receivables - ST liabilities) / (opex - depreciation) x 365",
    "X6": "retained earnings / TA", "X7": "EBIT / TA", "X8": "book value of equity / TL",
    "X9": "sales / TA", "X10": "equity / TA",
    "X11": "(gross profit + extraordinary items + financial expenses) / TA",
    "X12": "gross profit / ST liabilities", "X13": "(gross profit + depreciation) / sales",
    "X14": "(gross profit + interest) / TA", "X15": "(TL x 365) / (gross profit + depreciation)",
    "X16": "(gross profit + depreciation) / TL", "X17": "TA / TL", "X18": "gross profit / TA",
    "X19": "gross profit / sales", "X20": "(inventory x 365) / sales",
    "X21": "sales (n) / sales (n-1)", "X22": "profit on operating activities / TA",
    "X23": "net profit / sales", "X24": "gross profit (in 3 years) / TA",
    "X25": "(equity - share capital) / TA", "X26": "(net profit + depreciation) / TL",
    "X27": "profit on operating activities / financial expenses",
    "X28": "working capital / fixed assets", "X29": "log(TA)", "X30": "(TL - cash) / sales",
    "X31": "(gross profit + interest) / sales",
    "X32": "(current liabilities x 365) / cost of products sold",
    "X33": "operating expenses / ST liabilities", "X34": "operating expenses / TL",
    "X35": "profit on sales / TA", "X36": "total sales / TA",
    "X37": "(current assets - inventories) / LT liabilities", "X38": "constant capital / TA",
    "X39": "profit on sales / sales",
    "X40": "(current assets - inventory - receivables) / ST liabilities",
    "X41": "TL / ((profit on operating activities + depreciation) x 12/365)",
    "X42": "profit on operating activities / sales",
    "X43": "rotation receivables + inventory turnover in days",
    "X44": "(receivables x 365) / sales", "X45": "net profit / inventory",
    "X46": "(current assets - inventory) / ST liabilities",
    "X47": "(inventory x 365) / cost of products sold", "X48": "EBITDA / TA",
    "X49": "EBITDA / sales", "X50": "current assets / TL", "X51": "ST liabilities / TA",
    "X52": "(ST liabilities x 365) / cost of products sold", "X53": "equity / fixed assets",
    "X54": "constant capital / fixed assets", "X55": "working capital",
    "X56": "(sales - cost of products sold) / sales",
    "X57": "(current assets - inventory - ST liabilities) / (sales - gross profit - depreciation)",
    "X58": "total costs / total sales", "X59": "LT liabilities / equity",
    "X60": "sales / inventory", "X61": "sales / receivables",
    "X62": "(ST liabilities x 365) / sales", "X63": "sales / ST liabilities",
    "X64": "sales / fixed assets",
}


def read_arff(path: str | Path) -> pd.DataFrame:
    """Read a UCI ARFF file and return a tidy frame with X1..X64, default_flag and row_id."""
    raw, _meta = arff.loadarff(str(path))
    df = pd.DataFrame(raw)
    rename = {f"Attr{i}": f"X{i}" for i in range(1, 65)}
    df = df.rename(columns=rename)
    if "class" not in df.columns:
        raise ValueError(f"{path}: expected a 'class' column in the ARFF file")
    cls = df.pop("class")
    if cls.dtype == object:
        cls = cls.apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
    df[TARGET] = pd.to_numeric(cls).astype(int)
    df.insert(0, ROW_ID, np.arange(len(df)))  # position in the original file, for traceability
    return df


def download_from_uci(url: str, member_name: str, dest: Path, timeout: int = 60) -> Path:
    """Download the official UCI zip and extract the requested horizon file."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed official URL
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        matches = [n for n in zf.namelist() if n.endswith(member_name)]
        if not matches:
            raise FileNotFoundError(f"{member_name} not found in UCI archive; members: {zf.namelist()}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(matches[0]))
    return dest


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Load the configured horizon file, downloading it from UCI if it is not present locally."""
    path = resolve(cfg, cfg["data"]["raw_path"])
    if not path.exists():
        try:
            print(f"[data] {path.name} not found locally - downloading from UCI ...")
            download_from_uci(cfg["data"]["uci_zip_url"], cfg["data"]["horizon_file"], path)
        except Exception as exc:  # pragma: no cover - network dependent
            raise FileNotFoundError(
                f"Could not obtain {cfg['data']['horizon_file']}. Download the zip manually from "
                "https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data, "
                f"extract {cfg['data']['horizon_file']} and place it at {path}. ({exc})"
            ) from exc
    return read_arff(path)
