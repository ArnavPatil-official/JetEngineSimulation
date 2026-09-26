from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None


LOGGER = logging.getLogger("fetch_and_build_cfd_data")

REPO_ROOT = Path(__file__).resolve().parent
RAW_ROOT = REPO_ROOT / "data" / "raw" / "cfd_datasets"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
MASTER_DATASET_PATH = PROCESSED_ROOT / "master_shock_dataset.pt"

DEFAULT_A5 = 0.20
DEFAULT_A6 = 0.32
DEFAULT_PIN = 300000.0
DEFAULT_TIN = 900.0
GAS_CONSTANT = 287.0

KAGGLE_SEARCH_QUERY = "Convergence Divergence Nozzle Data CFD"
KAGGLE_SLUG_ENV = "KAGGLE_DATASET_SLUG"
GITHUB_OWNER = "BacchusX1"
GITHUB_REPO = "nozzle_flow_cfd"
NASA_CASE_URL = "https://www.grc.nasa.gov/www/wind/valid/transdif/transdif01/transdif01.html"

STANDARD_COLUMNS = ["x", "y", "u", "v", "P", "T", "rho"]
TABULAR_SUFFIXES = {".csv", ".txt", ".dat", ".tsv"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".tar.gz", ".tar.bz2", ".tar.xz"}
H5_SUFFIXES = {".h5", ".hdf5", ".hdf"}

REQUEST_TIMEOUT = 60
USER_AGENT = "JetEngineSimulationCFDFetcher/1.0"


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_repo_env(env_path: Path | None = None) -> None:
    target = env_path or (REPO_ROOT / ".env")
    if not target.exists():
        return

    for raw_line in target.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ensure_directories() -> dict[str, Path]:
    raw_dirs = {
        "root": RAW_ROOT,
        "kaggle": RAW_ROOT / "kaggle",
        "github": RAW_ROOT / "github",
        "nasa": RAW_ROOT / "nasa",
    }
    for path in [*raw_dirs.values(), PROCESSED_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    return raw_dirs


def requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        session.headers["Authorization"] = f"Bearer {github_token}"
    return session


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or "downloaded_file"


def is_archive(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def is_tabular(path: Path) -> bool:
    return path.suffix.lower() in TABULAR_SUFFIXES


def is_h5(path: Path) -> bool:
    return path.suffix.lower() in H5_SUFFIXES


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    overwrite: bool = False,
    timeout: int = REQUEST_TIMEOUT,
) -> Path:
    if destination.exists() and not overwrite:
        LOGGER.debug("Using cached download: %s", destination)
        return destination

    LOGGER.info("Downloading %s", url)
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return destination


def extract_archive(archive_path: Path, destination: Path, overwrite: bool = False) -> list[Path]:
    extracted: list[Path] = []
    if not archive_path.exists():
        return extracted

    LOGGER.info("Extracting %s", archive_path.name)
    destination.mkdir(parents=True, exist_ok=True)

    lower_name = archive_path.name.lower()
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                target = destination / member
                if target.exists() and not overwrite:
                    continue
                zf.extract(member, destination)
                extracted.append(target)
        return extracted

    tar_mode = None
    if tarfile.is_tarfile(archive_path):
        tar_mode = "r:*"
    elif lower_name.endswith(".gz"):
        tar_mode = "r:gz"
    elif lower_name.endswith(".bz2"):
        tar_mode = "r:bz2"
    elif lower_name.endswith(".xz"):
        tar_mode = "r:xz"

    if tar_mode:
        with tarfile.open(archive_path, tar_mode) as tf:
            members = tf.getmembers()
            for member in members:
                target = destination / member.name
                if target.exists() and not overwrite:
                    continue
                tf.extract(member, destination)
                extracted.append(target)
        return extracted

    LOGGER.warning("Unsupported archive format: %s", archive_path)
    return extracted


def iter_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def sanitize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def standard_column_mapping(columns: list[str]) -> dict[str, str]:
    aliases: dict[str, set[str]] = {
        "x": {
            "x",
            "points0",
            "point0",
            "coordx",
            "coordinatex",
            "xcoordinate",
            "positionx",
            "xcoord",
            "xpos",
            "gridx",
        },
        "y": {
            "y",
            "points1",
            "point1",
            "coordy",
            "coordinatey",
            "ycoordinate",
            "positiony",
            "ycoord",
            "ypos",
            "gridy",
        },
        "u": {
            "u",
            "velocity0",
            "vel0",
            "ux",
            "uvelocity",
            "xvelocity",
            "velocityx",
            "axialvelocity",
        },
        "v": {
            "v",
            "velocity1",
            "vel1",
            "uy",
            "vvelocity",
            "yvelocity",
            "velocityy",
            "radialvelocity",
        },
        "P": {
            "p",
            "pressure",
            "staticpressure",
            "pressurepa",
            "press",
        },
        "T": {
            "t",
            "temperature",
            "statictemperature",
            "temperaturek",
            "temp",
        },
        "rho": {
            "rho",
            "density",
            "rhou",
            "densitykgm3",
        },
    }

    mapping: dict[str, str] = {}
    for column in columns:
        key = sanitize_column_name(column)
        for standard_name, names in aliases.items():
            if key in names:
                mapping[column] = standard_name
                break
    return mapping


def load_text_table(path: Path) -> pd.DataFrame:
    separators = [",", r"\s+", "\t"]
    last_error: Exception | None = None
    for separator in separators:
        try:
            df = pd.read_csv(
                path,
                sep=separator,
                engine="python",
                comment="#",
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip",
            )
            if df.shape[1] >= 2:
                return df
        except Exception as exc:  # pragma: no cover - parser variability
            last_error = exc
    raise ValueError(f"Unable to parse tabular file {path}: {last_error}")


def pressure_ratio_to_static_pressure(pressure_ratio: float) -> float:
    gamma = 1.4
    freestream_mach = 0.46
    freestream_static_pressure_psi = 16.937
    psi_to_pa = 6894.757293168
    total_pressure = freestream_static_pressure_psi * psi_to_pa * (1.0 + 0.5 * (gamma - 1.0) * freestream_mach**2) ** (
        gamma / (gamma - 1.0)
    )
    return pressure_ratio * total_pressure


def parse_nasa_experimental_report(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    rows: list[dict[str, float]] = []
    current_velocity_x: float | None = None
    in_pressure_block = False
    in_velocity_block = False

    pressure_pattern = re.compile(
        r"^\s*([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s*$"
    )
    velocity_header_pattern = re.compile(r"X/H\s*=\s*([+-]?\d*\.?\d+)")
    velocity_pattern = re.compile(r"^\s*([+-]?\d*\.?\d+)\s+([+-]?\d*\.?\d+)\s*$")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if "SURFACE STATIC PRESSURE" in stripped:
            in_pressure_block = True
            in_velocity_block = False
            current_velocity_x = None
            continue

        if "VELOCITY PROFILES" in stripped:
            in_pressure_block = False
            in_velocity_block = True
            current_velocity_x = None
            continue

        velocity_header = velocity_header_pattern.search(stripped)
        if velocity_header:
            current_velocity_x = float(velocity_header.group(1))
            continue

        if in_pressure_block:
            match = pressure_pattern.match(line)
            if match:
                x_top, p_top_ratio, x_bottom, p_bottom_ratio = map(float, match.groups())
                rows.append(
                    {
                        "x": x_top,
                        "y": 1.0,
                        "u": 0.0,
                        "v": 0.0,
                        "P": pressure_ratio_to_static_pressure(p_top_ratio),
                        "T": DEFAULT_TIN,
                    }
                )
                rows.append(
                    {
                        "x": x_bottom,
                        "y": -1.0,
                        "u": 0.0,
                        "v": 0.0,
                        "P": pressure_ratio_to_static_pressure(p_bottom_ratio),
                        "T": DEFAULT_TIN,
                    }
                )
            continue

        if in_velocity_block and current_velocity_x is not None:
            match = velocity_pattern.match(line)
            if match and "M/S" not in stripped and "X-VELOCITY" not in stripped and "Y/H" not in stripped:
                y_over_h, u_velocity = map(float, match.groups())
                for signed_y in (y_over_h, -y_over_h):
                    rows.append(
                        {
                            "x": current_velocity_x,
                            "y": signed_y,
                            "u": u_velocity,
                            "v": 0.0,
                            "P": DEFAULT_PIN,
                            "T": DEFAULT_TIN,
                        }
                    )

    if not rows:
        raise ValueError(f"No experimental Sajben rows found in {path}")

    dataframe = pd.DataFrame(rows)
    dataframe["rho"] = dataframe["P"] / (GAS_CONSTANT * dataframe["T"])
    return dataframe


def dataframe_from_h5(path: Path) -> pd.DataFrame:
    if h5py is None:
        raise RuntimeError("h5py is required to parse HDF5 files")

    arrays: dict[str, np.ndarray] = {}

    def visitor(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        try:
            data = np.asarray(obj[()])
        except Exception:
            return

        dataset_name = sanitize_column_name(Path(name).name)
        if data.ndim == 1:
            arrays[dataset_name] = data
            return

        if data.ndim == 2 and data.shape[1] in (2, 3):
            if "point" in dataset_name:
                arrays["x"] = data[:, 0]
                arrays["y"] = data[:, 1]
            elif "velocity" in dataset_name or dataset_name.startswith("vel"):
                arrays["u"] = data[:, 0]
                arrays["v"] = data[:, 1]

    with h5py.File(path, "r") as handle:
        handle.visititems(visitor)

    dataframe = pd.DataFrame(arrays)
    if dataframe.empty:
        raise ValueError(f"No tabular CFD arrays found in {path}")
    return dataframe


def standardize_dataframe(df: pd.DataFrame, source: str, file_path: Path) -> pd.DataFrame | None:
    if df.empty:
        LOGGER.warning("Skipping empty table from %s", file_path)
        return None

    working = df.copy()
    working = working.rename(columns=standard_column_mapping(list(working.columns)))

    kaggle_design_columns = {
        "P1 _Convergence Length",
        "P2 Divergence Length",
        "P3_InletDiameter",
        "P4_Throttle_Diameter",
        "P5_ExitDiameter",
        "P6_Velocity",
        "P7_Pressure_Inlet",
    }
    if source == "kaggle" and kaggle_design_columns.issubset(set(df.columns)):
        LOGGER.warning(
            "Skipping %s from %s: dataset is a nozzle design summary table, not a CFD field export with x/y field samples",
            file_path.name,
            source,
        )
        return None

    for required in ("x", "y", "u", "v", "P"):
        if required not in working.columns:
            LOGGER.warning("Skipping %s from %s: missing %s", file_path.name, source, required)
            return None

    if "T" not in working.columns:
        LOGGER.warning("Using default temperature %.1f K for %s", DEFAULT_TIN, file_path.name)
        working["T"] = DEFAULT_TIN

    for column in ["x", "y", "u", "v", "P", "T"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working = working.dropna(subset=["x", "y", "u", "v", "P", "T"])
    if working.empty:
        LOGGER.warning("Skipping %s after numeric coercion", file_path.name)
        return None

    if "rho" in working.columns:
        working["rho"] = pd.to_numeric(working["rho"], errors="coerce")
    else:
        working["rho"] = np.nan

    missing_rho = working["rho"].isna()
    if missing_rho.any():
        working.loc[missing_rho, "rho"] = working.loc[missing_rho, "P"] / (GAS_CONSTANT * working.loc[missing_rho, "T"])

    standardized = working[STANDARD_COLUMNS].copy()
    standardized["source"] = source
    standardized["file_path"] = str(file_path)
    return standardized


def parse_candidate_file(file_path: Path, source: str) -> pd.DataFrame | None:
    suffix = file_path.suffix.lower()
    try:
        lower_name = file_path.name.lower()
        if source == "nasa" and lower_name == "data.mach46.txt":
            df = parse_nasa_experimental_report(file_path)
            return standardize_dataframe(df, source=source, file_path=file_path)
        if is_h5(file_path):
            df = dataframe_from_h5(file_path)
        elif is_tabular(file_path):
            df = load_text_table(file_path)
        else:
            return None
        return standardize_dataframe(df, source=source, file_path=file_path)
    except Exception as exc:
        LOGGER.warning("Failed to parse %s: %s", file_path, exc)
        return None


def choose_kaggle_dataset(kaggle_api: Any, explicit_slug: str | None) -> str:
    if explicit_slug:
        LOGGER.info("Using Kaggle slug from %s=%s", KAGGLE_SLUG_ENV, explicit_slug)
        return explicit_slug

    datasets = list(kaggle_api.dataset_list(search=KAGGLE_SEARCH_QUERY))
    if not datasets:
        raise RuntimeError(f"No Kaggle datasets matched query: {KAGGLE_SEARCH_QUERY}")

    exact_title = KAGGLE_SEARCH_QUERY.lower()
    for dataset in datasets:
        title = getattr(dataset, "title", "").strip().lower()
        ref = getattr(dataset, "ref", None) or getattr(dataset, "id", None)
        if title == exact_title and ref:
            return ref

    for dataset in datasets:
        ref = getattr(dataset, "ref", None) or getattr(dataset, "id", None)
        title = getattr(dataset, "title", "")
        if ref and "nozzle" in title.lower():
            LOGGER.info("Selected Kaggle dataset %s (%s)", ref, title)
            return ref

    fallback = getattr(datasets[0], "ref", None) or getattr(datasets[0], "id", None)
    if not fallback:
        raise RuntimeError("Kaggle dataset search did not return a usable slug")
    LOGGER.info("Falling back to first Kaggle search result: %s", fallback)
    return fallback


def download_kaggle_data_with_kagglehub(raw_dir: Path, slug: str | None, overwrite: bool = False) -> list[Path]:
    token = os.getenv("KAGGLE_API_TOKEN")
    if not token:
        return []

    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        LOGGER.warning("KAGGLE_API_TOKEN is set, but kagglehub is not installed (%s)", exc)
        return []

    if not slug:
        LOGGER.warning(
            "KAGGLE_API_TOKEN is set but %s is missing. Set it to the dataset slug, e.g. owner/dataset-name.",
            KAGGLE_SLUG_ENV,
        )
        return []

    os.environ.setdefault("KAGGLE_API_TOKEN", token)
    try:
        LOGGER.info("Downloading Kaggle dataset via kagglehub for slug %s", slug)
        dataset_path = Path(kagglehub.dataset_download(slug, force_download=overwrite))
    except Exception as exc:
        LOGGER.warning("kagglehub download failed for %s: %s", slug, exc)
        return []

    copied: list[Path] = []
    for source in iter_files(dataset_path):
        relative = source.relative_to(dataset_path)
        destination = raw_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not destination.exists():
            shutil.copy2(source, destination)
        copied.append(destination)

    LOGGER.info("Kaggle candidate files discovered via kagglehub: %d", len(copied))
    return [path for path in copied if is_tabular(path) or is_h5(path)]


def download_kaggle_data(raw_dir: Path, overwrite: bool = False) -> list[Path]:
    explicit_slug = os.getenv(KAGGLE_SLUG_ENV)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        LOGGER.warning("kaggle package not installed (%s); attempting kagglehub fallback", exc)
        return download_kaggle_data_with_kagglehub(raw_dir, explicit_slug, overwrite=overwrite)

    try:
        api = KaggleApi()
        api.authenticate()
        slug = choose_kaggle_dataset(api, explicit_slug)
        archive_stem = slug.replace("/", "__")
        archive_path = raw_dir / f"{archive_stem}.zip"
        if overwrite and archive_path.exists():
            archive_path.unlink()
        api.dataset_download_files(slug, path=str(raw_dir), quiet=False, unzip=False)

        downloaded_candidates = sorted(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not downloaded_candidates:
            raise RuntimeError(f"Kaggle download completed but no zip archive was found in {raw_dir}")

        latest_zip = downloaded_candidates[0]
        normalized_path = raw_dir / f"{archive_stem}.zip"
        if latest_zip != normalized_path:
            shutil.move(str(latest_zip), normalized_path)
        extract_archive(normalized_path, raw_dir, overwrite=overwrite)
    except Exception as exc:
        LOGGER.warning("Kaggle API client failed: %s", exc)
        fallback = download_kaggle_data_with_kagglehub(raw_dir, explicit_slug, overwrite=overwrite)
        if fallback:
            return fallback
        LOGGER.warning(
            "Skipping Kaggle source. The kaggle client expects KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json; "
            "KAGGLE_API_TOKEN requires kagglehub plus %s.",
            KAGGLE_SLUG_ENV,
        )
        return []

    candidates = [
        path
        for path in iter_files(raw_dir)
        if is_tabular(path) or is_h5(path)
    ]
    LOGGER.info("Kaggle candidate files discovered: %d", len(candidates))
    return candidates


def github_repo_metadata(session: requests.Session) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def github_tree(session: requests.Session, default_branch: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/git/trees/{default_branch}?recursive=1"
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    return payload.get("tree", [])


def select_github_paths(tree_items: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for item in tree_items:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        lower = path.lower()
        if not (lower.endswith(".csv") or lower.endswith(".h5") or lower.endswith(".hdf5")):
            continue
        if lower.startswith("data/") or lower.startswith("results/") or "/data/" in lower or "/results/" in lower:
            selected.append(path)

    if selected:
        return selected

    for item in tree_items:
        path = item.get("path", "")
        lower = path.lower()
        if item.get("type") == "blob" and (lower.endswith(".csv") or lower.endswith(".h5") or lower.endswith(".hdf5")):
            selected.append(path)
    return selected


def download_github_data(raw_dir: Path, session: requests.Session, overwrite: bool = False) -> list[Path]:
    try:
        metadata = github_repo_metadata(session)
        default_branch = metadata["default_branch"]
        tree_items = github_tree(session, default_branch)
        selected_paths = select_github_paths(tree_items)
        if not selected_paths:
            raise RuntimeError("No CSV/HDF5 files were found in the GitHub repository tree")

        downloaded: list[Path] = []
        for repo_path in selected_paths:
            url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{default_branch}/{repo_path}"
            destination = raw_dir / repo_path
            download_file(session, url, destination, overwrite=overwrite)
            downloaded.append(destination)
        LOGGER.info("GitHub candidate files downloaded: %d", len(downloaded))
        return downloaded
    except Exception as exc:
        LOGGER.warning("GitHub API download failed, attempting repo archive fallback: %s", exc)

    try:
        archive_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/archive/refs/heads/main.zip"
        archive_path = raw_dir / f"{GITHUB_REPO}.zip"
        download_file(session, archive_url, archive_path, overwrite=overwrite)
        extract_archive(archive_path, raw_dir, overwrite=overwrite)
    except Exception as exc:
        LOGGER.warning("Skipping GitHub source: %s", exc)
        return []

    candidates = [
        path
        for path in iter_files(raw_dir)
        if is_tabular(path) or is_h5(path)
    ]
    LOGGER.info("GitHub fallback candidate files discovered: %d", len(candidates))
    return candidates


def parse_nasa_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    keywords = ("sajben", "transdif", "mach", "pressure", "data", "cgd", "pts")
    extensions = (".cgd", ".dat", ".txt", ".pts", ".tar", ".tar.gz", ".tgz", ".zip", ".gz")

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        absolute_url = urljoin(base_url, href)
        lower = absolute_url.lower()
        if any(keyword in lower for keyword in keywords) or any(lower.endswith(ext) for ext in extensions):
            links.append(absolute_url)

    unique_links = []
    seen: set[str] = set()
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    return unique_links


def scrape_nasa_archive(raw_dir: Path, session: requests.Session, overwrite: bool = False) -> list[Path]:
    try:
        response = session.get(NASA_CASE_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:
        LOGGER.warning("Skipping NASA source: %s", exc)
        return []

    urls = parse_nasa_links(response.text, NASA_CASE_URL)
    if not urls:
        LOGGER.warning("NASA page did not expose any downloadable Sajben case links")
        return []

    downloaded: list[Path] = []
    for url in urls:
        destination = raw_dir / filename_from_url(url)
        try:
            path = download_file(session, url, destination, overwrite=overwrite)
            downloaded.append(path)
            if is_archive(path):
                extract_archive(path, raw_dir, overwrite=overwrite)
        except Exception as exc:
            LOGGER.warning("NASA download failed for %s: %s", url, exc)

    candidates = [
        path
        for path in iter_files(raw_dir)
        if is_tabular(path) or is_h5(path)
    ]
    LOGGER.info("NASA candidate files discovered: %d", len(candidates))
    return candidates


def compile_tensors(dataframes: list[pd.DataFrame], source_counts: dict[str, int] | None = None) -> dict[str, torch.Tensor]:
    input_tensors: list[torch.Tensor] = []
    target_tensors: list[torch.Tensor] = []

    for dataframe in dataframes:
        n_rows = len(dataframe)
        inputs = np.zeros((n_rows, 6), dtype=np.float32)
        inputs[:, 0] = dataframe["x"].to_numpy(dtype=np.float32)
        inputs[:, 1] = dataframe["y"].to_numpy(dtype=np.float32)
        inputs[:, 2] = DEFAULT_A5
        inputs[:, 3] = DEFAULT_A6
        inputs[:, 4] = DEFAULT_PIN
        inputs[:, 5] = DEFAULT_TIN

        targets = np.zeros((n_rows, 9), dtype=np.float32)
        targets[:, 0] = dataframe["rho"].to_numpy(dtype=np.float32)
        targets[:, 1] = dataframe["u"].to_numpy(dtype=np.float32)
        targets[:, 2] = dataframe["v"].to_numpy(dtype=np.float32)
        targets[:, 3] = dataframe["P"].to_numpy(dtype=np.float32)
        targets[:, 4] = dataframe["T"].to_numpy(dtype=np.float32)

        input_tensors.append(torch.from_numpy(inputs))
        target_tensors.append(torch.from_numpy(targets))

    if not input_tensors:
        diagnostics = ""
        if source_counts:
            diagnostics = " Source table counts: " + ", ".join(f"{source}={count}" for source, count in sorted(source_counts.items()))
        raise RuntimeError("No standardized CFD tables were available for tensor compilation." + diagnostics)

    return {
        "inputs": torch.cat(input_tensors, dim=0),
        "targets": torch.cat(target_tensors, dim=0),
    }


def gather_standardized_frames(source_name: str, candidate_paths: list[Path]) -> list[pd.DataFrame]:
    standardized_frames: list[pd.DataFrame] = []
    for path in sorted(set(candidate_paths)):
        parsed = parse_candidate_file(path, source=source_name)
        if parsed is not None:
            standardized_frames.append(parsed)
    LOGGER.info("%s standardized tables: %d", source_name, len(standardized_frames))
    return standardized_frames


def save_master_dataset(dataset: dict[str, torch.Tensor], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, destination)
    LOGGER.info(
        "Saved master dataset to %s with inputs=%s and targets=%s",
        destination,
        tuple(dataset["inputs"].shape),
        tuple(dataset["targets"].shape),
    )


def write_summary(frames: list[pd.DataFrame], destination: Path) -> None:
    summary = {
        "num_tables": len(frames),
        "num_rows": int(sum(len(frame) for frame in frames)),
        "sources": {},
    }
    for frame in frames:
        source = str(frame["source"].iloc[0])
        summary["sources"].setdefault(source, 0)
        summary["sources"][source] += int(len(frame))
    destination.write_text(json.dumps(summary, indent=2))


def build_dataset(skip_download: bool = False, overwrite: bool = False) -> None:
    dirs = ensure_directories()
    session = requests_session()

    source_candidates: dict[str, list[Path]] = {"kaggle": [], "github": [], "nasa": []}

    if not skip_download:
        source_candidates["kaggle"] = download_kaggle_data(dirs["kaggle"], overwrite=overwrite)
        source_candidates["github"] = download_github_data(dirs["github"], session=session, overwrite=overwrite)
        source_candidates["nasa"] = scrape_nasa_archive(dirs["nasa"], session=session, overwrite=overwrite)
    else:
        for name, directory in (("kaggle", dirs["kaggle"]), ("github", dirs["github"]), ("nasa", dirs["nasa"])):
            source_candidates[name] = [path for path in iter_files(directory) if is_tabular(path) or is_h5(path)]

    standardized_frames: list[pd.DataFrame] = []
    source_table_counts: dict[str, int] = {}
    for source_name, candidates in source_candidates.items():
        frames = gather_standardized_frames(source_name, candidates)
        source_table_counts[source_name] = len(frames)
        standardized_frames.extend(frames)

    dataset = compile_tensors(standardized_frames, source_counts=source_table_counts)
    save_master_dataset(dataset, MASTER_DATASET_PATH)
    write_summary(standardized_frames, PROCESSED_ROOT / "master_shock_dataset_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch open-source CFD nozzle datasets and build a unified LE-PINN tensor dataset."
    )
    parser.add_argument("--skip-download", action="store_true", help="Only parse files already present under data/raw/cfd_datasets.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload archives and replace extracted files.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_repo_env()
    configure_logging(verbose=args.verbose)
    build_dataset(skip_download=args.skip_download, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
