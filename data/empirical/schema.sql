-- CAT-JET empirical database (Phase 8, P8.6). Long format: one row per
-- measured quantity. A missing value is an absent row, never an invented one.
-- Vocabulary, units, tiers, quality classes: data/empirical/vocabulary.yaml.
-- Created by scripts/phase8/empirical_db.py (never edit the .sqlite by hand).
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source (
    source_id    TEXT PRIMARY KEY,               -- e.g. NASA-CR-168189
    citation     TEXT NOT NULL,
    identifier   TEXT,                            -- DOI, NTRS id or URL
    type         TEXT NOT NULL CHECK (type IN ('nasa_report','journal','thesis','databank',
                                                'certification_sheet','manufacturer','other')),
    access_date  TEXT NOT NULL,                   -- ISO date
    pdf_sha256   TEXT,                            -- of the file as downloaded
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS experiment (
    experiment_id    TEXT PRIMARY KEY,
    source_id        TEXT NOT NULL REFERENCES source(source_id),
    independence_key TEXT NOT NULL,   -- same physical rig/article/test campaign => same key
    facility         TEXT,
    test_article     TEXT NOT NULL,
    component        TEXT NOT NULL CHECK (component IN ('turbine','nozzle','combustor','compressor','fan','engine')),
    geometry_json    TEXT,
    working_fluid    TEXT NOT NULL,   -- 'cold_air', 'combustion_products', or a fuel id
    status           TEXT NOT NULL CHECK (status IN ('experiment','cfd','analytic','manufacturer')),
    fidelity_tier    INTEGER NOT NULL CHECK (fidelity_tier BETWEEN 0 AND 4),
    quality_class    TEXT NOT NULL CHECK (quality_class IN ('A','B','C')),
    notes            TEXT,
    CHECK (status <> 'experiment' OR fidelity_tier = 4),
    CHECK (fidelity_tier <> 4 OR status IN ('experiment','manufacturer'))
);

CREATE TABLE IF NOT EXISTS operating_point (
    op_id         TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiment(experiment_id),
    run_label     TEXT NOT NULL,
    p_amb_Pa      REAL,
    T_amb_K       REAL,
    UNIQUE (experiment_id, run_label)
);

CREATE TABLE IF NOT EXISTS observation (
    obs_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    op_id          TEXT NOT NULL REFERENCES operating_point(op_id),
    quantity       TEXT NOT NULL,        -- controlled vocabulary (checked in code)
    value_si       REAL NOT NULL,
    unit_si        TEXT NOT NULL,
    value_original REAL NOT NULL,
    unit_original  TEXT NOT NULL,
    sigma_si       REAL CHECK (sigma_si IS NULL OR sigma_si >= 0),
    sigma_kind     TEXT CHECK (sigma_kind IS NULL OR sigma_kind IN
                               ('reported','derived','class_default','digitisation')),
    role           TEXT NOT NULL CHECK (role IN ('input','output')),
    derived_from   TEXT,                 -- JSON list of obs_id, or NULL for a direct value
    location       TEXT NOT NULL,        -- table/figure/page in the source
    UNIQUE (op_id, quantity, location)
);

CREATE TABLE IF NOT EXISTS digitisation (
    obs_id          INTEGER PRIMARY KEY REFERENCES observation(obs_id),
    figure          TEXT NOT NULL,
    tool            TEXT NOT NULL,       -- e.g. WebPlotDigitizer 4.x
    axis_calibration_json TEXT NOT NULL,
    repeats_json    TEXT NOT NULL,       -- the three repeat digitisations (original units)
    spread_si       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS split (
    experiment_id     TEXT PRIMARY KEY REFERENCES experiment(experiment_id),
    role              TEXT NOT NULL CHECK (role IN ('calibration','validation','locked_test')),
    registration_hash TEXT NOT NULL
);

-- Validation and locked-test roles are Tier 4 only (plan P8.6).
CREATE TRIGGER IF NOT EXISTS split_tier4_only
BEFORE INSERT ON split
WHEN NEW.role IN ('validation','locked_test')
 AND (SELECT fidelity_tier FROM experiment WHERE experiment_id = NEW.experiment_id) <> 4
BEGIN
    SELECT RAISE(ABORT, 'validation/locked_test experiments must be fidelity tier 4');
END;
