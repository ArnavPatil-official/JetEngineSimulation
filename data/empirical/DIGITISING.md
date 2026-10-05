# Digitising plotted-only figures (WebPlotDigitizer)

For: Arnav. Purpose: turn plotted-only data from the NACA/NASA reports in
`data/empirical/pdf/` into database rows. Each curve is digitised **three
times, independently**. The importer (`scripts/phase8/wpd_import.py`) averages
the three repeats and stores their scatter as the digitisation uncertainty.
It refuses files that don't line up, so a mistake shows up as an error. It
never turns into a wrong number in the database.

## 1. What to digitise, in priority order

Source for the nozzle figures: **NACA TN-1757** (`19930082415.pdf`), not TR-933
(`19930091998.pdf`). It's the same study (Grey & Wilsted, 15 conical nozzles,
5 in. inlet, cold air, P1/p0 = 1.0–2.8), but TN-1757 gives each Figure 4 panel
a whole page. Its plot area is about 1.7 times larger in each direction than
the TR-933 page that fits all four panels, and its symbols are bigger and
open. Both PDFs are 300 dpi scans.

Only digitise **data symbols**. Never digitise the faired lines, and never
digitise Figures 5, 7, 9 or 10. Figures 5 and 7 are cross-plots of the faired
curves, Figure 9 (Cv,e) has no vocabulary term, and Figure 10 is computed.
None of these are measurements.

The pressure ratio P1/p0 is nozzle-inlet total pressure over ambient static
pressure. That is the vocabulary quantity `NPR`. The data span P1/p0 = 1.0 to
2.8, so they cross choking (about 1.89) but stop below 4.

### 1a. Flow (discharge) coefficient, NACA TN-1757 Figure 4

Caption: *"Figure 4. Variation of conical-nozzle flow coefficient with
pressure ratio across nozzle for various cone half-angles."* On every panel:

- x axis: pressure ratio P1/p0 (unitless), linear, labelled 1.0 to 2.8.
- y axis: flow coefficient Cd (unitless), linear, labelled .60 to 1.00.
- Importer mapping: x = `NPR`, unit `"1"`; y = `Cd`, unit `"1"`.

The curves chosen are the small and moderate cone angles. Those are closest to
a real convergent core or bypass nozzle, which has a shallow convergence and a
diameter ratio of about 0.7–0.9. The 90° (orifice-like) curves and the other
large-angle curves are left out.

| # | PDF page | Panel (D2/D1) | Curve = symbol (cone half-angle α) | ≈ points | Folder / curve name |
|---|---|---|---|---|---|
| 1 | 22 | 4(c), 0.80 | ○ α = 15° | 13 | `fig4c/alpha15_d080` |
| 2 | 23 | 4(d), 0.91 | □ α = 15° | 15 | `fig4d/alpha15_d091` |
| 3 | 23 | 4(d), 0.91 | ○ α = 6° | 14 | `fig4d/alpha06_d091` |
| 4 | 21 | 4(b), 0.67 | ○ α = 16° | 13 | `fig4b/alpha16_d067` |
| 5 | 20 | 4(a), 0.50 | □ α = 13° | 11 | `fig4a/alpha13_d050` |
| 6 | 20 | 4(a), 0.50 | ○ α = 5° | 19 | `fig4a/alpha05_d050` |

The point counts are my estimates. Your own count before repeat 1 (section 3)
is the one that counts. Many points sit close together between P1/p0 = 1.0 and
1.3. Where two symbols overlap but you can see two outlines, they are two
points.

### 1b. Velocity coefficient, NACA TN-1757 Figure 8 (PDF page 31, landscape)

Caption: *"Figure 8. Variation of conical-nozzle velocity coefficient with
pressure ratio for various cone half-angles and outlet-inlet diameter
ratios."* All 15 nozzles are on one plot, each with its own symbol.

- x axis: P1/p0, linear, labelled 1.0 to 2.8.
- y axis: velocity coefficient Cv, linear, labelled .80 to 1.04.
- Importer mapping: x = `NPR`; y = `Cv`, unit `"1"`. There is one caveat. The
  report defines Cv as measured over theoretical jet velocity *at the nozzle
  outlet*, with the pressure thrust kept separate. The vocabulary text "velocity
  (thrust) coefficient" doesn't say which basis it uses. The maintainer records
  the basis in the experiment notes or clarifies the vocabulary. The files don't
  change either way.

Curves, in priority order. Use the same names as in Figure 4, in folder `fig8/`.
Look at the legend at 300 % zoom first. Several symbols differ only by a small
flag or by which way a triangle points.

| # | Symbol in the legend | Nozzle | Curve name |
|---|---|---|---|
| 7 | △ upright triangle | 16°, 0.67 | `fig8/alpha16_d067` |
| 8 | ▷ triangle pointing right | 6°, 0.91 | `fig8/alpha06_d091` |
| 9 | ○ circle | 5°, 0.50 | `fig8/alpha05_d050` |
| 10 | right-angled triangle pointing left (vertical side on the right) | 15°, 0.80 | `fig8/alpha15_d080` |
| 11 | ◁ isosceles triangle pointing left | 15°, 0.91 | `fig8/alpha15_d091` |

Expect roughly as many points as that nozzle has in Figure 4, with most of them
crowded below P1/p0 = 1.3. In your inventory, only list a symbol you can
identify with certainty. If you can't tell whether a symbol is #10 or #11,
leave it out of *both* curves and write it in the notes (section 3).

### 1c. Cold-air turbine, NASA TN D-6967 (`19720024422.pdf`) Figure 15

This is the single most useful figure. It is **PDF page 23** (report page 21),
bottom figure, and it gives raw data symbols for the two-stage turbine. The
efficiency maps (Figures 10 and 17) were built by cross-plotting faired
curves, and Figure 18 shows only faired lines with no symbols. None of them are
data, so don't digitise them.

- Caption: *"Figure 15. Variation of torque with pressure ratio and speed for
  two-stage operation."*
- x axis: equivalent inlet-total to exit-static pressure ratio (p1'/p5)eq, linear,
  labelled 2.2 to 5.4.
- y axis: equivalent torque τε/δ. Use the **inner N·m scale**, labelled 40 to
  200. Don't use the outer ft-lbf scale.
- Curves: percent of equivalent design speed (design is 15 336 rpm, Table I,
  PDF page 30). Take design speed and its two neighbours:

| # | Symbol | Speed | ≈ points | Folder / curve name |
|---|---|---|---|---|
| 12 | ○ | 100 % | 14 | `fig15/speed100` |
| 13 | ◇ | 90 % | 12 | `fig15/speed090` |
| 14 | □ | 110 % | 14 | `fig15/speed110` |

Several of these are double symbols (repeated runs). Each one counts.

**Vocabulary gap.** Neither axis has a term in vocabulary v2:

- `PR` is total-to-total, and `PR_reported` is only for a source that doesn't
  state its basis. This one is total-to-static.
- `torque` is raw shaft torque, but this axis is corrected torque τε/δ.

The speed can be stored as `N_corr` = % × 15 336 rpm. Digitise it anyway. The
files will be imported after a vocabulary amendment adds, for example,
`PR_ts` (Pt_in/ps_out) and `torque_corr` (τε/δ). The companion Figure 16 (same
page, top) has y = equivalent mass flow, which maps to `mdot_corr`. It has the
same x-axis gap, and its y range is narrow (flat, choked), so it's optional.

## 2. Where the files go

```
data/empirical/digitised/<source_id>/<figure>/<curve>_r1.csv
                                             /<curve>_r2.csv
                                             /<curve>_r3.csv
                                             /<figure>_r1.json   (one project per repeat,
                                             /<figure>_r2.json    holding every curve of that
                                             /<figure>_r3.json    figure for that repeat)
                                             /NOTES.txt
```

`<source_id>` is `NACA-TN-1757` or `NASA-TN-D-6967`. For example:
`data/empirical/digitised/NACA-TN-1757/fig4d/alpha06_d091_r2.csv` and
`data/empirical/digitised/NACA-TN-1757/fig4d/fig4d_r2.json`. Don't commit page
images. The PDF plus the page number is the record.

## 3. How to do one repeat (WPD 4.x; 5.x writes the same files)

0. **Before repeat 1, once per figure:** count the symbols of each chosen curve
   on the page and write the counts in `NOTES.txt`, along with any symbol you
   will skip and why. This decides *which* points to digitise. The repeats
   only measure *where* they are, so every repeat must mark the same points.
1. **Image:** render the page at the scan's own resolution,
   `pdftoppm -r 300 -f 23 -l 23 -png data/empirical/pdf/19930082415.pdf /tmp/fig4d`,
   and load that PNG into WPD. You can also load the PDF page directly.
2. **Axes:** choose "2D (X-Y) Plot". Click X1 and X2 on the **outermost
   labelled x ticks**, where the tick meets the axis line (use the zoom
   window). Then click Y1 and Y2 on the outermost labelled y ticks. That's two
   points per axis. Type the tick values, leave "Log scale" unticked (every
   axis here is linear), and leave "assume axes are perfectly aligned" unticked.
3. **Datasets:** rename "Default Dataset" to the curve name from the tables
   above (for example `alpha06_d091`), and add one dataset per further curve.
   The name must match the file name exactly.
4. **Points:** use Manual Extraction → Add Point. Click the **centre of each
   symbol** of the current dataset, and correct with Adjust Point and the arrow
   keys. Don't use Automatic Extraction or the AI assist.
5. **Export each curve:** View Data → pick the dataset → Sort by X, Ascending.
   Keep number formatting at "Ignore" and the separator at ", ", then Download
   .CSV. Rename the file to `<curve>_rN.csv`. ("Export All Data" also works, but
   the importer then needs the dataset name, so prefer one CSV per curve.)
6. **Save the project:** File → Save Project To Disk → Download JSON. Rename it
   to `<figure>_rN.json`. The .tar is also accepted, but it's bigger.

**Independence rules:** do the three repeats on different days, or at least in
separate sessions. Start each one from a freshly loaded image with a **new axis
calibration**. Never load a previous project, and never look at a previous
repeat's numbers until all three are done.

## 4. Check each curve before you send it

```
.venv/bin/python scripts/phase8/wpd_import.py check data/empirical/digitised/NACA-TN-1757/fig4d alpha06_d091
```

It prints `OK` and a table of mean, SD and spread per point, or `NOT OK` with
the reason. What it checks:

- All three CSVs and JSONs exist.
- Each CSV matches the same-named dataset in its own project JSON.
- The stored values follow from the stored axis calibration.
- The three repeats have the same number of points.
- No two repeats are identical copies.
- Every matched point agrees with repeat 1 to within **1 % of the calibrated
  axis span**, in both x and y.

If it fails, fix the repeat by re-doing it; don't edit numbers by hand.

Quick self-check while you're still in WPD:

- The number of points in View Data equals your NOTES.txt count.
- The x range of the points sits inside the plotted data, and no point was
  placed on the legend.
- Zoom over the whole plot: every marker sits on a symbol centre, and no symbol
  is left without a marker.
- After calibrating, hover over a grid intersection, for example
  (2.0, 0.80). The live read-out should be within about half a minor division
  of the true value. If it isn't, redo the calibration.

What happens next (for the maintainer): an entry module in
`scripts/phase8/empirical_entries/` creates the source and one experiment per
nozzle. All 15 nozzles share one independence key, because they are one rig and
one campaign. For each curve the module calls `wpd_import.enter_digitised_curve`
with the mappings above. The point values are the mean of the three repeats,
sigma is their sample SD (kind `digitisation`), and every observation gets a
`digitisation` row with the three repeat values and all three axis
calibrations. Points at P1/p0 ≈ 1.00 can come out slightly below 1 after
averaging. QA then flags `NPR < 1`. Don't nudge them. The maintainer decides.
