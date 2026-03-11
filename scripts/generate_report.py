#!/usr/bin/env python3
"""
Generate a PDF report containing code and outputs from all simulation scripts.

This script:
1. Discovers all runnable Python scripts in the project
2. Executes each script and captures its output
3. Embeds any generated graphs/images
4. Generates a formatted PDF with alternating code/output sections
"""

import subprocess
import sys
import os
import glob
import re
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to sys.path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Try to import fpdf, install if not available
try:
    from fpdf import FPDF
except ImportError:
    print("Installing fpdf2 for PDF generation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

# Project root directory (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sanitize_text(text: str) -> str:
    """
    Sanitize text for PDF output by replacing problematic unicode characters.
    FPDF uses latin-1 encoding by default.
    """
    # Common unicode replacements
    replacements = {
        '\u2713': '[OK]',      # checkmark
        '\u2717': '[X]',       # X mark
        '\u2022': '*',         # bullet
        '\u2019': "'",         # right single quote
        '\u2018': "'",         # left single quote
        '\u201c': '"',         # left double quote
        '\u201d': '"',         # right double quote
        '\u2014': '--',        # em dash
        '\u2013': '-',         # en dash
        '\u2026': '...',       # ellipsis
        '\u00b7': '*',         # middle dot
        '\u2192': '->',        # right arrow
        '\u2190': '<-',        # left arrow
        '\u00d7': 'x',         # multiplication sign
        '\u00f7': '/',         # division sign
        '\u00b0': ' deg',      # degree symbol
        '\u00b2': '^2',        # superscript 2
        '\u00b3': '^3',        # superscript 3
        '\u03b3': 'gamma',     # greek gamma
        '\u03c1': 'rho',       # greek rho
        '\u03c6': 'phi',       # greek phi
        '\u0394': 'Delta',     # greek Delta
        '\u03c0': 'pi',        # greek pi
        '\u221e': 'inf',       # infinity
        '\u2265': '>=',        # greater than or equal
        '\u2264': '<=',        # less than or equal
        '\u2260': '!=',        # not equal
        '\u00b1': '+/-',       # plus minus
        '\u221a': 'sqrt',      # square root
        '\u2211': 'sum',       # summation
        '\u222b': 'integral',  # integral
        '\u2248': '~=',        # approximately equal
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Replace any remaining non-latin1 characters
    try:
        text.encode('latin-1')
    except UnicodeEncodeError:
        # Replace remaining problematic characters
        cleaned = []
        for char in text:
            try:
                char.encode('latin-1')
                cleaned.append(char)
            except UnicodeEncodeError:
                cleaned.append('?')
        text = ''.join(cleaned)

    return text


class ReportPDF(FPDF):
    """Custom PDF class with headers and footers."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "JetEngineSimulation - Code & Output Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_section_title(self, title: str):
        """Add a section title for a script."""
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(70, 130, 180)  # Steel blue
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, sanitize_text(title), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def add_code_section(self, code: str, filename: str):
        """Add a code section with syntax highlighting background."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(0, 100, 0)
        self.cell(0, 8, f"CODE: {filename}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

        self.set_font("Courier", "", 7)
        self.set_fill_color(245, 245, 245)  # Light gray background

        # Process code line by line
        lines = code.split('\n')
        for line in lines:
            # Truncate very long lines
            if len(line) > 115:
                line = line[:112] + "..."
            # Handle tabs
            line = line.replace('\t', '    ')
            # Sanitize for PDF
            line = sanitize_text(line)
            self.cell(0, 4, line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def add_output_section(self, output: str, success: bool = True):
        """Add an output section."""
        self.set_font("Helvetica", "B", 10)
        if success:
            self.set_text_color(0, 0, 139)  # Dark blue
            self.cell(0, 8, "OUTPUT:", new_x="LMARGIN", new_y="NEXT")
        else:
            self.set_text_color(139, 0, 0)  # Dark red
            self.cell(0, 8, "OUTPUT (Error):", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

        self.set_font("Courier", "", 7)
        if success:
            self.set_fill_color(240, 255, 240)  # Light green background
        else:
            self.set_fill_color(255, 240, 240)  # Light red background

        # Process output line by line
        lines = output.split('\n')
        for line in lines:
            # Truncate very long lines
            if len(line) > 115:
                line = line[:112] + "..."
            # Handle tabs
            line = line.replace('\t', '    ')
            # Sanitize for PDF
            line = sanitize_text(line)
            self.cell(0, 4, line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def add_image_section(self, image_path: str, caption: str = ""):
        """Add an image to the PDF with optional caption."""
        if not os.path.exists(image_path):
            return

        self.set_font("Helvetica", "B", 10)
        self.set_text_color(128, 0, 128)  # Purple
        label = f"GENERATED GRAPH: {caption}" if caption else "GENERATED GRAPH:"
        self.cell(0, 8, label, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

        # Calculate image dimensions to fit page
        page_width = self.w - 2 * self.l_margin
        max_height = 100  # Max height in mm

        try:
            self.image(image_path, x=self.l_margin, w=page_width, h=0)
        except Exception as e:
            self.set_font("Courier", "", 8)
            self.cell(0, 6, f"(Could not embed image: {e})", new_x="LMARGIN", new_y="NEXT")

        self.ln(5)

    def add_separator(self):
        """Add a visual separator between scripts."""
        self.ln(5)
        self.set_draw_color(150, 150, 150)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)


def get_all_scripts() -> list[dict]:
    """
    Get list of all useful Python scripts in the project.
    Returns list of dicts with 'path', 'name', 'output_images', 'can_run', and 'category' keys.
    """
    # All useful scripts organized by category
    # Format: (relative_path, [output_images], can_run, category)
    # can_run: True = execute and show output, False = show code only (requires torch/cantera)

    all_scripts = [
        # =====================================================================
        # CATEGORY: Core Engine Simulation (requires torch + cantera)
        # =====================================================================
        ("integrated_engine.py", [], False, "Core Engine Simulation"),
        ("scripts/optimization/optimize_blend.py", [], False, "Core Engine Simulation"),
        ("scripts/optimization/calibrate_lto.py", [], False, "Core Engine Simulation"),

        # =====================================================================
        # CATEGORY: Visualization Scripts (can run - only need pandas/matplotlib)
        # =====================================================================
        ("scripts/visualization/plot_validation.py", ["outputs/plots/validation_chart.png"], True, "Visualization"),
        ("scripts/visualization/pareto_visual.py", ["outputs/plots/pareto_front_2d.png", "outputs/plots/pareto_front_3d.png", "outputs/plots/pareto_correlations.png"], True, "Visualization"),
        ("scripts/visualization/marked_visuals.py", ["outputs/plots/marked_parallel_coordinates.png"], True, "Visualization"),
        ("scripts/visualization/optimization_plot.py", [], True, "Visualization"),

        # =====================================================================
        # CATEGORY: Test & Verification Scripts
        # =====================================================================
        ("tests/verify_thermo_fix.py", [], True, "Tests & Verification"),
        ("scripts/verify_requirements.py", [], False, "Tests & Verification"),
        ("scripts/test_emissions.py", [], False, "Tests & Verification"),

        # =====================================================================
        # CATEGORY: Data Analysis / EDA
        # =====================================================================
        ("evaluation/icao_eda.py", [], True, "Data Analysis"),

        # =====================================================================
        # CATEGORY: Simulation Components (library modules - code only)
        # =====================================================================
        ("simulation/compressor/compressor.py", [], False, "Simulation Components"),
        ("simulation/combustor/combustor.py", [], False, "Simulation Components"),
        ("simulation/turbine/turbine.py", [], False, "Simulation Components"),
        ("simulation/nozzle/nozzle.py", [], False, "Simulation Components"),
        ("simulation/fuels.py", [], False, "Simulation Components"),
        ("simulation/emissions.py", [], False, "Simulation Components"),
        ("simulation/thermo_utils.py", [], False, "Simulation Components"),
    ]

    # Build result list
    result = []
    for script_rel, images, can_run, category in all_scripts:
        script_path = PROJECT_ROOT / script_rel
        if script_path.exists():
            result.append({
                'path': script_path,
                'name': script_path.name,
                'rel_path': script_rel,
                'output_images': [str(PROJECT_ROOT / img) for img in images],
                'can_run': can_run,
                'category': category
            })

    return result


def get_runnable_scripts() -> list[dict]:
    """Legacy function - returns all scripts."""
    return get_all_scripts()


def run_script(script_path: Path, timeout: int = 180) -> tuple[str, bool]:
    """
    Run a Python script and capture its output.
    Scripts are run from the project root with proper PYTHONPATH.

    Returns:
        Tuple of (output_text, success_bool)
    """
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['MPLBACKEND'] = 'Agg'  # Use non-interactive matplotlib backend

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env=env
        )

        output = result.stdout
        if result.stderr:
            # Filter out common matplotlib/macOS warnings
            stderr_lines = result.stderr.split('\n')
            filtered_stderr = []
            for line in stderr_lines:
                if 'IMKClient' in line or 'IMKInputSession' in line:
                    continue
                if line.strip():
                    filtered_stderr.append(line)
            if filtered_stderr:
                output += "\n--- STDERR ---\n" + '\n'.join(filtered_stderr)

        success = result.returncode == 0

        if not output.strip():
            output = "(No output produced)"

        return output, success

    except subprocess.TimeoutExpired:
        return f"(Script timed out after {timeout} seconds)", False
    except Exception as e:
        return f"(Error running script: {str(e)})", False


def read_file_content(file_path: Path, max_lines: int = 100) -> str:
    """Read file content, limiting to max_lines for readability."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) > max_lines:
            # Show first and last portions
            half = max_lines // 2
            content = ''.join(lines[:half])
            content += f"\n\n... [{len(lines) - max_lines} lines omitted for brevity] ...\n\n"
            content += ''.join(lines[-half:])
            return content
        else:
            return ''.join(lines)

    except Exception as e:
        return f"(Error reading file: {str(e)})"


def find_generated_images(before_images: set, after_images: set) -> list[str]:
    """Find newly generated or modified images."""
    # Check for new images
    new_images = after_images - before_images

    # Also check for modified images (by modification time)
    modified = []
    for img in after_images:
        if os.path.exists(img):
            modified.append(img)

    return list(new_images) + [m for m in modified if m not in new_images]


def get_existing_images() -> set:
    """Get set of existing PNG images in project root."""
    return set(glob.glob(str(PROJECT_ROOT / "outputs" / "plots" / "*.png")))


def generate_report(output_path: str = "simulation_report.pdf",
                   run_scripts: bool = True,
                   scripts: list[dict] = None):
    """
    Generate the PDF report.

    Args:
        output_path: Path for the output PDF
        run_scripts: If True, execute scripts that can run. If False, just show code.
        scripts: Optional list of specific scripts to include
    """
    print("=" * 60)
    print("JetEngineSimulation Report Generator")
    print("=" * 60)

    if scripts is None:
        scripts = get_all_scripts()

    if not scripts:
        print("No scripts found!")
        return

    # Group scripts by category for display
    categories = {}
    for s in scripts:
        cat = s.get('category', 'Other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(s)

    print(f"\nFound {len(scripts)} scripts to process:")
    for cat, cat_scripts in categories.items():
        print(f"\n  {cat}:")
        for s in cat_scripts:
            run_status = "[run]" if s.get('can_run', True) else "[code only]"
            print(f"    - {s['name']} {run_status}")
    print()

    # Create PDF
    pdf = ReportPDF()
    pdf.alias_nb_pages()

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.ln(40)
    pdf.cell(0, 20, "JetEngineSimulation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, "Code & Output Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Total Scripts: {len(scripts)}", align="C", new_x="LMARGIN", new_y="NEXT")

    # Count runnable vs code-only
    runnable_count = sum(1 for s in scripts if s.get('can_run', True))
    pdf.cell(0, 10, f"Executable: {runnable_count} | Code-only: {len(scripts) - runnable_count}", align="C", new_x="LMARGIN", new_y="NEXT")

    # Table of contents organized by category
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    script_num = 1
    for cat in categories:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(70, 130, 180)
        pdf.cell(0, 8, cat, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        for script in categories[cat]:
            status = "" if script.get('can_run', True) else " [code only]"
            pdf.cell(0, 6, f"  {script_num}. {script['name']}{status}", new_x="LMARGIN", new_y="NEXT")
            script_num += 1
        pdf.ln(2)

    # Process each script by category
    script_num = 1
    for cat, cat_scripts in categories.items():
        # Add category header page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(70, 130, 180)
        pdf.ln(20)
        pdf.cell(0, 15, cat, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"{len(cat_scripts)} files", align="C", new_x="LMARGIN", new_y="NEXT")

        for script_info in cat_scripts:
            script_path = script_info['path']
            expected_images = script_info.get('output_images', [])
            can_run = script_info.get('can_run', True)
            rel_path = script_info.get('rel_path', script_info['name'])

            print(f"\n[{script_num}/{len(scripts)}] Processing: {rel_path}")

            pdf.add_page()
            pdf.add_section_title(f"{script_num}. {rel_path}")

            # Add file description based on category
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            if not can_run:
                pdf.cell(0, 6, "Note: This file requires torch/cantera - showing code only", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

            # Add code
            print(f"  Reading code...")
            code = read_file_content(script_path, max_lines=150)
            pdf.add_code_section(code, script_info['name'])

            # Run script if possible
            if run_scripts and can_run:
                print(f"  Running script...")

                # Track images before running
                images_before = get_existing_images()

                output, success = run_script(script_path)
                status = "OK" if success else "FAILED"
                print(f"  [{status}] Execution {'completed' if success else 'failed'}")
                pdf.add_output_section(output, success)

                # Find and embed generated images
                images_after = get_existing_images()

                # Add expected output images
                images_to_add = []
                for img_path in expected_images:
                    if os.path.exists(img_path):
                        images_to_add.append(img_path)

                # Also check for any new images
                new_images = images_after - images_before
                for img in new_images:
                    if img not in images_to_add:
                        images_to_add.append(img)

                # Add images to PDF
                for img_path in images_to_add:
                    img_name = os.path.basename(img_path)
                    print(f"  Adding image: {img_name}")
                    pdf.add_image_section(img_path, img_name)
            elif not can_run:
                # Code-only file - add note instead of output
                pdf.set_font("Helvetica", "I", 10)
                pdf.set_fill_color(255, 250, 230)  # Light yellow
                pdf.cell(0, 8, "OUTPUT: Requires torch and cantera to execute", fill=True, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)
                print(f"  [SKIP] Code-only (requires torch/cantera)")
            else:
                pdf.add_output_section("(Script execution skipped)", True)

            pdf.add_separator()
            script_num += 1

    # Save PDF
    print(f"\nSaving PDF to: {output_path}")
    pdf.output(output_path)
    print(f"Report generated successfully!")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    """Main entry point with CLI argument handling."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PDF report with code and outputs from simulation scripts"
    )
    parser.add_argument(
        "-o", "--output",
        default="outputs/plots/simulation_report.pdf",
        help="Output PDF filename (default: outputs/plots/simulation_report.pdf)"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Don't execute scripts, just include code"
    )

    args = parser.parse_args()

    generate_report(
        output_path=args.output,
        run_scripts=not args.no_run
    )


if __name__ == "__main__":
    main()
