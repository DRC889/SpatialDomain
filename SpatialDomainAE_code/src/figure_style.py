"""Shared matplotlib style for all manuscript figures.

Import and call `apply_style()` at the top of any figure-generation script
to keep fonts, sizes, and colors consistent across figures.
"""
import matplotlib
import matplotlib.pyplot as plt


# Method palette — used consistently across fig1, fig4, etc.
METHOD_COLORS = {
    "SpatialDomainAE": "#1565C0",   # deep blue (ours)
    "SpatialGATAE":    "#64B5F6",
    "ExprOnlyAE":      "#90CAF9",
    "STAGATE":         "#E65100",
    "GraphST":         "#2E7D32",
    "SpaGCN":          "#6A1B9A",
}
METHOD_EDGE_COLORS = {
    "SpatialDomainAE": "#0D47A1",
    "SpatialGATAE":    "#1976D2",
    "ExprOnlyAE":      "#BBDEFB",
    "STAGATE":         "#BF360C",
    "GraphST":         "#1B5E20",
    "SpaGCN":          "#4A148C",
}
METHOD_DISPLAY = {
    "SpatialDomainAE": "SpatialDomainAE\n(Ours)",
    "SpatialGATAE":    "SpatialGATAE",
    "ExprOnlyAE":      "ExprOnlyAE",
    "STAGATE":         "STAGATE",
    "GraphST":         "GraphST",
    "SpaGCN":          "SpaGCN",
}

# Sample palette — used consistently across attention/biology figures
SAMPLE_DISPLAY = {
    "Ctrl": "Control",
    "1DPI": "1 DPI",
    "3DPI": "3 DPI",
    "7DPI": "7 DPI",
}
SAMPLES_ORDERED = ["Ctrl", "1DPI", "3DPI", "7DPI"]

# Spatial domain (ground-truth) colors — used in fig1
DOMAIN_COLORS = {
    "CTX1-4": "#7FC7AF", "CTX5":   "#59A986", "CTX6":   "#3A8B67",
    "lCTX4-5":"#B5D8C7", "lCTX6":  "#8EC4A8",
    "HIP":    "#FFD54F", "AMY":    "#F9A825", "PIR":    "#FFB74D",
    "CP":     "#CE93D8", "PAL":    "#AB47BC", "TH":     "#7E57C2",
    "HY":     "#BA68C8",
    "CS":     "#64B5F6", "FT":     "#E0E0E0", "GLS":    "#8D6E63",
    "LV":     "#CFD8DC",
    "ISD1c":  "#EF5350", "ISD1p":  "#E57373",
    "ISD3c":  "#D32F2F", "ISD3p":  "#EF9A9A",
    "ISD7c":  "#C62828", "ISD7p":  "#FFCDD2",
}

# Domain category colors — used in fig11 (attention by domain)
DOMAIN_CATEGORY_COLORS = {
    "Intact anatomy":    "#4A90D9",
    "Lesioned cortex":   "#7BB3E0",
    "Glial scar":        "#8D6E63",
    "Ischemic penumbra": "#EF9A9A",
    "Ischemic core":     "#D32F2F",
}


def apply_style():
    """Apply manuscript-wide matplotlib style."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans",
                             "Liberation Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        # Figure
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Axes
        "axes.facecolor": "white",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Ticks
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # PDF text as text not paths (editable in Illustrator)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, name, figdir):
    """Save a figure as both PDF and PNG at 300 dpi.

    `name` may include a subdirectory (e.g. "archive/figX"); the parent
    directory is created on demand.
    """
    from pathlib import Path
    figdir = Path(figdir)
    out_base = figdir / name
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")
