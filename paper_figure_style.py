"""Shared source-PDF typography: labels/ticks 10 pt, titles 12 pt, legends 8 pt."""
from pathlib import Path

SOURCE_FONTS = {"label": 10.0, "tick": 10.0, "title": 12.0, "legend": 8.0}


def apply_source_fonts(fig):
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(SOURCE_FONTS["label"])
        ax.yaxis.label.set_fontsize(SOURCE_FONTS["label"])
        ax.tick_params(axis="both", which="both", labelsize=SOURCE_FONTS["tick"])
        for axis in (ax.xaxis, ax.yaxis):
            axis.get_offset_text().set_fontsize(SOURCE_FONTS["tick"])
        ax.title.set_fontsize(SOURCE_FONTS["title"])
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(SOURCE_FONTS["legend"])
            legend.get_title().set_fontsize(SOURCE_FONTS["legend"])
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(SOURCE_FONTS["legend"])
        legend.get_title().set_fontsize(SOURCE_FONTS["legend"])


def save_paper_figure(fig, stem, figure_number):
    """Export PDF, PNG, and EPS with the shared font sizes."""
    apply_source_fonts(fig)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.canvas.draw()
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "eps"):
        fig.savefig(stem.with_suffix("."+suffix), bbox_inches="tight", pad_inches=.1,
                    dpi=200, format=suffix)
    return {"figure": figure_number, "source_font_sizes_pt": dict(SOURCE_FONTS)}
