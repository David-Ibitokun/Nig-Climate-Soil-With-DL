import json
from pathlib import Path

import streamlit as st


_ROOT = Path(__file__).resolve().parents[1]


def _load_notebook(notebook_path: Path) -> dict:
    with notebook_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render(_: dict):
    st.title("📓 Combined Notebook")
    st.markdown(
        "This page renders the contents of the project notebook `New_Changes/yield_changes/Combined.ipynb`."
    )

    notebook_path = _ROOT / "New_Changes" / "yield_changes" / "Combined.ipynb"
    if not notebook_path.exists():
        st.error(f"Notebook not found at: {notebook_path}")
        return

    with notebook_path.open("rb") as f:
        st.download_button(
            label="Download Combined.ipynb",
            data=f,
            file_name="Combined.ipynb",
            mime="application/x-ipynb+json",
        )

    max_cells = st.sidebar.slider("Cells to render", min_value=10, max_value=500, value=120, step=10)
    show_code = st.sidebar.checkbox("Show code cells", value=True)

    try:
        nb = _load_notebook(notebook_path)
    except Exception as e:
        st.error(f"Failed to load notebook JSON: {e}")
        return

    cells = nb.get("cells", [])
    st.caption(f"Notebook cells: {len(cells)} (rendering first {min(max_cells, len(cells))})")

    rendered = 0
    for cell in cells[:max_cells]:
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", [])).strip("\n")
        if not source:
            continue

        if cell_type == "markdown":
            st.markdown(source)
            rendered += 1
        elif cell_type == "code" and show_code:
            st.code(source)
            rendered += 1

    if rendered == 0:
        st.info("No renderable cells found (try increasing the cell limit).")


if __name__ == "__main__":
    import sys

    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from data_loader import apply_global_style

    apply_global_style()
    render({})
