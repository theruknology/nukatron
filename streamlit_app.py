import json
import tempfile
from pathlib import Path
from typing import Optional, List

import streamlit as st


def _safe_name(name: str) -> str:
    # keep it simple: avoid paths, keep basic chars
    base = Path(name).name
    if not base:
        return "uploaded.pdb"
    return base


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


def main() -> None:
    st.set_page_config(page_title="Nukatron", layout="wide")

    st.title("Nukatron")
    st.caption("Upload a PDB, run the analysis pipeline, and download outputs.")

    with st.sidebar:
        st.header("Run options")
        legacy_ps = st.checkbox("Generate legacy PostScript (output.ps)", value=True)
        legacy_pdf = st.checkbox("Generate legacy PDF (output.pdf)", value=False)
        ligand_resnames_raw = st.text_input(
            "Ligand residue name(s) (comma-separated)",
            value="",
            help="Optional. Example: MBC or LIG. Leave blank to auto-detect.",
        )

        st.divider()
        st.write("Outputs always include the network visualization when PyVis is available.")

    uploaded = st.file_uploader("Upload a PDB file", type=["pdb"])

    if uploaded is None:
        st.info("Upload a PDB to begin.")
        return

    ligand_resnames: Optional[List[str]]
    ligand_resnames_raw = ligand_resnames_raw.strip()
    if ligand_resnames_raw:
        ligand_resnames = [x.strip() for x in ligand_resnames_raw.split(",") if x.strip()]
    else:
        ligand_resnames = None

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Input")
        st.write(f"**Filename:** `{_safe_name(uploaded.name)}`")
        st.write(f"**Size:** {uploaded.size} bytes")

        run_btn = st.button("Run analysis", type="primary")

    if not run_btn:
        st.stop()

    with st.spinner("Running Nukatron analysis... this can take a bit depending on structure size"):
        from main import analyze_single_pdb

        with tempfile.TemporaryDirectory(prefix="nukatron_run_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / _safe_name(uploaded.name)
            input_path.write_bytes(uploaded.getvalue())

            out_dir = tmpdir_path / "results"
            results = analyze_single_pdb(
                str(input_path),
                str(out_dir),
                ligand_resnames=ligand_resnames,
                legacy_ps=legacy_ps,
                legacy_pdf=legacy_pdf,
            )

            st.success("Analysis complete")

            with col_right:
                st.subheader("Summary")
                st.json(
                    {
                        "n_interactions": results.get("n_interactions", 0),
                        "n_water_bridges": results.get("n_water_bridges", 0),
                        "outputs": results.get("outputs", {}),
                    }
                )

            st.subheader("Downloads")
            outputs = results.get("outputs", {}) or {}

            # Standard expected outputs
            ordered_keys = [
                "legacy_ps",
                "legacy_pdf",
                "network_visualization",
                "interactions",
                "water_bridges",
                "residue_summary",
                "summary",
            ]

            cols = st.columns(2)
            col_i = 0
            for key in ordered_keys:
                p = outputs.get(key)
                if not p:
                    continue
                path = Path(p)
                data = _read_bytes(path)
                if not data:
                    continue
                with cols[col_i % 2]:
                    st.download_button(
                        label=f"Download {key}",
                        data=data,
                        file_name=path.name,
                        mime="application/octet-stream",
                    )
                col_i += 1

            # Render network.html if present
            net_path = outputs.get("network_visualization")
            if net_path:
                net_file = Path(net_path)
                if net_file.exists():
                    st.subheader("Network visualization")
                    st.components.v1.html(net_file.read_text(encoding="utf-8"), height=700, scrolling=True)

            # Also show analysis_summary.json in-page
            summary_path = outputs.get("summary")
            if summary_path:
                sp = Path(summary_path)
                if sp.exists():
                    st.subheader("analysis_summary.json")
                    try:
                        st.json(json.loads(sp.read_text(encoding="utf-8")))
                    except Exception:
                        st.code(sp.read_text(encoding="utf-8"), language="json")


if __name__ == "__main__":
    main()
