import logging
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResidueKey:
    resname: str
    resid: int

    def label(self) -> str:
        return f"{self.resname}{self.resid}"


class NucplotExporter:
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        water_bridges_df: Optional[pd.DataFrame] = None,
        residue_summary_df: Optional[pd.DataFrame] = None,
        ligand_label: Optional[str] = None,
    ):
        self.interactions = interactions_df if interactions_df is not None else pd.DataFrame()
        self.water_bridges = water_bridges_df if water_bridges_df is not None else pd.DataFrame()
        self.residue_summary = residue_summary_df if residue_summary_df is not None else pd.DataFrame()
        self.ligand_label = ligand_label

    def _infer_ligand_label(self) -> str:
        if self.ligand_label:
            return self.ligand_label
        for col in ("ligand_resname", "ligand_resid"):
            if col not in self.interactions.columns:
                continue
        if "ligand_resname" in self.interactions.columns:
            series = self.interactions["ligand_resname"].dropna()
            if len(series) > 0:
                return str(series.mode().iloc[0])
        return "LIG"

    def _dna_base_letter(self, dna_resname: str) -> str:
        res = (dna_resname or "").strip().upper()
        if res in {"DA", "ADE", "A"}:
            return "A"
        if res in {"DT", "THY", "T"}:
            return "T"
        if res in {"DG", "GUA", "G"}:
            return "G"
        if res in {"DC", "CYT", "C"}:
            return "C"
        if len(res) == 1:
            return res
        if len(res) >= 2 and res[0] == "D":
            return res[1]
        return res[:1] if res else "N"

    def _ps_escape(self, s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def _export_nucplot_like_ps(self, output_path: Path, title: Optional[str]) -> str:
        residue_summary = self.residue_summary if self.residue_summary is not None else pd.DataFrame()
        dna_df = residue_summary[residue_summary.get("group", pd.Series(dtype=str)) == "dna"].copy() if len(residue_summary) > 0 else pd.DataFrame()
        if len(dna_df) == 0:
            dna_residues = self._collect_dna_residues()
            if len(dna_residues) == 0:
                raise ValueError("No DNA residues found")
            dna_df = pd.DataFrame(
                {
                    "resname": [r.resname for r in dna_residues],
                    "resid": [r.resid for r in dna_residues],
                    "chain_id": [None] * len(dna_residues),
                    "segid": [None] * len(dna_residues),
                }
            )

        chain_col = "chain_id" if "chain_id" in dna_df.columns and dna_df["chain_id"].notna().any() else ("segid" if "segid" in dna_df.columns and dna_df["segid"].notna().any() else None)
        if chain_col is not None:
            chains = []
            for _, sub in dna_df.sort_values([chain_col, "resid"]).groupby(chain_col, dropna=False):
                chains.append(sub.sort_values("resid").reset_index(drop=True))
            chains = [c for c in chains if len(c) > 0]
        else:
            chains = []

        if len(chains) >= 2:
            chain_a = chains[0]
            chain_b = chains[1]
        else:
            ordered = dna_df.sort_values("resid").reset_index(drop=True)
            half = int(math.ceil(len(ordered) / 2))
            chain_a = ordered.iloc[:half].reset_index(drop=True)
            chain_b = ordered.iloc[half:].reset_index(drop=True)

        if len(chain_b) > 0 and len(chain_a) == len(chain_b):
            chain_b = chain_b.iloc[::-1].reset_index(drop=True)

        n_rows = max(len(chain_a), len(chain_b))
        if n_rows == 0:
            raise ValueError("No DNA residues found")

        y_top = 733.74
        dy = 36.29
        ys = [y_top - i * dy for i in range(n_rows)]

        x_left_base = 281.85
        x_right_base = 318.15
        x_bp_l = 290.83
        x_bp_r = 307.17

        x_left_sugar = 254.22
        x_left_sugar_tip = 267.33
        x_left_backbone = 245.56

        x_right_sugar = 345.78
        x_right_sugar_tip = 332.67
        x_right_backbone = 354.44

        def base_color(letter: str) -> str:
            l = (letter or "").upper()
            if l == "A":
                return "RED"
            if l == "T":
                return "BLUE"
            if l == "G":
                return "GREEN"
            if l == "C":
                return "ORANGE"
            return "BLACK"

        def complement_base(letter: str) -> str:
            l = (letter or "").upper()
            if l == "A":
                return "T"
            if l == "T":
                return "A"
            if l == "G":
                return "C"
            if l == "C":
                return "G"
            if l == "U":
                return "A"
            return l

        def dna_target_from_atom(atom_name: Optional[str]) -> str:
            a = (str(atom_name) if atom_name is not None else "").strip().upper()
            if a in {"P", "OP1", "OP2"}:
                return "phosphate"
            if "'" in a or a in {"O3*", "O5*", "C1*", "C2*", "C3*", "C4*", "C5*"}:
                return "sugar"
            if a.startswith("H") and ("'" in a or a.endswith("*")):
                return "sugar"
            return "base"

        def clip_circle_border(cx: float, cy: float, r: float, ox: float, oy: float) -> Tuple[float, float]:
            dx, dy = ox - cx, oy - cy
            d = math.hypot(dx, dy)
            if d <= 1e-6:
                return (cx + r, cy)
            return (cx + r * dx / d, cy + r * dy / d)

        def clip_rect_border(cx: float, cy: float, half_w: float, half_h: float, ox: float, oy: float) -> Tuple[float, float]:
            dx, dy = ox - cx, oy - cy
            if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
                return (cx + half_w, cy)
            sx = (half_w / abs(dx)) if abs(dx) > 1e-6 else float("inf")
            sy = (half_h / abs(dy)) if abs(dy) > 1e-6 else float("inf")
            s = min(sx, sy)
            return (cx + dx * s, cy + dy * s)

        def clip_horizontal_from_label_edge(x: float, y: float, side: int, pad: float = 3.0) -> Tuple[float, float]:
            return (x + pad if side > 0 else x - pad, y)

        ps: List[str] = []

        ps.append("%!PS-Adobe-3.0")
        ps.append("%%Creator: Nukatron")
        ps.append("%%DocumentNeededResources: font Times-Bold Symbol")
        ps.append("%%BoundingBox: 0 0 650 951")
        ps.append("%%Pages: 1")
        ps.append(f"%%Title: {self._ps_escape(title or 'nukatron')}")
        ps.append("%%EndComments")
        ps.append("%%BeginProlog")
        ps.append("/L { moveto lineto stroke } bind def")
        ps.append("/W { setlinewidth } bind def")
        ps.append("/D { setdash } bind def")
        ps.append("/Black { 0 0 0 setrgbcolor } bind def")
        ps.append("/BLACK { 0 0 0 setrgbcolor } bind def")
        ps.append("/RED { 1 0 0 setrgbcolor } bind def")
        ps.append("/GREEN { 0 0.8 0.2 setrgbcolor } bind def")
        ps.append("/BLUE { 0.4 0.4 1 setrgbcolor } bind def")
        ps.append("/ORANGE { 0.8 0.5 0 setrgbcolor } bind def")
        ps.append("/PURPLE { 0.4 0 0.7 setrgbcolor } bind def")
        ps.append("/BRICK_RED { 0.8 0 0 setrgbcolor } bind def")
        ps.append("/DARK_BLUE { 0 0 1 setrgbcolor } bind def")
        ps.append("/BROWN { 0.5 0 0 setrgbcolor } bind def")
        ps.append("/Print { /Times-Bold findfont exch scalefont setfont show } bind def")
        ps.append("/Center { dup /Times-Bold findfont exch scalefont setfont exch stringwidth pop -2 div exch -3 div rmoveto } bind def")
        ps.append("/Lalign { dup /Times-Bold findfont exch scalefont setfont exch stringwidth pop pop -3 div 0.0 exch rmoveto } bind def")
        ps.append("/Ralign { dup /Times-Bold findfont exch scalefont setfont exch stringwidth pop -1 mul exch -3 div rmoveto } bind def")
        ps.append("/Circle { gsave newpath 0 360 arc gsave 1 setgray fill grestore stroke grestore } bind def")
        ps.append("%%EndProlog")
        ps.append("%%BeginSetup")
        ps.append("1 setlinecap 1 setlinejoin 1 setlinewidth 0 setgray [ ] 0 setdash newpath")
        ps.append("%%EndSetup")
        ps.append("%%Page: 1 1")
        ps.append("/NucplotSave save def")
        ps.append(" -1.00 -1.00 moveto 650.00 -1.00 lineto 650.00 951.00 lineto -1.00 951.00 lineto closepath")
        ps.append("gsave 1 setgray fill grestore stroke")
        ps.append("50.00 50.00 moveto 550.00 50.00 lineto 550.00 800.00 lineto 50.00 800.00 lineto closepath gsave 1 setgray fill grestore stroke")

        chain_a_label = str(chain_a[chain_col].iloc[0]) if chain_col and len(chain_a) > 0 else "A"
        chain_b_label = str(chain_b[chain_col].iloc[0]) if chain_col and len(chain_b) > 0 else "B"
        if chain_a_label == chain_b_label:
            chain_b_label = "B"

        ps.append("BLACK")
        ps.append(f" {x_left_base:6.2f} {max(ys) + 24.2:7.2f} moveto")
        ps.append("(5') 14 Center")
        ps.append("(5') 14 Print")
        ps.append(f" {x_left_base - 36.29:6.2f} {max(ys) + 43.6:7.2f} moveto")
        ps.append(f"(Chain {self._ps_escape(chain_a_label)}) 10 Center")
        ps.append(f"(Chain {self._ps_escape(chain_a_label)}) 10 Print")
        ps.append(f" {x_right_base + 36.29:6.2f} {max(ys) + 43.6:7.2f} moveto")
        ps.append(f"(Chain {self._ps_escape(chain_b_label)}) 10 Center")
        ps.append(f"(Chain {self._ps_escape(chain_b_label)}) 10 Print")
        ps.append(f" {x_right_base:6.2f} {min(ys) - 24.2:7.2f} moveto")
        ps.append("(5') 14 Center")
        ps.append("(5') 14 Print")

        if title:
            ps.append(f" {300.00:6.2f} {82.00:7.2f} moveto")
            ps.append(f"({self._ps_escape(title)}) 22 Center")
            ps.append(f"({self._ps_escape(title)}) 22 Print")

        def draw_sugar_left(y: float, resid: int) -> None:
            ps.append("gsave")
            ps.append("   0.83 W")
            ps.append("BROWN")
            ps.append(f" {x_left_sugar:6.2f} {y + 4.29:7.2f} moveto {x_left_sugar:6.2f} {y - 4.29:7.2f} lineto")
            ps.append(f" {x_left_sugar + 8.19:6.2f} {y - 6.87:7.2f} lineto {x_left_sugar_tip:6.2f} {y:7.2f} lineto {x_left_sugar + 7.89:6.2f} {y + 6.97:7.2f} lineto")
            ps.append("gsave 1 setgray fill grestore closepath stroke")
            ps.append("grestore")
            ps.append(f" {x_left_sugar + 5.85:6.2f} {y:7.2f} moveto")
            ps.append(f"({resid}) 7 Center")
            ps.append(f"({resid}) 7 Print")

        def draw_sugar_right(y: float, resid: int) -> None:
            ps.append("gsave")
            ps.append("   0.83 W")
            ps.append("BROWN")
            ps.append(f" {x_right_sugar:6.2f} {y + 4.29:7.2f} moveto {x_right_sugar:6.2f} {y - 4.29:7.2f} lineto")
            ps.append(f" {x_right_sugar - 8.19:6.2f} {y - 6.87:7.2f} lineto {x_right_sugar_tip:6.2f} {y:7.2f} lineto {x_right_sugar - 7.89:6.2f} {y + 6.97:7.2f} lineto")
            ps.append("gsave 1 setgray fill grestore closepath stroke")
            ps.append("grestore")
            ps.append(f" {x_right_sugar - 5.85:6.2f} {y:7.2f} moveto")
            ps.append(f"({resid}) 7 Center")
            ps.append(f"({resid}) 7 Print")

        def draw_phosphate_left(y: float, is_end: bool) -> None:
            if is_end:
                ps.append("gsave")
                ps.append("   0.83 W")
                ps.append("BLACK")
                ps.append(f" {x_left_sugar:6.2f} {y + 4.29:7.2f} {x_left_backbone:6.2f} {y + 13.79:7.2f} L")
                ps.append("RED")
                ps.append(f" {x_left_backbone:6.2f} {y + 13.79:7.2f} 1.09 Circle")
                ps.append("grestore")
                return
            ps.append("gsave")
            ps.append("   0.83 W")
            ps.append("BLACK")
            ps.append(f" {x_left_sugar:6.2f} {y + 4.29:7.2f} {x_left_backbone:6.2f} {y + 18.15:7.2f} L")
            ps.append(f" {x_left_backbone:6.2f} {y + 18.15:7.2f} {x_left_sugar:6.2f} {y + 32.01:7.2f} L")
            ps.append("PURPLE")
            ps.append(f" {x_left_backbone:6.2f} {y + 18.15:7.2f} 4.36 Circle")
            ps.append(f" {x_left_backbone:6.2f} {y + 18.15:7.2f} moveto")
            ps.append("(P) 5 Center")
            ps.append("(P) 5 Print")
            ps.append("grestore")

        def draw_phosphate_right(y: float, is_end: bool) -> None:
            if is_end:
                ps.append("gsave")
                ps.append("   0.83 W")
                ps.append("BLACK")
                ps.append(f" {x_right_sugar:6.2f} {y - 4.29:7.2f} {x_right_backbone:6.2f} {y - 18.15:7.2f} L")
                ps.append(f" {x_right_backbone:6.2f} {y - 18.15:7.2f} {x_right_sugar:6.2f} {y - 32.01:7.2f} L")
                ps.append("PURPLE")
                ps.append(f" {x_right_backbone:6.2f} {y - 18.15:7.2f} 4.36 Circle")
                ps.append(f" {x_right_backbone:6.2f} {y - 18.15:7.2f} moveto")
                ps.append("(P) 5 Center")
                ps.append("(P) 5 Print")
                ps.append("BLACK")
                ps.append(f" {x_right_sugar:6.2f} {y + 4.29:7.2f} {x_right_backbone:6.2f} {y + 13.79:7.2f} L")
                ps.append("RED")
                ps.append(f" {x_right_backbone:6.2f} {y + 13.79:7.2f} 1.09 Circle")
                ps.append("grestore")
                return
            ps.append("gsave")
            ps.append("   0.83 W")
            ps.append("BLACK")
            ps.append(f" {x_right_sugar:6.2f} {y - 4.29:7.2f} {x_right_backbone:6.2f} {y - 18.15:7.2f} L")
            ps.append(f" {x_right_backbone:6.2f} {y - 18.15:7.2f} {x_right_sugar:6.2f} {y - 32.01:7.2f} L")
            ps.append("PURPLE")
            ps.append(f" {x_right_backbone:6.2f} {y - 18.15:7.2f} 4.36 Circle")
            ps.append(f" {x_right_backbone:6.2f} {y - 18.15:7.2f} moveto")
            ps.append("(P) 5 Center")
            ps.append("(P) 5 Print")
            ps.append("grestore")

        for i in range(n_rows):
            y = ys[i]
            left = chain_a.iloc[i] if i < len(chain_a) else None
            right = chain_b.iloc[i] if i < len(chain_b) else None

            if left is not None:
                lb = self._dna_base_letter(str(left.get("resname")))
                ps.append("gsave")
                ps.append(base_color(lb))
                ps.append(f" {x_left_base:6.2f} {y:7.2f} moveto")
                ps.append(f"({self._ps_escape(lb)}) 19 Center")
                ps.append(f"({self._ps_escape(lb)}) 19 Print")
                ps.append("grestore")
                draw_sugar_left(y, int(left.get("resid")))
                draw_phosphate_left(y, is_end=(i == 0))

            if right is not None:
                rb = self._dna_base_letter(str(right.get("resname")))
                if left is not None:
                    expected = complement_base(lb)
                    if rb != expected:
                        rb = expected
                ps.append("gsave")
                ps.append(base_color(rb))
                ps.append(f" {x_right_base:6.2f} {y:7.2f} moveto")
                ps.append(f"({self._ps_escape(rb)}) 19 Center")
                ps.append(f"({self._ps_escape(rb)}) 19 Print")
                ps.append("grestore")
                draw_sugar_right(y, int(right.get("resid")))
                draw_phosphate_right(y, is_end=(i == 0))

            if left is not None and right is not None:
                ps.append("gsave")
                ps.append("   1.27 W")
                ps.append("BLACK")
                ps.append(f" {x_bp_l:6.2f} {y:7.2f} {x_bp_r:6.2f} {y:7.2f} L")
                ps.append("grestore")

        ligand_x, ligand_y = 300.00, float(np.median(ys))
        ligand_half_w, ligand_half_h = 16.0, 8.0

        dna_y_by_label: Dict[str, float] = {}
        dna_anchor_by_label: Dict[str, Tuple[float, float]] = {}
        dna_side_by_label: Dict[str, int] = {}
        dna_sugar_border_by_label: Dict[str, Tuple[float, float]] = {}
        dna_phosphate_center_by_label: Dict[str, Tuple[float, float]] = {}
        for i in range(n_rows):
            y = ys[i]
            if i < len(chain_a):
                a_label = f"{chain_a.iloc[i].get('resname')}{int(chain_a.iloc[i].get('resid'))}"
                dna_y_by_label[a_label] = y
                dna_anchor_by_label[a_label] = (x_left_base, y)
                dna_side_by_label[a_label] = -1
                dna_sugar_border_by_label[a_label] = (x_left_sugar, y)
                dna_phosphate_center_by_label[a_label] = (x_left_backbone, y + 18.15)
            if i < len(chain_b):
                b_label = f"{chain_b.iloc[i].get('resname')}{int(chain_b.iloc[i].get('resid'))}"
                dna_y_by_label[b_label] = y
                dna_anchor_by_label[b_label] = (x_right_base, y)
                dna_side_by_label[b_label] = 1
                dna_sugar_border_by_label[b_label] = (x_right_sugar, y)
                dna_phosphate_center_by_label[b_label] = (x_right_backbone, y - 18.15)

        protein_chain_by_key: Dict[Tuple[str, int], Optional[str]] = {}
        if len(residue_summary) > 0 and "group" in residue_summary.columns:
            prot_df = residue_summary[residue_summary["group"] == "protein"].copy()
            if len(prot_df) > 0:
                for _, r in prot_df.iterrows():
                    try:
                        protein_chain_by_key[(str(r.get("resname")), int(r.get("resid")))] = r.get("chain_id")
                    except Exception:
                        continue

        def style(interaction_type: str) -> Tuple[str, str]:
            it = (interaction_type or "").strip()
            if it == "HBond":
                return ("DARK_BLUE", "[  1  1 ]")
            if it == "VdW":
                return ("BRICK_RED", "[  2  3 ]")
            if it == "PiStacking":
                return ("BLACK", "[  3  2 ]")
            if it == "SaltBridge":
                return ("BLACK", "[  1  2 ]")
            if it == "CationPi":
                return ("BLACK", "[  2  2 ]")
            if it == "WaterBridge":
                return ("BLACK", "[  1  1 ]")
            return ("BLACK", "[ ]")

        def emit_line(x1: float, y1: float, x2: float, y2: float, interaction_type: str, water: bool = False) -> None:
            col, dash = style(interaction_type)
            ps.append("gsave")
            ps.append("   0.29 W")
            ps.append(col)
            ps.append(f"{dash} 0 setdash")
            ps.append(f" {x1:6.2f} {y1:7.2f} {x2:6.2f} {y2:7.2f} L")
            ps.append("[] 0 setdash")
            if water:
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ps.append(f" {xm:6.2f} {ym:7.2f} 2.20 Circle")
            ps.append("grestore")

        def protein_label(prot_resname: str, prot_resid: int) -> str:
            chain = protein_chain_by_key.get((prot_resname, prot_resid))
            if chain and str(chain).strip() not in {"", "nan", "None"}:
                base = f"{prot_resname}{prot_resid}({chain})"
            else:
                base = f"{prot_resname}{prot_resid}"
            return base

        label_events: List[Dict[str, object]] = []

        if len(self.interactions) > 0:
            # Protein-DNA: aggregate to residue-level events with counts per interaction type
            prot_dna = self.interactions[self.interactions.get("pair_type", "") == "Protein-DNA"].copy()
            if len(prot_dna) > 0 and {"protein_resname", "protein_resid", "dna_resname", "dna_resid", "interaction_type"}.issubset(
                set(prot_dna.columns)
            ):
                if "atom2_name" in prot_dna.columns:
                    prot_dna["dna_target"] = prot_dna["atom2_name"].apply(dna_target_from_atom)
                else:
                    prot_dna["dna_target"] = "base"
                prot_dna = prot_dna.dropna(subset=["protein_resname", "protein_resid", "dna_resname", "dna_resid", "interaction_type"])
                grouped = prot_dna.groupby(
                    ["protein_resname", "protein_resid", "dna_resname", "dna_resid", "interaction_type"],
                    dropna=False,
                ).agg(
                    n=("interaction_type", "size"),
                    dna_target=("dna_target", lambda s: str(s.value_counts().idxmax()) if len(s) else "base"),
                ).reset_index()
                for _, r in grouped.iterrows():
                    prot_resname = str(r["protein_resname"])
                    prot_resid = int(r["protein_resid"])
                    dna_label = f"{r['dna_resname']}{int(r['dna_resid'])}"
                    if dna_label not in dna_y_by_label:
                        continue
                    itype = str(r["interaction_type"])
                    n = int(r["n"])
                    side = int(dna_side_by_label.get(dna_label, -1))
                    from_x = 209.26 if side < 0 else 390.74
                    target = str(r.get("dna_target", "base"))
                    if target == "phosphate" and dna_label in dna_phosphate_center_by_label:
                        cx, cy = dna_phosphate_center_by_label[dna_label]
                        ax, ay = clip_circle_border(cx, cy, 4.36, from_x, float(cy))
                    elif target == "sugar" and dna_label in dna_sugar_border_by_label:
                        ax, ay = dna_sugar_border_by_label[dna_label]
                    else:
                        bx, by = dna_anchor_by_label.get(dna_label, (x_left_base if side < 0 else x_right_base, float(dna_y_by_label[dna_label])))
                        ax, ay = clip_circle_border(bx, by, 7.0, from_x, float(by))
                    text = f"{protein_label(prot_resname, prot_resid)} {itype} x{n}"
                    label_events.append(
                        {
                            "side": side,
                            "y_target": float(ay),
                            "text": text,
                            "interaction_type": itype,
                            "anchor_x": float(ax),
                            "anchor_y": float(ay),
                            "water": False,
                        }
                    )

        if len(self.water_bridges) > 0:
            for _, row in self.water_bridges.iterrows():
                bridge_type = str(row.get("bridge_type", "WaterBridge")).strip()
                if bridge_type == "Protein-Water-DNA":
                    if pd.isna(row.get("protein_resname")) or pd.isna(row.get("protein_resid")):
                        continue
                    if pd.isna(row.get("dna_resname")) or pd.isna(row.get("dna_resid")):
                        continue
                    prot_resname = str(row.get("protein_resname"))
                    prot_resid = int(row.get("protein_resid"))
                    dna_label = f"{row.get('dna_resname')}{int(row.get('dna_resid'))}"
                    if dna_label not in dna_y_by_label:
                        continue
                    text = protein_label(prot_resname, prot_resid)
                    side = int(dna_side_by_label.get(dna_label, -1))
                    ax, ay = dna_anchor_by_label.get(dna_label, (x_left_base if side < 0 else x_right_base, float(dna_y_by_label[dna_label])))
                    label_events.append(
                        {
                            "side": side,
                            "y_target": float(ay),
                            "text": f"{text} WaterBridge",
                            "interaction_type": "WaterBridge",
                            "anchor_x": float(ax),
                            "anchor_y": float(ay),
                            "water": True,
                        }
                    )

        def _dedupe_events(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
            grouped: Dict[Tuple, Dict[str, object]] = {}
            counts: Dict[Tuple, int] = {}
            for ev in events:
                key = (
                    int(ev.get("side", 0)),
                    str(ev.get("text", "")),
                    str(ev.get("interaction_type", "")),
                    bool(ev.get("water", False)),
                    float(ev.get("anchor_x", 0.0)),
                    float(ev.get("anchor_y", 0.0)),
                )
                if key not in grouped:
                    grouped[key] = dict(ev)
                    counts[key] = 1
                else:
                    counts[key] += 1

            out: List[Dict[str, object]] = []
            for key, ev in grouped.items():
                c = counts.get(key, 1)
                if c > 1:
                    ev["text"] = f"{ev.get('text')} x{c}"
                out.append(ev)
            return out

        label_events = _dedupe_events(label_events)

        left_events = [e for e in label_events if int(e.get("side", -1)) < 0]
        right_events = [e for e in label_events if int(e.get("side", 1)) > 0]

        y_top_labels = max(ys) + 10.0
        y_bottom_labels = 110.0
        min_sep = 7.0
        col_dx = 55.0
        max_cols = 4

        def pack_events_into_columns(
            events: List[Dict[str, object]],
            base_text_x: float,
            base_line_x: float,
            side: int,
        ) -> List[Dict[str, object]]:
            events_sorted = sorted(events, key=lambda ev: float(ev.get("y_target", 0.0)), reverse=True)
            col = 0
            last_y = None
            for ev in events_sorted:
                if col >= max_cols:
                    col = max_cols - 1

                x_shift = (-col_dx * col) if side < 0 else (col_dx * col)
                text_x = base_text_x + x_shift
                line_x = base_line_x + x_shift

                y = float(ev.get("y_target", ligand_y))
                if y > y_top_labels:
                    y = y_top_labels

                if last_y is not None and last_y - y < min_sep:
                    y = last_y - min_sep

                if y < y_bottom_labels:
                    col += 1
                    last_y = None
                    x_shift = (-col_dx * col) if side < 0 else (col_dx * col)
                    text_x = base_text_x + x_shift
                    line_x = base_line_x + x_shift
                    y = y_top_labels

                last_y = y
                ev["y"] = y
                ev["text_x"] = text_x
                ev["line_x"] = line_x
            return events_sorted

        left_events = pack_events_into_columns(left_events, base_text_x=206.42, base_line_x=209.26, side=-1)
        right_events = pack_events_into_columns(right_events, base_text_x=393.58, base_line_x=390.74, side=1)

        for ev in left_events:
            y = float(ev.get("y", float(ev.get("y_target", ligand_y))))
            text = str(ev.get("text", ""))
            text_x = float(ev.get("text_x", 206.42))
            line_x = float(ev.get("line_x", 209.26))
            ps.append(f" {text_x:6.2f} {y:7.2f} moveto")
            ps.append(f"({self._ps_escape(text)}) 7 Ralign")
            ps.append(f"({self._ps_escape(text)}) 7 Print")
            lx, ly = clip_horizontal_from_label_edge(line_x, y, side=-1)
            emit_line(float(lx), float(ly), float(ev.get("anchor_x")), float(ev.get("anchor_y")), str(ev.get("interaction_type")), water=bool(ev.get("water")))

        for ev in right_events:
            y = float(ev.get("y", float(ev.get("y_target", ligand_y))))
            text = str(ev.get("text", ""))
            text_x = float(ev.get("text_x", 393.58))
            line_x = float(ev.get("line_x", 390.74))
            ps.append(f" {text_x:6.2f} {y:7.2f} moveto")
            ps.append(f"({self._ps_escape(text)}) 7 Lalign")
            ps.append(f"({self._ps_escape(text)}) 7 Print")
            lx, ly = clip_horizontal_from_label_edge(line_x, y, side=1)
            emit_line(float(lx), float(ly), float(ev.get("anchor_x")), float(ev.get("anchor_y")), str(ev.get("interaction_type")), water=bool(ev.get("water")))

        ps.append("NucplotSave restore")
        ps.append("showpage")
        ps.append("%%EOF")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(ps) + "\n", encoding="utf-8")
        logger.info(f"Saved Nucplot-like legacy schematic to {output_path}")
        return str(output_path)

    def _collect_dna_residues(self) -> List[ResidueKey]:
        residues: Dict[Tuple[str, int], ResidueKey] = {}

        if "dna_resname" in self.interactions.columns and "dna_resid" in self.interactions.columns:
            for _, row in self.interactions[["dna_resname", "dna_resid"]].dropna().iterrows():
                key = (str(row["dna_resname"]), int(row["dna_resid"]))
                residues[key] = ResidueKey(*key)

        if len(self.water_bridges) > 0 and "dna_resname" in self.water_bridges.columns and "dna_resid" in self.water_bridges.columns:
            for _, row in self.water_bridges[["dna_resname", "dna_resid"]].dropna().iterrows():
                key = (str(row["dna_resname"]), int(row["dna_resid"]))
                residues[key] = ResidueKey(*key)

        return sorted(residues.values(), key=lambda r: r.resid)

    def _collect_protein_residues(self) -> List[ResidueKey]:
        residues: Dict[Tuple[str, int], ResidueKey] = {}

        if "protein_resname" in self.interactions.columns and "protein_resid" in self.interactions.columns:
            for _, row in self.interactions[["protein_resname", "protein_resid"]].dropna().iterrows():
                key = (str(row["protein_resname"]), int(row["protein_resid"]))
                residues[key] = ResidueKey(*key)

        if len(self.water_bridges) > 0 and "protein_resname" in self.water_bridges.columns and "protein_resid" in self.water_bridges.columns:
            for _, row in self.water_bridges[["protein_resname", "protein_resid"]].dropna().iterrows():
                key = (str(row["protein_resname"]), int(row["protein_resid"]))
                residues[key] = ResidueKey(*key)

        return sorted(residues.values(), key=lambda r: (r.resname, r.resid))

    def _protein_side(self, protein_label: str) -> int:
        return -1 if (sum(ord(c) for c in protein_label) % 2 == 0) else 1

    def _interaction_style(self, interaction_type: str) -> Dict:
        itype = (interaction_type or "").strip()
        if itype == "HBond":
            return {"color": "black", "linestyle": "--", "linewidth": 0.9}
        if itype == "VdW":
            return {"color": "0.4", "linestyle": ":", "linewidth": 0.8}
        if itype == "WaterBridge":
            return {"color": "black", "linestyle": "-", "linewidth": 0.8}
        return {"color": "black", "linestyle": "-", "linewidth": 0.8}

    def export(self, output_path: str = "output.ps", title: Optional[str] = None) -> str:
        out_path = Path(output_path)
        ext = out_path.suffix.lower()
        if ext == ".ps":
            return self._export_nucplot_like_ps(out_path, title=title)

        dna_residues = self._collect_dna_residues()
        if len(dna_residues) == 0:
            raise ValueError("No DNA residues found (expected columns: dna_resname, dna_resid)")

        ligand_label = self._infer_ligand_label()

        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "font.family": "Helvetica",
                "font.size": 8,
                "ps.useafm": True,
                "pdf.use14corefonts": True,
            }
        )

        n_dna = len(dna_residues)
        fig_h = max(4.0, 0.35 * n_dna + 1.5)
        fig_w = 8.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        ax.set_axis_off()

        y_positions = {res.label(): float(n_dna - 1 - i) for i, res in enumerate(dna_residues)}
        dna_center = {label: (0.0, y) for label, y in y_positions.items()}

        ligand_y = float(np.median(list(y_positions.values())))

        dna_box_w = 1.6
        dna_box_h = 0.45
        backbone_x_left = -0.95
        backbone_x_right = 0.95
        ax.plot([backbone_x_left, backbone_x_left], [-0.5, n_dna - 0.5], color="black", linewidth=1.0)
        ax.plot([backbone_x_right, backbone_x_right], [-0.5, n_dna - 0.5], color="black", linewidth=1.0)

        for label, (x, y) in dna_center.items():
            rect = Rectangle(
                (x - dna_box_w / 2, y - dna_box_h / 2),
                dna_box_w,
                dna_box_h,
                facecolor="white",
                edgecolor="black",
                linewidth=0.9,
            )
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center", color="black")

        protein_residues = self._collect_protein_residues()
        protein_coords: Dict[str, Tuple[float, float]] = {}
        if len(protein_residues) > 0:
            y_targets: Dict[str, List[float]] = {r.label(): [] for r in protein_residues}

            has_protein_cols = (
                "protein_resname" in self.interactions.columns and "protein_resid" in self.interactions.columns
            )
            has_dna_cols = "dna_resname" in self.interactions.columns and "dna_resid" in self.interactions.columns

            for _, row in self.interactions.iterrows():
                if not has_protein_cols or not has_dna_cols:
                    continue
                if pd.isna(row.get("protein_resname")) or pd.isna(row.get("protein_resid")):
                    continue
                prot_label = f"{row.get('protein_resname')}{int(row.get('protein_resid'))}"

                dna_label = None
                if not pd.isna(row.get("dna_resname")) and not pd.isna(row.get("dna_resid")):
                    dna_label = f"{row.get('dna_resname')}{int(row.get('dna_resid'))}"

                if dna_label and dna_label in y_positions and prot_label in y_targets:
                    y_targets[prot_label].append(y_positions[dna_label])

            if len(self.water_bridges) > 0:
                for _, row in self.water_bridges.iterrows():
                    if pd.isna(row.get("protein_resname")) or pd.isna(row.get("protein_resid")):
                        continue
                    prot_label = f"{row.get('protein_resname')}{int(row.get('protein_resid'))}"

                    dna_label = None
                    if not pd.isna(row.get("dna_resname")) and not pd.isna(row.get("dna_resid")):
                        dna_label = f"{row.get('dna_resname')}{int(row.get('dna_resid'))}"

                    if dna_label and dna_label in y_positions and prot_label in y_targets:
                        y_targets[prot_label].append(y_positions[dna_label])

            prot_y = {}
            for prot_label, ys in y_targets.items():
                if len(ys) == 0:
                    prot_y[prot_label] = ligand_y
                else:
                    prot_y[prot_label] = float(np.mean(ys))

            items = sorted(prot_y.items(), key=lambda kv: kv[1], reverse=True)
            adjusted = {}
            min_sep = 0.28
            last_y_left = None
            last_y_right = None

            for prot_label, y in items:
                side = self._protein_side(prot_label)
                if side < 0:
                    if last_y_left is not None and last_y_left - y < min_sep:
                        y = last_y_left - min_sep
                    last_y_left = y
                else:
                    if last_y_right is not None and last_y_right - y < min_sep:
                        y = last_y_right - min_sep
                    last_y_right = y
                adjusted[prot_label] = y

            for prot_label, y in adjusted.items():
                side = self._protein_side(prot_label)
                x = -4.0 if side < 0 else 4.0
                protein_coords[prot_label] = (x, y)
                ha = "right" if side < 0 else "left"
                ax.text(x, y, prot_label, ha=ha, va="center", color="black")

        if len(self.interactions) > 0:
            for _, row in self.interactions.iterrows():
                pair_type = str(row.get("pair_type", "")).strip()
                interaction_type = str(row.get("interaction_type", "Contact")).strip()

                if pair_type == "Protein-DNA":
                    if pd.isna(row.get("dna_resname")) or pd.isna(row.get("dna_resid")):
                        continue
                    if pd.isna(row.get("protein_resname")) or pd.isna(row.get("protein_resid")):
                        continue
                    dna_label = f"{row.get('dna_resname')}{int(row.get('dna_resid'))}"
                    prot_label = f"{row.get('protein_resname')}{int(row.get('protein_resid'))}"
                    if dna_label not in dna_center or prot_label not in protein_coords:
                        continue
                    x1, y1 = dna_center[dna_label]
                    x2, y2 = protein_coords[prot_label]

                else:
                    continue

                style = self._interaction_style(interaction_type)
                ax.plot([x1, x2], [y1, y2], **style)

        if len(self.water_bridges) > 0:
            for _, row in self.water_bridges.iterrows():
                bridge_type = str(row.get("bridge_type", "WaterBridge")).strip()

                if bridge_type == "Protein-Water-DNA":
                    if pd.isna(row.get("protein_resname")) or pd.isna(row.get("protein_resid")):
                        continue
                    if pd.isna(row.get("dna_resname")) or pd.isna(row.get("dna_resid")):
                        continue
                    prot_label = f"{row.get('protein_resname')}{int(row.get('protein_resid'))}"
                    dna_label = f"{row.get('dna_resname')}{int(row.get('dna_resid'))}"
                    if prot_label not in protein_coords or dna_label not in dna_center:
                        continue
                    x1, y1 = protein_coords[prot_label]
                    x2, y2 = dna_center[dna_label]

                else:
                    continue

                style = self._interaction_style("WaterBridge")
                ax.plot([x1, x2], [y1, y2], **style)
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ax.plot(
                    [xm],
                    [ym],
                    marker="o",
                    markersize=4.0,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    linewidth=0,
                )

        if title:
            ax.text(0.0, n_dna + 0.3, title, ha="center", va="bottom", color="black")

        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-1.0, max(n_dna, ligand_y + 1.0) + 1.0)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        fmt = "pdf" if ext == ".pdf" else "ps"
        fig.savefig(out_path, format=fmt, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved Nucplot-style legacy schematic to {out_path}")
        return str(out_path)


def create_legacy_exporter(
    interactions_df: pd.DataFrame,
    water_bridges_df: Optional[pd.DataFrame] = None,
    residue_summary_df: Optional[pd.DataFrame] = None,
    ligand_label: Optional[str] = None,
) -> NucplotExporter:
    return NucplotExporter(
        interactions_df,
        water_bridges_df=water_bridges_df,
        residue_summary_df=residue_summary_df,
        ligand_label=ligand_label,
    )
