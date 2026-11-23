"""
Library sanity check: Formula + Precursor_type must reproduce PrecursorMz.

On the bundled EMBL-MCF negative library this finds 3 bad entries out of 302
(Hyodeoxycholic acid and Glyceraldehyde are [2M-H]- ions labelled [M-H]-;
Asparagine is the [M-H]- ion labelled [M-H-H2O]-).
It cannot see a wrong SMILES (N-Acetylglutamine's is pimelic acid) -- that
needs RDKit and is a manual fix in the .msp.
"""
from __future__ import annotations
import re
from collections import Counter

MONO = {"H":1.00782503207,"C":12.0,"N":14.0030740048,"O":15.99491461956,"P":30.97376163,
        "S":31.97207100,"Cl":34.96885268,"Br":78.9183376,"F":18.99840322,"I":126.904473,
        "Na":22.9897692809,"K":38.96370668,"Si":27.9769265325,"Se":79.9165213,"B":11.0093054}
ELECTRON = 0.00054858
PROTON = 1.00727646688

def parse_formula(formula: str) -> dict[str,int]:
    """Element counts of a molecular formula such as "C6H12O6"; ValueError for an element not in MONO."""
    counts: Counter = Counter()
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula.strip()):
        if el not in MONO:
            raise ValueError(f"unsupported element {el!r} in {formula!r}")
        counts[el] += int(n) if n else 1
    return dict(counts)

def neutral_mass(formula: str) -> float:
    """Monoisotopic neutral mass of a formula [Da]."""
    return sum(MONO[e]*n for e, n in parse_formula(formula).items())

# adduct -> (multiplier of M, additive mass, charge)
ADDUCTS = {
    "[M-H]-":      (1, -PROTON, 1),
    "[M-H-H2O]-":  (1, -PROTON - 18.010565, 1),
    "[M-H2O-H]-":  (1, -PROTON - 18.010565, 1),
    "[2M-H]-":     (2, -PROTON, 1),
    "[M+Cl]-":     (1, MONO["Cl"] + ELECTRON, 1),
    "[M+CH3COO]-": (1, 59.013851, 1),       # acetate anion C2H3O2-
    "[M+FA-H]-":   (1, 46.005479 - PROTON, 1),
    "[M-2H]2-":    (1, -2*PROTON, 2),
}

def expected_mz(formula: str, adduct: str) -> float:
    """m/z a formula should have as the given adduct (ADDUCTS table: multiplier, mass shift, charge)."""
    mult, add, z = ADDUCTS[adduct.strip()]
    return (mult*neutral_mass(formula) + add) / z

def check_entries(entries: list[dict], ppm_tol: float = 20.0) -> list[dict]:
    """entries: dicts with name, formula, adduct, precursor_mz. Returns problems."""
    problems = []
    for e in entries:
        adduct = (e.get("adduct") or "").strip() or None
        if not e.get("formula"):
            problems.append({**e, "problem": "no formula"}); continue
        if adduct is None:
            problems.append({**e, "problem": "no adduct"}); continue
        if adduct not in ADDUCTS:
            problems.append({**e, "problem": f"unknown adduct {adduct}"}); continue
        try:
            exp = expected_mz(e["formula"], adduct)
        except ValueError as exc:
            problems.append({**e, "problem": str(exc)}); continue
        ppm = (e["precursor_mz"] - exp) / exp * 1e6
        if abs(ppm) > ppm_tol:
            # does a different adduct explain it?
            alt = [a for a in ADDUCTS
                   if a != adduct and abs((e["precursor_mz"]-expected_mz(e["formula"], a))/e["precursor_mz"]*1e6) <= ppm_tol]
            problems.append({**e, "problem": f"{ppm:+.0f} ppm vs {adduct}", "explained_by": alt})
    return problems


def apply_library_qc(library: list, ppm_tol: float = 20.0) -> tuple[list, dict]:
    """Returns (cleaned_library, report).

    * entry consistent with its stated adduct            -> kept
    * inconsistent, exactly one other adduct explains it -> adduct corrected
    * inconsistent, no/ambiguous explanation             -> excluded
    * no formula, no adduct or an adduct not in ADDUCTS  -> kept, counted as unchecked
    """
    from dataclasses import replace
    cleaned, corrected, excluded = [], [], []
    n_unchecked = 0
    for ref in library:
        adduct = (ref.adduct or "").strip()
        if not getattr(ref, "formula", None) or adduct not in ADDUCTS:
            n_unchecked += 1
            cleaned.append(ref)
            continue
        probs = check_entries([{"name": ref.compound_name, "formula": ref.formula,
                                "adduct": adduct, "precursor_mz": ref.precursor_mz}], ppm_tol)
        if not probs:
            cleaned.append(ref)
        elif len(probs[0].get("explained_by", [])) == 1:
            new = probs[0]["explained_by"][0]
            corrected.append({"name": ref.compound_name, "from": adduct, "to": new})
            cleaned.append(replace(ref, adduct=new))
        else:
            excluded.append({"name": ref.compound_name, "adduct": adduct, "problem": probs[0]["problem"]})
    return cleaned, {"n_input": len(library), "n_unchecked": n_unchecked,
                     "corrected": corrected, "excluded": excluded}
