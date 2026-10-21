#!/usr/bin/env python3
"""
Second corpus: USAID funding announcements on grants.gov (forward-looking).

DEC documents mostly describe awards already running, often written by
implementers, and lag policy by a year or more. Funding announcements (RFAs,
APSs, NOFOs) are written by USAID and dated when the agency decided what to
solicit. This script uses the grants.gov XML extract (title + synopsis text
and metadata for every opportunity; not the full solicitation PDFs) to ask
the same question on that corpus:

  * how many USAID opportunities were posted per year, and what share were health
  * which health subjects and policy phrases the health announcements mention
  * Biden vs Trump differences: opportunity-level permutation test (BH-FDR) and a
    trend-adjusted linear probability model (linear time trend + COVID period)

Unit = one opportunity (deduplicated by OpportunityID, latest version kept).

    python scripts/08_grants_gov.py --xml GrantsDBExtract20250215v2.xml --out-dir outputs/grants_gov
"""
import argparse
import html
import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import _common  # noqa: F401
import config as cfg
from src.analysis.prevalence import bh_fdr, permutation_test
from src.preprocessing import administration_for

NS = "{http://apply.grants.gov/system/OpportunityDetail-V1.0}"

SUBJECTS = {
    "hiv_aids": r"\bhiv\b|\baids\b|pepfar|antiretroviral",
    "malaria": r"malaria|\bpmi\b|insecticide|bed ?nets?\b",
    "tuberculosis": r"tuberculosis|\btb\b",
    "family_planning_rh": r"family planning|contracepti|reproductive health",
    "maternal_child": r"maternal|newborn|neonatal|child health|immuni[sz]ation|vaccin",
    "nutrition": r"nutrition|stunting|malnutrition|food security",
    "wash": r"\bwater\b|sanitation|hygiene|\bwash\b",
    "health_security": r"global health security|pandemic|outbreak|zoonotic|one health|antimicrobial|\bghsa\b",
    "covid": r"covid|sars-cov-2|coronavirus",
    "health_systems": r"health system|supply chain|health workforce|health financing|human resources for health",
    "ntd": r"neglected tropical|\bntds?\b",
}


def _text(el, tag):
    return (el.findtext(NS + tag) or "").strip()


def parse(xml_path: Path) -> pd.DataFrame:
    rows = []
    for _, el in ET.iterparse(xml_path):
        if el.tag == NS + "OpportunitySynopsisDetail_1_0":
            if _text(el, "AgencyCode").startswith("USAID"):
                cats = [c.text for c in el.findall(NS + "CategoryOfFundingActivity") if c.text]
                rows.append({
                    "opportunity_id": _text(el, "OpportunityID"),
                    "number": _text(el, "OpportunityNumber"),
                    "title": _text(el, "OpportunityTitle"),
                    "agency_code": _text(el, "AgencyCode"),
                    "agency_name": _text(el, "AgencyName"),
                    "post_date": _text(el, "PostDate"),
                    "categories": "|".join(cats),
                    "instrument": _text(el, "FundingInstrumentType"),
                    "cfda": _text(el, "CFDANumbers"),
                    "total_funding": _text(el, "EstimatedTotalProgramFunding"),
                    "award_ceiling": _text(el, "AwardCeiling"),
                    "version": _text(el, "Version"),
                    "description": _text(el, "Description"),
                })
            el.clear()
        elif el.tag.endswith("Detail_1_0"):  # forecasts etc.: free memory (never clear child fields early)
            el.clear()
    df = pd.DataFrame(rows)
    df["version_n"] = pd.to_numeric(df["version"].str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
    df = df.sort_values("version_n").drop_duplicates("opportunity_id", keep="last")
    d = pd.to_datetime(df["post_date"], format="%m%d%Y", errors="coerce")
    df["date"] = d
    df["year"] = d.dt.year
    df["administration"] = [administration_for(x.date()) if pd.notna(x) else "Unknown" for x in d]
    clean = lambda s: " ".join(re.sub(r"<[^>]+>", " ", html.unescape(s)).split())
    df["text"] = (df["title"].map(clean) + ". " + df["description"].map(clean)).str.lower()
    df["health_category"] = df["categories"].str.split("|").map(lambda c: "HL" in c)
    for col in ["total_funding", "award_ceiling"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)
    return df


def trend_lpm(y: np.ndarray, df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Biden jump at the inauguration net of a linear trend and a COVID-period indicator (HC3)."""
    import statsmodels.api as sm

    t = df["date"].dt.year + (df["date"].dt.dayofyear - 1) / 365.25 - 2021.05
    X = pd.DataFrame({
        "const": 1.0,
        "biden": (df["administration"] == "Biden").astype(float),
        "year_c": t,
        "covid": (df["date"] >= pd.Timestamp("2020-03-11")).astype(float),
    })
    if np.nanstd(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    lo, hi = fit.conf_int().loc["biden"]
    return fit.params["biden"], lo, hi, fit.pvalues["biden"]


def compare(df: pd.DataFrame, families: dict[str, str], label: str) -> pd.DataFrame:
    m = pd.DataFrame({k: df["text"].str.contains(p, regex=True) for k, p in families.items()}).astype(float)
    is_b = (df["administration"] == "Biden").to_numpy()
    obs, p = permutation_test(m.to_numpy(), is_b)
    rows = []
    for j, k in enumerate(m.columns):
        jump, lo, hi, pt = trend_lpm(m[k].to_numpy(), df)
        rows.append({"family": k, "set": label, "trump_share": m[k][~is_b].mean(), "biden_share": m[k][is_b].mean(),
                     "diff": obs[j], "p_perm": p[j], "jump_at_inauguration": jump, "ci_low": lo, "ci_high": hi,
                     "p_trend": pt})
    out = pd.DataFrame(rows)
    out["q_perm"] = bh_fdr(out["p_perm"].to_numpy())
    out["q_trend"] = bh_fdr(out["p_trend"].fillna(1).to_numpy())
    by_year = m.assign(year=df["year"].to_numpy()).groupby("year").mean()
    by_year["n"] = df.groupby("year").size()
    return out, by_year


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=cfg.OUTPUT_DIR / "grants_gov")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    allopps = parse(args.xml)
    allopps.drop(columns=["description"]).to_csv(out / "usaid_opportunities.csv", index=False)
    df = allopps[allopps["administration"].isin(["Trump", "Biden"])].reset_index(drop=True)

    yearly = df.groupby("year").agg(n_opportunities=("opportunity_id", "size"),
                                    health_share=("health_category", "mean"),
                                    n_health=("health_category", "sum"))
    yearly.to_csv(out / "opportunities_by_year.csv")
    print(yearly.round(3).to_string())

    health = df[df["health_category"]].reset_index(drop=True)
    print(f"\nUSAID opportunities in window: {len(df)} | health: {len(health)} "
          f"(Trump {int((health.administration == 'Trump').sum())}, Biden {int((health.administration == 'Biden').sum())})")
    print(f"Median synopsis length (health): {health['text'].str.split().str.len().median():.0f} words")

    # health share of all USAID opportunities, as a 0/1 outcome
    hs = compare(df.assign(text=np.where(df["health_category"], "HEALTHFLAG", "")), {"health_category": "HEALTHFLAG"}, "all")[0]
    subj, subj_year = compare(health, SUBJECTS, "health subjects")
    pol, pol_year = compare(health, cfg.MENTION_TERMS, "policy terms")
    res = pd.concat([hs, subj, pol], ignore_index=True)
    res.to_csv(out / "grants_gov_tests.csv", index=False)
    subj_year.to_csv(out / "subjects_by_year.csv")
    pol_year.to_csv(out / "policy_terms_by_year.csv")

    fund = health.copy()
    for k, p in SUBJECTS.items():
        fund[k] = fund["text"].str.contains(p, regex=True)
    cov = fund["total_funding"].notna().mean()
    print(f"Health opportunities with an estimated total funding figure: {cov:.0%}")
    funding = pd.DataFrame({k: fund[fund[k]].groupby("administration")["total_funding"].agg(["count", "median", "sum"]).stack()
                            for k in SUBJECTS}).T
    funding.to_csv(out / "funding_by_subject.csv")

    pd.set_option("display.width", 200)
    print("\n", res[["set", "family", "trump_share", "biden_share", "diff", "q_perm", "jump_at_inauguration",
                    "ci_low", "ci_high", "q_trend"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
