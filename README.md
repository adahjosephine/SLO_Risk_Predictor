# SLO Risk Predictor 

# ESG-Driven Supply Chain Risk in Critical Minerals: Synthesis and an ML Proof-of-Concept

**The complementarity is direct**: Sarkheil et al.(2026) show *machine learning can turn many correlated ESG indicators into an explainable, transferable risk score*, but their indicators are assembled by expert panels — expensive, slow, and hard to replicate at the hundreds of mine sites that make up a real critical-mineral supply chain. Heydari et al.(2026) show that *open, free, global geospatial datasets can proxy for many of those same indicators* (population exposure, infrastructure access, deforestation, hydrological connectivity) without an expert panel — but they stop at descriptive scoring and don't use ML to learn, generalize, or predict from that data.

The goal is to create a screening tool (Proof-of-concept) that could estimate ESG supply-chain risk for any mine site on Earth, without a bespoke expert panel, using only open source data.

Sources: Sarkheil, Salahjou & Hashemi (2026), *Results in Engineering* — the `esg_assessor` hybrid ML framework for cobalt (Iran); Heydari et al. (2026), *The Extractive Industries and Society* — geospatial ESG analysis of Zambia's copper mining sector.


---

## 1. How the two papers complement each other

| | Sarkheil et al. (2026) (cobalt, Iran) | Heydari et al.(2026) (copper, Zambia) |
|---|---|---|
| **Unit of analysis** | Single-commodity, single-country ESG *scoring* | Multi-site *spatial* comparison within one country |
| **Data strategy** | Expert Delphi panel + national field data + proxy data borrowed from DRC (Kolwezi habitat-loss imagery) | Fully open geospatial data (Hansen Global Forest Change, ASTER GDEM, OpenStreetMap/HDX, WorldPop) + one proprietary mine database |
| **Modeling approach** | Hybrid: interpretable linear expert-weighted score as "ground truth" + XGBoost ensemble trained to reproduce it, explained via SHAP/LIME/ICE | Deterministic geospatial overlay + a simple rank-based composite scoring formula (no ML) |
| **What it's strong at** | Explaining *why* ESG risk is high — nonlinear interactions, feature attribution, threshold effects | Showing *where* risk is concentrated — deforestation footprints, water-flow pollution pathways, per-mine social vulnerability |
| **What it's missing** | No spatial/geographic resolution below "site"; no treatment of infrastructure or community exposure as measurable geodata | No predictive or explainable model — scores are hand-built ranks, not learned from data; no mechanism to generalize beyond the 10 mines actually mapped |





---

## 2. Supply chain risks identified, and which ones ML can address

| Risk | Source | Addressable by ML? |
|---|---|---|
| Hydrogeological stress / water depletion at mine sites | Sarkheil (top SHAP feature) | Yes — regression/classification on hydrology + climate features |
| Ecosystem fragmentation & deforestation-driven biodiversity loss | Both papers | **Yes — remote-sensing time series (Hansen GFC) is exactly the kind of structured, global, labeled-enough data ML handles well** |
| Water pollution pathways from mine runoff into protected areas / rivers | Heydari (flow-accumulation mapping) | Yes — but needs hydrological simulation, not pure ML; ML can rank priority monitoring points |
| Governance failure / corruption / weak enforcement | Sarkheil | Partially — proxies (inspector density, distance to regulatory infrastructure) can be modeled, but ground truth is scarce |
| **Community / social-license-to-operate (SLO) disruption** — protest, resettlement conflict, work stoppages | Both (Sarkheil: "social unrest"; Heydari: social vulnerability scoring) | **Yes — Heydari's open infrastructure/population features are a ready-made, reproducible, low-cost feature set** |
| Geographic supply concentration risk (>70% of cobalt from DRC) | Sarkheil | Not really an ML problem — it's a structural/statistical exposure metric |
| Reputational/regulatory sanction risk from ESG non-compliance | Both (implied) | Yes, if disclosure/news data is available — future extension |

I picked **community / social-license-to-operate (SLO) disruption risk** for the MVP because:
1. It is a genuine supply-chain risk — SLO breakdowns are one of the most common real-world causes of mine stoppages and export disruptions for cobalt and copper alike.
2. It's the risk for which Heydari et al. (2026) already provide a **real, published, per-site numeric dataset** (Table 8) — so the MVP can be trained on actual data rather than invented numbers.
3. It directly operationalizes Sarkheil et al.'s call to make ESG assessment "transferable" to other cobalt/copper geographies (DRC, Philippines, Australia) using only data that is available *without* convening a new expert panel.

---

## 3. MVP: Open-Data SLO Disruption Risk Screener

**Data used (real, open, cited):** Heydari et al. (2026), Table 8 — per-mine building count, road length, distance-to-amenities, distance-to-education, distance-to-health-facility scores (rank-normalized 0–9), and population, for all 10 major Zambian copper mines. These sub-scores are themselves derived entirely from open sources: OpenStreetMap (via HDX) for buildings/roads/amenities, and WorldPop (2020) for population.

**Method:**
- Derived a 3-tier risk label (High/Medium/Low) from the paper's own composite score.
- Trained a `RandomForestRegressor` and `RandomForestClassifier` with **Leave-One-Out Cross-Validation** — the same validation protocol Sarkheil et al. use for their small ESG dataset, chosen for the same reason (n=10 is too small for a train/test split).
- Computed permutation feature importance.
- Ran a proof-of-concept "screen a new, unaudited site" scenario using only the same five open-data-derivable inputs, applied to an illustrative DRC cobalt-mining-area profile — directly testing the cross-commodity, cross-country transferability that both papers argue for.

**Results:**
- LOOCV **rank correlation** between predicted and actual composite score: **Spearman ρ = 0.65 (p = 0.043)** — the model recovers *relative* ordering of mines moderately well.
- LOOCV **3-class tier accuracy: 30%**, which is *below* the 40% majority-class baseline. With only 10 labeled sites, discrete High/Medium/Low classification is not yet reliable.
- Feature importance (see `feature_importance.png`) ranks **amenities and health-facility distance** and **population** as the strongest drivers in this small sample, broadly consistent with Heydari et al.'s narrative that infrastructure access — not raw population size — differentiates newer, better-planned mines (e.g., Trident) from legacy Copperbelt sites (e.g., Mufulira).
- On the illustrative new DRC site (sparse mapped infrastructure, moderate population), the model predicts **"High risk"** with 84% class probability — a sensible directional result given deliberately low infrastructure inputs, though it should be read as a demo of the pipeline, not a real assessment of any actual site.

**limitation:** ten mines is nowhere near enough data to train a trustworthy classifier — this is a proof of *pipeline*, not a proof of *accuracy*. The real value demonstrated here is that the feature-engineering approach (rank-normalized, population-adjusted, all derived from free global datasets) is mechanically ready to scale. Both papers converge on exactly the next step needed: Sarkheil et al. (2026) explicitly call for retraining on region-specific data as the framework expands to the DRC, Philippines, and Australia; Heydari et al.'s open-data pipeline is precisely how that retraining data could be gathered mine by mine, across all of those geographies. Scaling from n=10 to n=100+ mines (globally, via the same open-data pipeline) is the natural next step and would very likely change both the classification accuracy and the feature-importance ranking meaningfully.

**Path to production**
```
mine coordinates (public company disclosures / S&P-type registries)
        │
        ▼
Overpass API (OpenStreetMap) → buildings, roads, amenities, schools, clinics within 10km LIA buffer
WorldPop raster              → population within same buffer
Hansen Global Forest Change  → forest-loss % within buffer, by year
ASTER GDEM + hydrology       → elevation, flow-accumulation to protected areas
        │
        ▼
   feature engineering (rank-normalize, population-adjust, log-transform)
        │
        ▼
   esg_assessor-style hybrid model (transparent linear baseline + ML ensemble)
        │
        ▼
   per-site SLO / environmental risk tier + SHAP-based explanation
```



