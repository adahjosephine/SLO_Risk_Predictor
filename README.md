# SLO Risk Predictor 

# ESG-Driven Supply Chain Risk in Critical Minerals: Synthesis and an ML Proof-of-Concept

The goal is to create a screening tool (Proof-of-concept) that could estimate ESG supply-chain risk for any mine site on Earth, without a bespoke expert panel, using only open source data. It combines insights from the reference literature sources.

Sarkheil et al.(2026) show *machine learning can turn many correlated ESG indicators into an explainable, transferable risk score*, but their indicators are assembled by expert panels — expensive, slow, and hard to replicate at the hundreds of mine sites that make up a real critical-mineral supply chain. Heydari et al.(2026) show that *open, free, global geospatial datasets can proxy for many of those same indicators* (population exposure, infrastructure access, deforestation, hydrological connectivity) without an expert panel — but they stop at descriptive scoring and don't use ML to learn, generalize, or predict from that data.

Sources: 
-- Sarkheil, H., Salahjou, T. and Hashemi, A. (2026) ‘A robust hybrid machine learning framework for multidimensional ESG assessment in critical mineral supply chains: The case of cobalt mining’, Results in Engineering, 29, Article 109501. Available at: https://doi.org/10.1016/j.rineng.2026.109501

-- Heydari, M., Noskov, A., Cervantes Barron, K., Ciftci, M.M., Andrieu, B., Chabala, R.M., Matokwani, M., Serrenho, A.C. and Cullen, J.M. (2026) Toward responsible mining: Linking ESG strategies with spatial analysis in Zambia's copper mining industry. The Extractive Industries and Society, 27, Article 101908. https://doi.org/10.1016/j.exis.2026.101908


---

## 1. How the two papers complement each other

| | Sarkheil et al. (2026) (cobalt, Iran) | Heydari et al.(2026) (copper, Zambia) |
|---|---|---|
| **Unit of analysis** | Single-commodity, single-country ESG *scoring* | Multi-site *spatial* comparison within one country |
| **Data strategy** | Expert Delphi panel + national field data + proxy data borrowed from DRC (Kolwezi habitat-loss imagery) | Fully open geospatial data (Hansen Global Forest Change, ASTER GDEM, OpenStreetMap/HDX, WorldPop) + one proprietary mine database |
| **Modeling approach** | Hybrid: interpretable linear expert-weighted score as "ground truth" + XGBoost ensemble trained to reproduce it, explained via SHAP/LIME/ICE | Deterministic geospatial overlay + a simple rank-based composite scoring formula (no ML) |
| **What it's strong at** | Explaining *why* ESG risk is high, nonlinear interactions, feature attribution, threshold effects | Showing *where* risk is concentrated, deforestation footprints, water-flow pollution pathways, per-mine social vulnerability |
| **What it's missing** | No spatial/geographic resolution below "site"; no treatment of infrastructure or community exposure as measurable geodata | No predictive or explainable model, scores are hand-built ranks, not learned from data; no mechanism to generalize beyond the 10 mines actually mapped |





---

## 2. Supply chain risks identified, and which ones ML can address

| Risk | Source | Addressable by ML? |
|---|---|---|
| Hydrogeological stress / water depletion at mine sites | Sarkheil (top SHAP feature) | Yes, regression/classification on hydrology + climate features |
| Ecosystem fragmentation & deforestation-driven biodiversity loss | Both papers | **Yes,remote-sensing time series (Hansen GFC) is exactly the kind of structured, global, labeled-enough data ML handles well** |
| Water pollution pathways from mine runoff into protected areas / rivers | Heydari (flow-accumulation mapping) | Yes,  but needs hydrological simulation, not pure ML; ML can rank priority monitoring points |
| Governance failure / corruption / weak enforcement | Sarkheil | Partially, proxies (inspector density, distance to regulatory infrastructure) can be modeled, but ground truth is scarce |
| **Community / social-license-to-operate (SLO) disruption** — protest, resettlement conflict, work stoppages | Both (Sarkheil et. al., 2026): "social unrest"; Heydari et. al., (2026): social vulnerability scoring) | **Yes, Heydari's open infrastructure/population features are a ready-made, reproducible, low-cost feature set** |
| Geographic supply concentration risk (>70% of cobalt from DRC) | Sarkheil | Not really an ML problem, it's a structural/statistical exposure metric |
| Reputational/regulatory sanction risk from ESG non-compliance | Both (implied) | Yes, if disclosure/news data is available, future extension |

I picked **community / social-license-to-operate (SLO) disruption risk** for the MVP because:
1. It is a genuine supply-chain risk; SLO breakdowns are one of the most common real-world causes of mine stoppages and export disruptions for cobalt and copper alike.
2. It's the risk for which Heydari et al. (2026) already provide a **real, published, per-site numeric dataset** (Table 8), so the MVP can be trained on actual data rather than invented numbers.
3. It directly operationalizes Sarkheil et al.'s call to make ESG assessment "transferable" to other cobalt/copper geographies (DRC, Philippines, Australia) using only data that is available *without* convening a new expert panel.

---

## 3. MVP: Open-Data SLO Disruption Risk Screener

**Data used** Heydari et al. (2026), Table 8: per-mine building count, road length, distance-to-amenities, distance-to-education, distance-to-health-facility scores (rank-normalized 0–9), and population, for all 10 major Zambian copper mines. These sub-scores are themselves derived entirely from open sources: OpenStreetMap (via HDX) for buildings/roads/amenities, and WorldPop (2020) for population.

**Method:**
- Derived a 3-tier risk label (High/Medium/Low) from the paper's own composite score.
- Trained a `RandomForestRegressor` and `RandomForestClassifier` with **Leave-One-Out Cross-Validation**. This is the same validation protocol Sarkheil et al.(2026) use for their small ESG dataset, chosen for the same reason (n=10 is too small for a train/test split).
- Computed permutation feature importance.
- Ran a proof-of-concept "screen a new, unaudited site" scenario using only the same five open-data-derivable inputs, applied to an illustrative DRC cobalt-mining-area profile — directly testing the cross-commodity, cross-country transferability that both papers argue for.

## Interpreting the results

<img width="712" height="494" alt="image" src="https://github.com/user-attachments/assets/c5eaef62-66c7-404f-a894-594a88e0ee6e" />

**Figure 1.** validation metrics.

**Validation metrics.** LOOCV Mean Absolute Error(MAE) (1.07 on a 0–9 scale) and Spearman ρ (0.65, p=0.043)
indicate the model can moderately recover the *relative ordering* of mines by risk.
Classification accuracy into High/Medium/Low tiers (30%) falls *below* the 40%
majority-class baseline — with only 10 labeled sites, discrete risk tiering is not
statistically supportable yet, even though rank-ordering holds up reasonably well.
**Takeaway: use the continuous score, not the tier, until more sites are added.**

<img width="1464" height="443" alt="image" src="https://github.com/user-attachments/assets/b15ac37f-4918-45cd-851b-840c36f48e5e" />

**Figure 2.** permutation feature importance plot.


**Feature importance.** Buildings and Education dominate; Population, Health, and
Roads contribute close to nothing. This is *not* evidence that building density and
school access drive real community disruption risk. The prediction target (composite
score) is itself constructed as a rank-average of these same six inputs, so this chart
shows which of the scoring formula's own attributes happened to carry the most
separating signal across these 10 specific mines.
The model depends most on buildings and education to best distinguish the low-infrastructure/high-risk mines (Lumwana, Trident, Mwambashi) from the high-infrastructure/low-risk ones (Mufulira, Kansanshi). Population/Health/Roads didn't add unique separating information once those two attributes were already used.

<img width="453" height="677" alt="image" src="https://github.com/user-attachments/assets/98b3d40a-2c45-4ba8-955d-5c925ece9a8b" />

**Figure 3.** Hypothetical DRC site risk screening using the Zambia-trained SLO model

**Transferability screen.** The hypothetical-site panel demonstrates the intended
*use case*; screening an unaudited site using only open-data-derivable inputs,
without an expert panel, and determine the predicted risk. 

With only 10 mines in the dataset, the model can moderately rank sites by relative risk (Spearman ρ = 0.65, p = 0.043), but it cannot yet reliably classify sites into risk tiers (30% accuracy, below the 40% baseline from always predicting the most common class).

This means a “High risk” prediction for the hypothetical DRC site should be interpreted as a directional signal rather than a confident category assignment. The uncertainty applies across all tier predictions.

The challenge comes from fixed category thresholds (High ≤ 3.4, Medium ≤ 4.4, Low > 4.4), which create artificial boundaries. Two sites with very similar scores can be placed into different categories simply because they fall on opposite sides of a cutoff.

The continuous risk score avoids this limitation by showing the actual level of risk rather than forcing sites into broad groups. For small datasets, continuous scores should therefore be the primary decision tool, with risk tiers introduced only when a larger dataset supports more reliable thresholds.

## So what — the actual contribution

This POC does not discover a new risk driver, and it does not validate that spatial/
Infrastructure openness predicts real disruption events. What it does show:

1. **A working, reusable pipeline**: open-data-style features feed into the LOOCV model that feeds into
   per-site risk score, and then new-site screening. This mirrors the validation discipline
   (LOOCV) the source literature (Sarkheil et al.(2026))  uses for small ESG datasets.
2. **A concrete methodological trap**, worth documenting for anyone extending this
   work: constructing an ML target from the same indicators used as features produces
   internally consistent but circular results. Any future version must label mines
   against a real outcome (strikes, protests, permit suspensions) rather than a
   hand-built composite score.
3. A quantitative demonstration that, with small ESG datasets, models can identify relative risk differences (Spearman ρ = 0.65) before they can reliably assign "High/Medium/Low" risk categories (30% accuracy vs. a 40% majority-class baseline). This shows why categorical risk products should be treated cautiously at low sample sizes, and why continuous risk scores are often a more reliable first output.
   
**Bottom line:** This is a template for how Sarkheil et al.'s (2026) ML methodology and
Heydari et al.'s (2026) open-data feature layer could be combined into a predictive
screening tool; proven feasible as a pipeline, not yet proven valid as a predictor.
Closing that gap requires real disruption-event labels and more sites, both enabled
by the same open-data infrastructure this POC already uses.

**Limitation:** 10 mine sites are nowhere near enough data to train a trustworthy classifier. This is a proof of *pipeline*, not a proof of *accuracy*. The real value demonstrated here is that the feature-engineering approach (rank-normalized, population-adjusted, all derived from free global datasets) is mechanically ready to scale. Both papers converge on exactly the next step needed: Sarkheil et al. (2026) explicitly call for retraining on region-specific data as the framework expands to the DRC, Philippines, and Australia; Heydari et al.'s open-data pipeline is precisely how that retraining data could be gathered mine by mine, across all of those geographies. Scaling from n=10 to n=100+ mines (globally, via the same open-data pipeline) is the natural next step and would very likely change both the classification accuracy and the feature-importance ranking meaningfully.

**Path to production**
```
mine coordinates (public company disclosures / S&P-type registries)
        │
        ▼
Overpass API (OpenStreetMap) → buildings, roads, amenities, schools, clinics within 10km LIA buffer
WorldPop raster              → population within the same buffer
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

**AI Use**: AI was used to synthesize the research papers and aided the proof-of-concept coding
