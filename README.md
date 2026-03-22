# Reg-InSaAF

**Regulatory-Indic Semi-automated Audit Framework**

Reg-InSaAF is a statutory reasoning dataset and evaluation framework that audits
whether Large Language Models exhibit *Sovereign Privilege* — a systematic bias
toward applying legal exemptions to state-affiliated actors while over-flagging
identical content from state-critical individuals — under India's Information
Technology (Intermediary Guidelines and Digital Media Ethics Code) Amendment
Rules, 2026 and the Digital Personal Data Protection Act, 2023.

This work extends the InSaAF framework (Tripathi et al., 2024) from demographic
bias axes (religion, caste, region, gender) to institutional-deference axes
grounded in India's emerging AI content moderation law.

---

## Key Contributions

- **Dataset** — 2,500 counterfactually paired statutory reasoning prompts across
  6 bias axes, 3 legal clauses, 30 names, 30 posting scenarios, and 10 platforms.
  Every entry has a direct counterfactual pair differing only in the actor's
  relationship to state power.

- **Bias axes** — Institutional Role (T), Political Alignment (P), Legal History
  (L), Media Status (M), Economic-Institutional Utility (E), Temporal Pressure (W).

- **Metrics** — Sovereign Privilege Score (SPS), Temporal Amplification Factor
  (TAF), Deference Gradient, extending InSaAF's LSSβ framework.

- **Legal grounding** — Rule 2(1)(wa) IT Amendment Rules 2026, Section 7(c)
  DPDP Act 2023, Section 79 IT Act 2000.

---

## Dataset Structure
```
data/
  reg_bsr_withid.csv           Full 2500 entries, both prompt columns
  reg_withid_train.json        1500 entries, Alpaca format, actor intact
  reg_withid_val.json          500 entries, Alpaca format, actor intact
  reg_withoutid_train.json     1500 entries, actor replaced with X. (Veil of Ignorance)
  reg_withoutid_val.json       500 entries, actor replaced with X.
  reg_test_withid.csv          500 entries, held-out test set
```

Each entry follows the Alpaca instruction-tuning format:
```json
{
  "instruction": "Consider yourself as my law advisor...",
  "input": "Law Description: Rule 2(1)(wa) IT Amendment Rules 2026\n...\n Situation: ...",
  "output": "Yes"
}
```

The `reg_bsr_withid.csv` reference file additionally includes:
- `Prompt_wo_reg_id` — actor description replaced with `X.` for Veil of Ignorance experiments
- `Axis`, `Condition`, `CF_Pair_ID` — metadata for bias measurement

---

## Experimental Pipeline

Mirrors InSaAF's three-variant structure:

| Experiment | Training data | Test data | Purpose |
|------------|--------------|-----------|---------|
| Zero-shot baseline | — | reg_test_withid.csv | Measure innate bias |
| LLM_with_reg_id | reg_withid_train.json | reg_test_withid.csv | Fine-tune with identity |
| LLM_without_reg_id | reg_withoutid_train.json | reg_test_withid.csv | Veil of Ignorance |

---

## Quickstart

**Generate dataset:**
```bash
python scripts/generate_reginsaaf.py --total 2500 --min-per-condition 30 --seed 42
```

**Evaluate zero-shot (Groq API):**
```bash
export API_KEY=your_groq_key

python scripts/evaluate_reginsaaf.py \
  --dataset data/reg_test_withid.csv \
  --backend groq \
  --model llama-3.3-70b-versatile \
  --beta 1.0 \
  --out results/70b/eval
```

**Merge distributed results:**
```bash
python scripts/merge_results.py
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **RFS** | Relative Fairness Score — proportion of counterfactual pairs where the model gives consistent answers (InSaAF eq. 3) |
| **F1** | Accuracy against legal ground truth |
| **LSSβ** | Legal Safety Score — β-weighted harmonic mean of RFS and F1 (InSaAF eq. 4) |
| **SPS** | Sovereign Privilege Score — P(Takedown\|state-critical) − P(Takedown\|state-affiliated) |
| **TAF** | Temporal Amplification Factor — whether the 180-minute deadline amplifies SPS |

---

## Results

| Model | RFS | F1 | LSS (β=1) | SPS | Abstain |
|-------|-----|----|-----------|-----|---------|
| LLaMA-3.3-70B (zero-shot) | — | — | — | — | — |
| LLaMA-3.1-8B (zero-shot) | — | — | — | — | — |

*Results pending full evaluation run.*

---

## Legal Context

India's IT Amendment Rules 2026 introduced a 180-minute mandatory takedown
window for Synthetically Generated Information (SGI) under Rule 2(1)(wa). This
creates structural pressure on automated moderation systems to prioritise
compliance speed over nuanced legal reasoning. Simultaneously, the DPDP Act 2023
Section 7(c) grants the State broad exemptions for data processing in the interest
of sovereignty — creating an asymmetric legal landscape that this dataset is
designed to probe.

---

## Citation
```bibtex
@article{tripathi2024insaaf,
  title     = {InSaAF: Incorporating Safety through Accuracy and Fairness |
               Are LLMs ready for the Indian Legal Domain?},
  author    = {Tripathi, Yogesh and Donakanti, Raghav and Girhepuje, Sahil and
               Kavathekar, Ishan and Vedula, Bhaskara Hanuma and
               Krishnan, Gokul S and Goyal, Shreya and Goel, Anmol and
               Ravindran, Balaraman and Kumaraguru, Ponnurangam},
  journal   = {arXiv preprint arXiv:2402.10567},
  year      = {2024}
}
```

---

## Acknowledgements

Built on the InSaAF framework by Tripathi et al. (2024). Dataset uses the
InDeepFake corpus (pending access) for multimodal grounding.
Evaluated using the Groq inference API.

---

## License

Dataset: CC BY 4.0  
Code: MIT
```

---

**Topics to add on GitHub** (the tags under the repo name):
```
india  legal-nlp  fairness  llm-evaluation  statutory-reasoning
content-moderation  deepfake  bias-detection  it-rules-2026  dpdp-act
