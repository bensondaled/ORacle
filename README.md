# ORacle: Intraoperative Vital Sign Prediction and Hypotension Onset Detection

Deep learning system for real-time prediction of vital signs during surgery, with focus on mean arterial pressure (MAP) forecasting and hypotension onset detection.

Code for Ghanem, Deverett, et al.

---

## System Requirements

### Software Dependencies

- Python >= 3.9
- PyTorch >= 2.0.0
- numpy >= 1.21.0
- pandas >= 1.4.0
- pyarrow >= 10.0.0
- tables >= 3.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- pyyaml >= 6.0

### Tested Configuration

| OS | Python | PyTorch | GPU |
|----|--------|---------|-----|
| Ubuntu | 3.10 | 2.9.1 | NVIDIA L40S |

### Hardware

- NVIDIA GPU with CUDA support recommended
- No non-standard hardware required

---

## Installation Guide

### Clone the Repository

```bash
git clone https://github.com/bensondaled/ORacle.git
cd ORacle
```

### Install with Poetry (Recommended)

```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

### Alternative: Install with pip

```bash
python -m venv oracle_env
source oracle_env/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Demo

### Demo Data

Simulated dataset included at `demo/demo_data.feather`:
- 10 surgical cases
- 1,089 total timepoints
- Vital signs: MAP, SBP, DBP, HR, SpO2, EtCO2
- 15 medication channels
- 4 anesthetic gas channels
- 5 hypotension onset events

### Run Demo

```bash
python demo/run_demo.py
```

With trained checkpoint:

```bash
python demo/run_demo.py --checkpoint path/to/model.pt
```

### Expected Output

```
======================================================================
ORacle Demo - Intraoperative Vital Sign Prediction
======================================================================

Using device: cuda
GPU: NVIDIA L40S

Loading config from demo/config.yaml...
Loading demo data from demo/demo_data.feather...
Loaded 10 surgical cases with 1,089 total timepoints

Creating dataset...
Created 899 samples

Initializing model...
Model parameters: 1,150,181

----------------------------------------------------------------------
Running Inference
----------------------------------------------------------------------

Case DEMO_001:
  Current vitals: MAP=74 mmHg, HR=74 bpm, SpO2=98%
  Predicted MAP (next 15 min): [-0.2, -0.2, -0.2, -0.2, -0.2, ...]
  Hypotension risk: 3.6%

Case DEMO_002:
  Current vitals: MAP=77 mmHg, HR=59 bpm, SpO2=98%
  Predicted MAP (next 15 min): [-0.2, -0.2, -0.3, -0.3, -0.3, ...]
  Hypotension risk: 3.4%

...

======================================================================
Demo completed successfully!
======================================================================
```

Note: Output shown is with randomly initialized weights. Use a trained checkpoint for meaningful predictions.

---

## Instructions for Use

### Data Format

Input data should be a pandas DataFrame (`.feather` or `.csv`) with:

**Required Columns:**
- `mpog_case_id` - Unique case identifier
- `time_since_start` - Minutes elapsed

**Vital Signs:**
- `phys_bp_sys_non_invasive` - Systolic BP (mmHg)
- `phys_bp_dias_non_invasive` - Diastolic BP (mmHg)
- `phys_bp_mean_non_invasive` - Mean arterial pressure (mmHg)
- `phys_spo2_pulse_rate` - Heart rate (bpm)
- `phys_spo2_%` - Oxygen saturation (%)
- `phys_end_tidal_co2_(mmhg)` - End-tidal CO2 (mmHg)

**Medications:**
- `meds_propofol`, `meds_fentanyl`, `meds_phenylephrine`, `meds_ketamine`
- `meds_norepinephrine`, `meds_ephedrine`, `meds_esmolol`, `meds_vasopressin`
- `meds_remifentanil`, `meds_dexmedetomidine`, `meds_glycopyrrolate`
- `meds_labetalol`, `meds_hydromorphone`, `meds_etomidate`, `meds_epinephrine`

**Anesthetic Gases:**
- `phys_sevoflurane_exp_%`, `phys_isoflurane_exp_%`
- `phys_desflurane_exp_%`, `phys_nitrous_exp_%`

**Demographics:**
- `age`, `sex`, `weight`

### Running Inference

See `demo/run_demo.py` for a complete example of loading data, initializing the model, and running inference.

---

## Model Output

- **MAP Prediction**: 15-step ahead trajectory of mean arterial pressure
- **Hypotension Risk**: Probability of hypotension onset (MAP < 65 mmHg)

---
```
