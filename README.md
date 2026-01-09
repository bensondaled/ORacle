# ORacle: Intaoperative Physiologic Trajectory Forecasting

Deep learning system for physiologic measurements forecasting during surgery.

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

# Install PyTorch with CUDAa
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

### Run Demo

```bash
python demo/run_demo.py
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
Loading checkpoint from demo/demo_checkpoint.pt...
Checkpoint loaded successfully!
Model parameters: 1,150,504

======================================================================
Demo completed successfully!
======================================================================
```

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

- **Physiologic Predictions**: 15-step ahead trajectory of 6 physiologic measurements (SBP, DBP, MAP, HR, SpO2, EtCO2)
