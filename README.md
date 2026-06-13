<<<<<<< HEAD
# Hieroglyph Benchmark: CNN vs RNN vs ViT
### A Comparative Deep Learning Study for Ancient Egyptian Hieroglyph Classification
**Author:** Samruddhi Adhikary | Roll No.: UCE2024006

---

## Dataset Recommendation & How to Get It

### ✅ USE: Fuentes-Ferrer et al. Dataset (GitHub) — RECOMMENDED

**Why NOT the Kaggle link you found:**
The Kaggle dataset (Menna Alaa Raslan) mirrors the Franken dataset (~4,210 images,
171 classes, single historical period — Pyramid of Unas only). This is too small for
ViT to train properly and limits your benchmark's credibility.

**Why the Fuentes-Ferrer GitHub dataset:**
- 13,729 images across 310 Gardiner classes
- Covers Old, Middle, and New Kingdom (nearly 3,000 years)
- Largest public hieroglyph dataset available (as of 2025)
- Directly cited in the paper you found — keeps your citation chain consistent
- Enough data for ViT pretraining fine-tuning to be meaningful

### Step-by-Step Dataset Download

```bash
# Option A — Git clone (recommended, ~1.2 GB)
git clone https://github.com/rfuentesfe/EgyptianHieroglyphicText.git
mv EgyptianHieroglyphicText/dataset ./data/

# Option B — Download ZIP from GitHub
# Go to: https://github.com/rfuentesfe/EgyptianHieroglyphicText
# Click green "Code" button → "Download ZIP"
# Extract the dataset/ folder into ./data/

# Option C — Kaggle CLI (if the repo mirrors to Kaggle)
pip install kaggle
kaggle datasets download -d mennaalaarasslan/egyptian-hieroglyphic-text
unzip egyptian-hieroglyphic-text.zip -d ./data/
```

After download, your data/ folder should look like:
```
data/
  dataset/
    a1/   (image files: .jpg, .png)
    a2/
    a17/
    b1/
    d1/
    ...
    z9/
```

The folder names ARE the Gardiner codes (class labels). This is standard
ImageFolder format — PyTorch reads it automatically.

---

## Project Structure

```
hieroglyph_benchmark/
├── README.md
├── requirements.txt
├── config.py          ← All hyperparameters in one place
├── dataset.py         ← Data loading, augmentation, degradation transforms
├── models/
│   ├── __init__.py
│   ├── glyphnet.py    ← Custom compact CNN (Barucci et al. 2021)
│   ├── cnn_models.py  ← ResNet-50, EfficientNet-B3 wrappers
│   ├── rnn_model.py   ← CNN-LSTM hybrid
│   └── vit_models.py  ← ViT-B/16, DeiT-Small wrappers
├── train.py           ← Training loop with early stopping
├── evaluate.py        ← Metrics, confusion matrix, per-class accuracy
├── degradation.py     ← Robustness benchmark (blur, occlusion, noise)
├── interpretability.py← Grad-CAM, saliency maps, ViT attention rollout
├── ablation.py        ← Data scarcity threshold experiment
├── utils.py           ← Logging, plotting, checkpoint saving
└── main.py            ← Orchestration — run everything from here
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Full Benchmark

```bash
# 1. Train and evaluate all models
python main.py --mode train_all

# 2. Run degradation robustness benchmark
python main.py --mode degradation

# 3. Generate interpretability maps (pick a model)
python main.py --mode interpret --model resnet50

# 4. Run data ablation experiment
python main.py --mode ablation

# 5. Run everything sequentially
python main.py --mode all
```

---

## Citation

If you use this code, please cite:
- Fuentes-Ferrer et al. (2025) for the dataset
- Barucci et al. (2021) for Glyphnet architecture
- This study: Adhikary, S. (2025). A Comparative Deep Learning Study of CNN, RNN,
  and Vision Transformers for Ancient Egyptian Hieroglyph Classification.
=======
