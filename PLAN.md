# ArtSense AI — Development Plan

## Context
ArtSense AI is a Machine Learning course project that identifies painting artists using EfficientNet-B3 and provides LLM-powered educational explanations. Originally scoped as a Kaggle Notebook only, the goal is now to also build a Gradio demo app, with a potential Flutter + FastAPI mobile app as future work. The project is starting from scratch.

**Dataset:** Best Artworks of All Time (Kaggle, ~8,000 images, 50 artists, with `artists.csv` metadata)
**LLM:** Claude (Anthropic API)
**UI:** Gradio demo first, Flutter + FastAPI later (out of scope now)

---

## Project Structure

```
artsense/
├── notebooks/
│   └── train_artsense.ipynb      # Kaggle training notebook
├── app/
│   ├── app.py                    # Gradio UI entry point
│   ├── classifier.py             # Pipeline A: image classification
│   ├── qa_pipeline.py            # Pipeline B: conversational Q&A
│   ├── llm_agent.py              # Claude API wrapper
│   └── fuzzy_match.py            # RapidFuzz artist name matching
├── model/
│   └── efficientnet_b3_artsense.pth   # Downloaded after Kaggle training
├── data/
│   └── artists.csv               # Downloaded from Kaggle dataset
├── images/                       # Artist image folders (downloaded from Kaggle)
│   ├── Vincent_van_Gogh/
│   ├── Claude_Monet/
│   └── ...
├── .env                          # ANTHROPIC_API_KEY (not committed)
├── .env.example                  # Template for .env
└── requirements.txt
```

---

## Phase 1: Kaggle Training Notebook (`notebooks/train_artsense.ipynb`)

**Goal:** Fine-tune EfficientNet-B3 on 50 artists and export weights.

Steps inside the notebook:
1. Add dataset: `ikarus777/best-artworks-of-all-time`
2. Parse folder structure → build label list (50 classes)
3. Train/val split (80/20 stratified)
4. Transforms: resize to 300×300, augment (flip, rotate, color jitter), normalize (ImageNet stats)
5. `WeightedRandomSampler` to handle class imbalance
6. Load `efficientnet_b3` from `torchvision.models` with pretrained ImageNet weights
7. Replace final classifier head: `nn.Linear(1536, 50)`
8. Training strategy:
   - Phase 1: freeze backbone, train head for 5 epochs
   - Phase 2: unfreeze all, train for 10–15 epochs
   - Optimizer: AdamW + CosineAnnealingLR
9. Save artifacts:
   - `efficientnet_b3_artsense.pth` — model state dict
   - `class_indices.json` — `{artist_name: class_index}` mapping

**After training:** Download both files from Kaggle and place them in the `model/` folder.

---

## Phase 2: Core App Modules

### `app/fuzzy_match.py`
- Load `artists.csv` at startup
- `match_artist(name) -> dict | None`: fuzzy-match against 50 artist names using RapidFuzz, return CSV row if score ≥ 80

### `app/llm_agent.py`
- `extract_artist_name(question, artist_list) -> str`: send question + full artist list to Claude Haiku → returns best-matching name
- `explain_artwork(artist_row, user_context) -> str`: send artist metadata + user context to Claude Sonnet → returns beginner-friendly explanation

### `app/classifier.py`
- Load EfficientNet-B3 + `class_indices.json` at startup
- `classify_painting(image) -> dict`: preprocess → inference → softmax → return top-3 predictions + confidence
- If top-1 confidence < **80%** → flag as `below_threshold`

### `app/qa_pipeline.py`
- `run_image_pipeline(image) -> dict`: classify → confidence gate → metadata lookup → LLM explanation → sample images
- `run_text_pipeline(question) -> dict`: LLM name extraction → fuzzy match → LLM explanation → sample images

---

## Phase 3: Gradio App (`app/app.py`)

Two-tab layout using `gr.Blocks`:

**Tab 1 — Identify Artwork (Pipeline A)**
| Component | Purpose |
|-----------|---------|
| `gr.Image` | Upload a painting |
| `gr.Textbox` | Shows identified artist + confidence |
| `gr.Label` | Top-3 confidence bar |
| `gr.Textbox` | LLM explanation |
| `gr.Gallery` | 3–5 sample artworks from artist's folder |

**Tab 2 — Ask About an Artist (Pipeline B)**
| Component | Purpose |
|-----------|---------|
| `gr.Textbox` | User types a question |
| `gr.Textbox` | Resolved artist name |
| `gr.Textbox` | LLM explanation |
| `gr.Gallery` | 3–5 sample artworks |

Both tabs show a clear decline message when the system cannot identify or find the artist.

**Run locally:**
```bash
python -m app.app
```

---

## Suggested Improvements Over Original Proposal

1. **Grad-CAM heatmap**: Overlay a visual heatmap on the uploaded painting to show which regions the model focused on. Educational and already referenced in the proposal (`pytorch-grad-cam`).
2. **Two-tier LLM usage**: Use `claude-haiku-4-5` for fast/cheap name extraction, `claude-sonnet-4-6` for high-quality explanations.
3. **Visual confidence bar**: Use Gradio's `gr.Label` to show top-3 predictions as a bar chart instead of plain text.

---

## Setup Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Download dataset from Kaggle (requires kaggle CLI)
kaggle datasets download ikarus777/best-artworks-of-all-time
unzip best-artworks-of-all-time.zip
# Place artists.csv in data/ and image folders in images/

# 4. After Kaggle training, place model files in model/
#    - model/efficientnet_b3_artsense.pth
#    - model/class_indices.json

# 5. Run the app
python -m app.app
```

---

## Verification Checklist

- [ ] Upload a known painting (e.g., Starry Night) → correct artist at >80% confidence
- [ ] Upload a photo or AI-generated image → graceful decline message
- [ ] Ask "Who painted the Sunflowers?" → resolves to Vincent van Gogh with explanation
- [ ] Ask "Tell me about Claud Net" → fuzzy matches to Claude Monet
- [ ] Ask about an artist outside the dataset → "not in collection" message
- [ ] Sample gallery images appear for every successful identification

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | EfficientNet-B3 model |
| `anthropic` | Claude API (Haiku + Sonnet) |
| `gradio` | Web demo UI |
| `rapidfuzz` | Fuzzy artist name matching |
| `pandas` | artists.csv metadata |
| `pillow` | Image loading/preprocessing |
| `python-dotenv` | Load `.env` API key |
| `grad-cam` | Grad-CAM heatmap visualisation |

---

## Future Work (Out of Scope Now)

- **Flutter + FastAPI mobile app**: Serve the model via FastAPI, call from a Flutter frontend
- **Multi-turn chatbot**: Pass conversation history to Claude for follow-up questions
- **Hugging Face Spaces deployment**: Host the Gradio app publicly for free
