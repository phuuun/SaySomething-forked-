# SaySomething — Running with Real Models (No Gemini)

## What was changed

| File | What changed |
|------|-------------|
| `my nlp models/tf_idf_results (1).ipynb` | Added **Cell 11b** — saves vectorizers + models to `saved_models/` |
| `my nlp models/lstm_results.ipynb` | Added **Cell 11b** — saves Keras model + tokenizer to `saved_models/` |
| `my nlp models/bert_finetuned_jigsaw (1).ipynb` | Added **Cell 11b** — saves tokenizer to `saved_models/` |
| `my nlp models/roberta_finetuned_jigsaw.ipynb` | Added **Cell 11b** — saves tokenizer to `saved_models/` |
| `server_models.py` | **NEW** — Python Flask server that loads and serves all 4 real models |
| `src/utils/mockInference.ts` | Tiny update — better error messages if server is down |

`server.ts` (the Gemini version) is **untouched**. You can still use it for Vercel.

---

## Step 1: Run all 4 notebooks on Colab and save the models

> Do this ONCE. After this you have the model files forever.

### For each notebook:

**1. Open Google Colab** → `File → Upload notebook` → upload the notebook

**2. Enable GPU** → `Runtime → Change runtime type → T4 GPU`

**3. Upload the CSV files** → click the folder icon on the left sidebar in Colab, drag in `train.csv` and `test.csv`

**4. Run all cells** → `Runtime → Run all`

**5. After training, run the new Cell 11b** (the SAVE cell) — it creates a `saved_models/` folder in Colab

**6. Download `saved_models/` to your computer**:
   ```python
   # Run this in a new Colab cell to zip and download
   import shutil
   shutil.make_archive('saved_models', 'zip', 'saved_models')
   
   from google.colab import files
   files.download('saved_models.zip')
   ```

**7. Unzip `saved_models.zip` into your project root** (same folder as `package.json`)

After all 4 notebooks, your project should look like:
```
SaySomething/
├── saved_models/
│   ├── tfidf_word_vectorizer.joblib
│   ├── tfidf_char_vectorizer.joblib
│   ├── tfidf_logreg_models.joblib
│   ├── lstm_model.h5
│   ├── lstm_tokenizer.joblib
│   ├── lstm_config.json
│   ├── distilbert_tokenizer/          ← this is a folder
│   ├── distilbert_weights.pt
│   ├── roberta_tokenizer/             ← this is a folder
│   └── roberta_weights.pt
├── server_models.py                   ← new Flask server
├── server.ts                          ← old Gemini server (untouched)
├── package.json
└── ...
```

---

## Step 2: Install Python dependencies

Open a terminal in your project folder:

```bash
pip install flask flask-cors scikit-learn joblib torch transformers tensorflow numpy
```

This is a big install (~2-3GB). Do it once.

---

## Step 3: Run locally with real models

You need **two terminals** open at the same time:

**Terminal 1 — Python Flask (models):**
```bash
python server_models.py
```

You'll see:
```
✅ Phase 1 (TF-IDF + LogReg) loaded
✅ Phase 2 (BiLSTM + BiGRU) loaded
✅ Phase 3 (DistilBERT) loaded with trained weights
✅ Phase 4 (RoBERTa) loaded with trained weights
📍 Server running at http://localhost:3000
```

**Terminal 2 — Vite frontend:**
```bash
npm run dev
```

Open http://localhost:5173 → type a comment → click Analyze → **real model predictions!**

---

## Step 4: Verify it's working

Open http://localhost:3000/api/health in your browser.

You should see:
```json
{
  "status": "ok",
  "models": {
    "TF-IDF + LogReg": true,
    "LSTM": true,
    "DistilBERT": true,
    "RoBERTa": true
  }
}
```

If any model shows `false`, check the terminal — it will say exactly which file is missing.

---

## Troubleshooting

### "Model not loaded" error in the app
→ Check the terminal running `server_models.py`. It will say exactly which file is missing.
→ Make sure you ran all 4 notebooks AND ran the save cell (Cell 11b) in each.

### Port 3000 already in use
→ Change the port in `server_models.py`: `app.run(port=3001)`
→ Also update `vite.config.ts` proxy to point to the new port (see below).

### Vite can't reach the Flask server (CORS / proxy issues)
Add this to your `vite.config.ts`:
```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:3000'
    }
  }
})
```

### LSTM is very slow on first prediction
→ Normal — Keras loads lazily. Second prediction will be fast.

### Out of memory loading all 4 models
→ Comment out the models you don't need in `server_models.py` (around lines 25-75)
→ Or close other apps to free RAM

---

## For Vercel (production)

Keep using `server.ts` with Gemini for the Vercel deployment — that's fine.
The Flask server (`server_models.py`) is only for running locally.

If you eventually want to deploy the real models, look into:
- **Railway.app** or **Render.com** — can host Python servers cheaply
- **Hugging Face Spaces** — free hosting specifically for ML models
