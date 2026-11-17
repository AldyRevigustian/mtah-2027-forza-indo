# FORZA INDO: More than a Hack 2027, Phase 1

Ball possession and dead ball for every frame of FC Barcelona vs Athletic Club, from the automatic tracking. This
repository regenerates our submission file, `FORZA INDO.csv`, from the competition data bundle.
`technical_document.md` (and its PDF, `FORZA INDO.pdf`) explains the method and the choices behind it.

## Install

Python 3.10.6.

```bash
python -m pip install -r requirements.txt
```

## Data

The match data is not in this repository; it is not redistributable.

- Put the competition bundle under `data/` as its `DATA.md` describes (`data/tactical/...`), or point `MTAH_DATA_DIR`
  at it.
- Only to retrain: `possession/download_external.ipynb` (see below) downloads the seven DFL / IDSSE matches (Bassek et al.,
  2025, CC BY 4.0, [doi:10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177)) to
  `data/external/dfl/` and Metrica sample games 1-3
  ([github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)) to
  `data/external/metrica/Sample_Game_N/`, about 2.8 GB, and checks every file. `MTAH_EXTERNAL_DIR` can point elsewhere.

## Build the file

We committed the trained model (`possession/models/possession_v1/`), so you can build the file without training. Run
these commands in the environment where you installed the requirements (the notebook kernel is that `python`):

```bash
cd possession
python -m nbconvert --to notebook --execute run_decoder.ipynb --output run.ipynb
```

or open `possession/run_decoder.ipynb` and run all cells. The first run takes about 10 minutes because the notebook
parses and cleans the ATD, then caches it in `possession/artifacts/`. It writes `output/FORZA INDO.csv`, validates it
and, as a check only, scores it on the published 5 minutes of ground truth. Expected sha256:

`dc023a10c259f5c629bbaa7a4ac4f1ba851dc630898b1d26f36c2013b54ba733`

## Retrain and choose the decoder (optional)

Run these notebooks from `possession/` in this order, like `run_decoder.ipynb` (open and run all cells, or
`python -m nbconvert --to notebook --execute <notebook> --output <notebook>`):

| Step | Notebook | Time |
|---|---|---|
| Download the DFL / IDSSE and Metrica matches (2.8 GB, checksums verified) | `download_external.ipynb` | depends on the connection |
| Train the CatBoost model with `run_modeling.ipynb`, unchanged, and save the validation-match probabilities | `train_model.ipynb` | ~30 min, 16-thread CPU |
| Choose the decoder setting | `choose_decoder.ipynb` | ~2 min |

`train_model.ipynb` does nothing while the committed model and `possession/artifacts/possession_v1_val_probs.npz` exist; delete
both to train from scratch. `choose_decoder.ipynb` scores the decoder grid on the two external validation matches
only and writes `possession/decoder_choice.json` (read by `run_decoder.ipynb`) and `possession/decoder_tuning.json`
(every setting compared). Training and every tuned setting use external data only.

## Layout

| Path | Content |
|---|---|
| `possession/helper_*.ipynb` | definitions: preprocessing and ball repair, spatial features, CatBoost classifier, external data and degradation, HMM decoder, reporting |
| `possession/run_modeling.ipynb` | trains the classifier on 7 DFL + 3 Metrica matches broken like the target |
| `possession/run_decoder.ipynb` | loads the model, decodes BAR-ATH, writes and validates the file (the decoded bundle goes to `possession/artifacts/models/`) |
| `possession/download_external.ipynb`, `possession/train_model.ipynb`, `possession/choose_decoder.ipynb` | the optional steps above |
| `possession/models/possession_v1/` | the trained model (CatBoost, feature columns, temperature, decoder priors) |
| `docs/` | the two figures shown in `technical_document.md` |
| `evaluation/`, `games.py` | the competition's scorer, validator, public 5-minute ground truth and asset paths, unchanged |

## Licence

The competition files (`evaluation/`, `games.py` and the public 5-minute ground truth) are under their MIT licence,
reproduced in `LICENSE`.
