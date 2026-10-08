# Natural Language Processing — BERT & BART

An end-to-end deep-learning NLP research project built around BERT and BART with PyTorch and Hugging Face tooling.

## Tasks

- **SST:** 5-class sentiment classification with BERT
- **QQP:** paraphrase detection with paired-sentence self-attention
- **STS:** semantic textual similarity regression
- **ETPC detection:** multilabel paraphrase-type detection with BART
- **ETPC generation:** controlled paraphrase generation with BART

## Main research ideas

The BERT experiments compare Siamese sentence encoding with joint paired-sentence attention, regularization, label smoothing, learning-rate scheduling, richer classification heads, gradient clipping, and task-specific loss design.

The BART detection experiments address class imbalance with per-label weighting and tune individual decision thresholds using Matthews Correlation Coefficient. The generation experiments compare vanilla and diverse beam-search decoding and test type-aware input augmentation.

## Final development results

| Task | Main metric | Final dev result |
|---|---:|---:|
| SST | Accuracy | 0.542 |
| QQP | Accuracy | 0.880 |
| STS | Pearson correlation | 0.866 |
| ETPC type detection | MCC | 0.292 |
| ETPC generation | Penalized BLEU | 24.430 |

## Repository structure

```text
.
├── bert.py
├── base_bert.py
├── config.py
├── tokenizer.py
├── utils.py
├── optimizer.py
├── datasets.py
├── evaluation.py
├── multitask_classifier.py
├── bart_detection.py
├── bart_generation.py
├── paraphrase_type_generation_helpers.py
├── runlogger.py
├── setup.sh
├── setup_gwdg.sh
├── run_train.sh
├── requirements.txt
├── data/
└── docs/
```

## Setup

```bash
./setup.sh
conda activate dnlp
```

For the Göttingen/GWDG environment:

```bash
./setup_gwdg.sh
conda activate dnlp
```

## Example runs

```bash
python multitask_classifier.py --option finetune --task sst --use_gpu --local_files_only --lr 1e-5 --label_smoothing 0.05 --hidden_dropout_prob 0.1

python multitask_classifier.py --option finetune --task sts --use_gpu --local_files_only --hidden_dropout_prob 0.2 --weight_decay 0.05 --lr 1e-5

python multitask_classifier.py --option finetune --task qqp --use_gpu --local_files_only --hidden_dropout_prob 0.3 --epochs 3 --lr 1e-5

python bart_detection.py --use_gpu
python bart_generation.py --use_gpu
```

## Data

The original SST, QQP, STS and ETPC coursework CSVs are intentionally not committed. See `data/README.md` for the expected filenames.

## Contributors

Heere, Rahman, Halim, Imran, Mehmood

## References

- Vaswani et al., *Attention Is All You Need*.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*.
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*.
- Boughorbel, Jarray & El-Anbari, *Optimal classifier for imbalanced data using Matthews Correlation Coefficient metric*.
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*.
- Vijayakumar et al., *Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models*.
