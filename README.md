# 16-Age-Verification-via-Efficientnetv2-s
16 years old Age Verification via Efficientnetv2-s

## Goal: 

The challenge-age sweep with leak-rate/adult-friction trade-off is directly aligned with real regulation (UK Ofcom/ICO age-assurance guidance). Academic papers report MAE — they don't answer "how many under-16s get through vs. how many adults are annoyed?" This is the answer a regulator actually wants.

Age-assurance system for the UK under-16 boundary — predict age from a face photo, then determine whether a challenge-age buffer is needed to keep underage leaks ≤ 1%.

## What we've built

- train.py	

Dataset-aware training, age-weighted L1 loss, balanced sampling, AMP, best-checkpoint, --exclude-csv, per-age-bin logging
- evaluate.py	

Binary boundary metrics, challenge-age sweep, per-dataset breakdown, per-image error tracing (--worst, --leakers)
- extract_suspicious.py	

Copies misclassified images to review/ for manual label audit
- data.py	

Subject-disjoint split (no identity leakage), image-extension filter, path tracking

## Evaluation 

| | v1 (ResNet18, L1) | 	v2 (ResNet18, weighted) | 	v3 (EffNetV2-S, leak-aware)| 
| --- | --- |--- | --- | 
| MAE	| 4.65	| 4.62	| 4.50 ↓| 
| Under-16 leak	| 6.8%	| 5.6%	| 5.1% ↓| 
| FG-NET leak	| 13.5%	| 9.0%	| 8.1% ↓| 
| UTKFace leak	| 4.9%	| 4.6%	| 4.2% ↓| 
| ≤1% leak target| 	Age 26, 32% friction | 	Age 26, 32% friction | 	Age 25, 22% friction| 
| Binary acc @ 16	|  97.2%	| 97.9%	| 97.9%| 
