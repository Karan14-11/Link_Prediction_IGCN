# Link_Prediction_IGCN


The real value of the DynamicGNN approach (especially SVD) isn't speed here — it's accuracy stability. SVD achieves 0.9341 AUC (vs 0.9069 for retrain) with the lowest variance (±0.006). It maintains knowledge from prior snapshots while adapting, whereas retrain starts fresh each time and doesn't converge well with limited epochs.

Bottom line: SVD Selective from the DynamicGNN repo gives +2.7% better AUC and 3× more stable predictions than standard GCN retraining, at the cost of ~1.75× more time per snapshot on GPU. The accuracy gain is the main takeaway, not speed.