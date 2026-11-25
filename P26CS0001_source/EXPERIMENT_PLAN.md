# Experiment plan — P26CS0001

## Controls and hypotheses

One saved stratified split: 45,000/5,000 using seed 260001. No training on validation
or test data. Official test data remain inaccessible to development entry points.
Training channel statistics only. All hyperparameter grids and budgets are in
configs/experiment.json. Test evaluation follows a content-hashed freeze manifest.

Q1(a)/Q2(a): hypothesise that a locality-based architectural prior contributes to
the CNN advantage, competing with parameter count and unequal training opportunity.
Compare MLP 3072–112–112–10 with CNN conv(3,32), conv(32,64), conv(64,128),
2x2 max pool after each, flatten–128–10. All convolutions 3x3 padded, ReLU.
Approximately matched parameter counts (within 1%), equal examples, batch size,
AdamW, LR schedule, epochs, loss, and selection. FLOPs and wall time are not
matched; identical optimisation settings need not be optimal for each architecture.
Two paired initialisation/order seeds (17,29), same fixed data split.
This comparison alone is associative. Q2(b)'s interaction supplies a stronger
intervention on locality, but cannot eliminate every optimisation explanation.

Q2(b): one saved permutation of 1,024 positions, shared RGB mapping for all splits.
Prediction: CNN degradation exceeds MLP degradation. For the permuted MLP transport
initial first-layer columns by the same permutation; identical minibatch ordering
makes it a functionally equivalent optimisation problem up to floating point
reduction. This controls representational capacity and much of the optimisation
explanation. CNN uses identical initial parameters across clean/permuted inputs.
Report numerical rather than exact equality and remaining convolutive training
interactions. Save invertibility and value-preservation checks.

Q2(c): retrain both networks (seed 17) on fixed luminance grayscale, repeat to RGB.
Competing explanations: CNN advantage chiefly depends on colour cues versus local
intensity geometry. Prediction: CNN retains a substantial advantage if geometry
matters; removal of colour can reduce both accuracies. Same mapping at train,
validation, test; same budget, no extra tuning. This intervention loses information
and cannot isolate every colour/spatial interaction. Report NLL, multiclass Brier,
15-bin ECE, confidence on mistakes and confusion matrices for all conditions.

Q1(b): stratified 5,000-example training subset, 40% selected without replacement;
replacement target sampled uniformly from the other nine classes (always wrong),
fixed forever. Save subset, mask, clean/noisy targets. MLP 3072–512–512–10, 150
epochs, no regularisation, no augmentation. Log clean validation accuracy, clean
subset accuracy, corrupt-target agreement and original-label agreement every epoch.
Operational onset: first of two consecutive epochs with corrupt-target accuracy
>20%; saturation: >80%; useful structure: clean validation >20% twice. These
thresholds are descriptive, not a theoretical definition. Do not assume structure
precedes onset. If corruption does not fit, report insufficient demonstration.

Q3(a): fixed conv encoder 3–16–32–64, stride 2, 4x4x64 flattened features;
linear bottlenecks 16/64/256, symmetric decoder, sigmoid pixels. Only the two
bottleneck-adjacent linear layers change. MSE over all pixel components; same
seed and budget. Choose lowest validation MSE (ties smaller dimension).
Q3(b): train a separate DAE with the selected dimension, additive N(0,0.2²) noise
clipped to [0,1], fresh training noise per epoch; fixed separate validation/test
noise seeds. Compare identical corrupted examples, selected clean AE, DAE, using
seed-17 clean CNN checkpoint frozen for every condition. Reconstructions remain
[0,1] then receive the CNN's training normalisation. Prediction: MSE improves;
classifier recovery is uncertain because smoothing may destroy useful detail.
Q3(c): frozen clean AEs and DAE, clean inputs for representation/probe comparison;
standardisation fit on training features; multinomial logistic regression with
C in {0.01,0.1,1,10}, validation selection. Report test MSE for **all** these
representations as a separately mandated comparison, not a selection change.
No guarantee that reconstruction and semantic rankings agree.

Q4 representation: seed-17 clean CNN's 128-dimensional penultimate ReLU, no
fine tuning, training-fitted standardisation shared SVM/clustering. Fix this choice
now, independent of validation performance of any competing representation.
Q4(a): one stratified 6,000-example subset fixed before selection, all 18 SVMs use
it; C={.001,.01,.1,1,10,100}, linear/RBF/poly, gamma=scale, poly degree3 coef0=1.
Support fraction denominator 6,000. Full 5,000 validation set. Selection by accuracy;
ties smaller C then listed kernel order. Test the winner once. Hypotheses: low C
underfits; high C may increase train/validation gap; support fraction need not
monotonically track generalisation. No results assumed.
Q4(b): after predictions frozen, rank unordered class pairs by summed off-diagonal
test confusion counts (ties lexicographic), choose three. For each inspect training
support vectors belonging to those classes and test images of those classes with
smallest absolute **one-versus-one functional decision scores**. scikit-learn/libsvm
binary convention must be verified: for multiclass OVO, positive favours first
class; report scores, true/predicted labels, support status and pair errors. Functional
scores, not kernel-space normalised distances; third-class decisions can override.
Q4(c): KMeans full training features, k={8,10,12,15,20}; three seeds, n_init=10.
Select k by mean validation silhouette (fixed 2,000-example sample, no labels).
Stability: ARI between seed assignments (cluster labels only), report without using
semantic labels for selection. Fit final training model at selected k with seed83.
Predict test clusters before loading test labels. Then ARI/NMI/contingency and
class/pair mixing versus SVM pair confusion. Label-derived summaries evaluate only.
Clustering uses all 45,000 training examples, unlike SVM's fixed 6,000; state this.

## Randomness, selection and limits

Seeds: split260001, classifier17/29, memorisation43, AE59, SVM71,
clustering83/97/109; offset seeds documented in code for permutation, corruption,
noise and sampling. Python/NumPy/PyTorch seeded; deterministic CUDA operations,
TF32 disabled, single training process, CPU threads capped. Fixed test corruption.
Checkpoint selection lowest validation CE/MSE; training resumes with optimiser,
scheduler and RNG states. No test-based model updates. Predictions and full metrics
saved for analysis; report tables generated programmatically. No significance
claim from two seeds. Hardware and runtime are logged. Compact models keep the comparison within
a practical training budget. Numerical reproducibility depends on the software
and hardware used.

Classifier batch order uses a dedicated per-epoch generator seeded by
`run_seed * 100000 + epoch`, independently of architecture initialisation.
Selection uses a four-thread BLAS/OpenMP cap. Autoencoder runs log
reconstruction loss and leave classification accuracy unset.
