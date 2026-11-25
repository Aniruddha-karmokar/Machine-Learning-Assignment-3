"""Generate the editable CVPR manuscript directly from frozen saved measurements."""
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,CFG
R=ROOT/'artifacts/results'; T=R/'test'; P=ROOT/'report'
def j(path): return json.loads(Path(path).read_text())
def pct(x): return f'{100*float(x):.2f}'
def num(x,n=4): return f'{float(x):.{n}f}'
def esc(x):
    return str(x).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(headers,rows,caption,label,width=None):
    body='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    return r'\begin{table}[!ht]\centering\small\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+r'}\toprule'+'\n'+' & '.join(headers)+r' \\ \midrule'+'\n'+body+'\n'+r'\bottomrule\end{tabular}\caption{'+caption+'}\\label{'+label+'}\\end{table}\n'
def fig(name,caption,wide=True,width=None):
    env='figure*' if wide else 'figure'; w=width or (r'\textwidth' if wide else r'\linewidth')
    return '\\begin{'+env+'}[!ht]\n\\centering\\includegraphics[width='+w+']{../figures/'+name+'.pdf}\n\\caption{'+caption+'}\\label{fig:'+name+'}\n\\end{'+env+'}\n'
def page(): return '\n'

def main():
    P.mkdir(exist_ok=True)
    name_path = P/'student_name.txt'
    if not name_path.is_file() or not name_path.read_text().strip():
        raise RuntimeError('Enter the student name in report/student_name.txt.')
    name = name_path.read_text().strip()
    commit_path=P/'source_commit.txt'
    if not commit_path.exists(): raise RuntimeError('Commit reviewed experiment code/results, then write report/source_commit.txt before report generation.')
    commit=commit_path.read_text().strip()
    subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT,check=True)
    url_path=P/'repository_url.txt'; url=url_path.read_text().strip() if url_path.exists() else ''
    if url and not url.startswith('https://github.com/'): raise ValueError('Expected actual GitHub repository URL')
    repo= r'\url{'+url+'}' if url else r'\textbf{Repository: local only; GitHub publication pending.}'
    cls=pd.read_csv(T/'classifier_table.csv'); summary=pd.read_csv(R/'classifier_summary.csv')
    metrics=j(T/'classifiers.json'); mem=j(R/'memorisation_analysis.json'); interaction=j(R/'paired_interactions.json')
    ae=pd.read_csv(T/'autoencoders.csv'); sel=j(R/'ae_selection.json'); den=j(T/'denoising.json')
    svm=j(T/'svm.json'); svm_sel=j(R/'svm_selection.json'); sweep=pd.read_csv(R/'svm_sweep.csv')
    clustering=j(T/'clustering.json'); clusterchoice=j(R/'clustering_choice.json'); clusters=pd.read_csv(R/'clustering_summary.csv')
    pairs=pd.read_csv(T/'difficult_pairs.csv'); setup=j(R/'setup.json')
    selected_dim=sel['dim']; names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    def acc(kind,condition,seed=17): return float(cls[(cls.model==kind)&(cls.condition==condition)&(cls.seed==seed)].accuracy.iloc[0])
    clean_gap=acc('cnn','clean')-acc('mlp','clean'); gray_gap=acc('cnn','grayscale')-acc('mlp','grayscale')
    def epoch(x): return 'not reached' if x is None else str(int(x))
    source_hash_line=r'\\{\footnotesize Experiment code/results commit: \texttt{'+commit+r'}}'
    s=r'''\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[pagenumbers]{cvpr}
\usepackage{hyperref}
% Overleaf defaults to pdfLaTeX; fontspec is only for XeLaTeX/LuaLaTeX.
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage{tgtermes}
  \usepackage{courier}
\else
  \usepackage{fontspec}
  \setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
  \setmonofont{lmmono10-regular.otf}
\fi
\usepackage{array}
\hypersetup{colorlinks=true,urlcolor=blue,citecolor=blue,linkcolor=blue}
\setlength{\emergencystretch}{1em}
\title{Capacity, Locality, Reconstruction and Semantic Geometry on CIFAR10}
\author{'''+esc(name)+r'''\\Roll number: P26CS0001\\
'''+repo+source_hash_line+r'''}
\begin{document}
\maketitle
\begin{abstract}
We test whether parameter count alone explains the CNN--MLP gap, whether noisy
labels are learned after useful structure, and whether reconstruction quality
tracks class information. One fixed stratified split and an explicit pre-test
selection freeze connect supervised, unsupervised and kernel experiments.
'''
    s+=f'The seed-17 clean CNN and MLP achieve {pct(acc("cnn","clean"))}\\% and {pct(acc("mlp","clean"))}\\% test accuracy, respectively. '
    s+=f'Their accuracy losses under a fixed spatial permutation are {pct(interaction[0]["cnn_drop"])} and {pct(interaction[0]["mlp_drop"])} percentage points. '
    s+=f'The selected SVM reaches {pct(svm["accuracy"])}\\%; clustering the same representation yields ARI {num(clustering["ARI"])} and NMI {num(clustering["NMI"])}. '
    s+=r'''All reported outcomes come from saved executions. These are controlled
small-model experiments rather than a claim of state-of-the-art accuracy.
\end{abstract}
\section{Questions, controls and provenance}
\paragraph{Split seed.}
The digits of roll number P26CS0001, in order, give the random seed 260001.
\paragraph{Prospective decisions.}
\texttt{EXPERIMENT\_PLAN.md} and \texttt{configs/experiment.json} record the
hypotheses, controls, budgets, operational thresholds, candidate grids and
selection rules before training. We test locality against capacity/optimisation,
colour cues against intensity geometry, and reconstruction against semantic
information.
\paragraph{Data.}
Only official CIFAR10~\cite{cifar} is used: 50,000 training images, ten balanced
classes, and 10,000 official test images. Exactly one stratified 45,000/5,000 split
uses seed 260001. Each training class has 4,500 examples, each validation class
500. Saved indices are unique, disjoint and exhaustive. Channel means and
standard deviations are fitted on the 45,000 training images only. AEs consume
pixels in $[0,1]$; classifiers consume training-standardised pixels. There is no
augmentation, external data, pretrained network, foundation model or architecture
search in the experiments.
\paragraph{Development versus final evaluation.}
Classifiers select minimum validation cross-entropy; AEs select minimum
validation pixel MSE. Probes and SVMs select maximum validation accuracy with
smaller-$C$ tie preference. Clustering uses validation silhouette, without class
labels. A SHA256 manifest seals checkpoints, preprocessing, selections and
experiment source before the test entry point runs. All selected classifiers,
probes and the SVM are then evaluated. Mandatory post-test error and representation
analyses do not change selection. Test cluster assignments are saved before test
labels are accessed.
\paragraph{Execution and assistance.}
Python, NumPy and PyTorch seeds, deterministic CUDA operations and disabled
TF32 are recorded. CPU threads are capped at four. Codex assisted with code and
report generation.
'''
    hardware = setup['hardware']
    device = hardware['gpu'] or hardware['device']
    s+=f'Execution environment: Python {esc(hardware["python"])}, PyTorch {esc(hardware["torch"])}, {esc(device)}. '
    s+=page()+r'''\section{Q1(a), Q2(a--b): testing locality}
\paragraph{Question and competing explanations.}
A CNN may outperform a flattened MLP because local feature extraction is a useful
prior, or simply because it has more trainable parameters or a larger training
opportunity. We match parameter counts and update budgets before measuring
outcomes. An architecture comparison alone cannot establish causation; the
permutation interaction is the more targeted intervention.
\paragraph{Architecture and fairness.}
The MLP is flatten--112--112--10, with ReLU hidden activations. The CNN has three
3$\times$3 padded convolutions, channels 3--32--64--128, each followed by ReLU
and 2$\times$2 max pooling, then flatten 2,048--128--10 with one ReLU. Counts are
357,962 and 356,810 (0.32\% apart). Both use AdamW, initial LR $10^{-3}$,
weight decay $10^{-4}$, cosine decay over 24 epochs, batch 256 and cross-entropy.
Each sees the same examples and minibatch order for a given seed. FLOPs and
runtime differ; identical settings need not be optimal for either family.
\paragraph{Intervention and prediction.}
A saved permutation $p$ of the 1,024 spatial positions, seed 260002, is shared
across RGB and all splits: $x'_{c,j}=x_{c,p(j)}$. It is invertible and preserves
all pixel values. We predicted a larger CNN accuracy loss. For the MLP,
$W'_{:,c,j}=W_{:,c,p(j)}$ transports its initial function, controlling capacity
and much of the optimisation explanation; small numerical divergences remain
possible. The CNN receives the same initial weights in both conditions.
'''
    rows=[]
    for condition in ['clean','permuted','grayscale']:
        for kind in ['mlp','cnn']:
            a=summary[(summary.condition==condition)&(summary.model==kind)].iloc[0]
            result=pct(a.test_mean)+(f' [{pct(a.test_min)}, {pct(a.test_max)}]' if a.n_seeds>1 else '')
            rows.append([esc(condition),kind.upper(),f'{int(a.parameters):,}',result])
    s+=table(['Condition','Model','Params',r'Test \% [range]'],rows,'Validation-selected checkpoints. Clean and permuted results average seeds17/29 with observed range, not a confidence interval. Grayscale uses seed 17.','tab:classifiers')
    s+=fig('classifier_learning','Seed17 end-of-epoch training and validation trajectories. Solid: train; dashed: validation. All 24 epochs execute; test checkpoints use minimum validation CE.')
    s+=r'''\paragraph{Results and limits.}'''
    runtime={}
    for kind in ['mlp','cnn']:
        runtime[kind]=np.mean([j(ROOT/f'artifacts/runs/{kind}_clean_{seed}/complete.json')['seconds'] for seed in [17,29]])
    s+=f'Both models receive 4,224 optimiser steps (176 per epoch), but measured clean-run times average {runtime["mlp"]:.1f}s for the MLP and {runtime["cnn"]:.1f}s for the CNN ({runtime["cnn"]/runtime["mlp"]:.1f} times longer), including epoch evaluations and checkpoint I/O. '
    s+=f'The paired CNN-minus-MLP permutation penalties are {pct(interaction[0]["interaction"])} and {pct(interaction[1]["interaction"])} points for seeds17 and29. '
    if all(row['interaction'] > 0 for row in interaction):
        s+='Both seeds show a larger permutation penalty for the CNN, supporting the architectural-prior explanation. '
    else:
        s+='The permutation result does not consistently support a larger CNN penalty across these seeds. '
    s+='Locality, weight sharing, pooling and optimisation remain coupled within the CNN. '
    s+=page()+r'''\section{Q1(b): structure and memorisation}
\paragraph{Setting and operational definition.}
A larger MLP, 3072--512--512--10 with ReLU, is trained on a fixed stratified
5,000-example subset of the training split. Exactly 2,000 examples are selected
without replacement (seed 43). Each replacement label is uniform among the other
nine classes, hence always disagrees with its original label. The mask, original
labels, replacements and subset indices are saved. AdamW uses LR $10^{-3}$,
zero weight decay, batch 128 and 150 cosine-decayed epochs; no augmentation.
This network has 1,841,162 parameters and enough empirical capacity only if it
actually fits the corrupt targets.
Useful structure is operationally defined as clean validation accuracy above
20\% for two consecutive epochs. Memorisation onset is corrupt-target accuracy
above 20\% twice; saturation is above 80\% twice. These predeclared thresholds
are descriptive and resolution is one epoch. They do not identify the first
individual example memorised or a universal phase transition.
'''
    last=mem['last']
    s+=f'\\paragraph{{Results.}} Useful structure first meets the criterion at epoch {epoch(mem["useful_epoch"])}; memorisation onset at epoch {epoch(mem["memorisation_onset_epoch"])}; saturation at epoch {epoch(mem["saturation_epoch"])}. '
    s+=f'At epoch 150, corrupt-target fit is {pct(last["corrupt_target_accuracy"])}\\%, original-label agreement on those same images {pct(last["corrupt_original_accuracy"])}\\%, uncorrupted training accuracy {pct(last["uncorrupted_accuracy"])}\\%, and clean validation accuracy {pct(last["val_accuracy"])}\\%. '
    if mem['useful_epoch'] is not None and mem['memorisation_onset_epoch'] is not None:
        if mem['useful_epoch']<mem['memorisation_onset_epoch']:
            s+='Useful structure precedes the operational onset in this run. '
        elif mem['useful_epoch']==mem['memorisation_onset_epoch']:
            s+='Both criteria are first met in the same epoch; temporal precedence is unresolved at this resolution. '
        else: s+='Memorisation meets its operational criterion first; these measurements do not support the claim that useful structure appears earlier. '
    s+=r'''Generalisation to clean validation examples and agreement with deliberately wrong
training targets measure different phenomena. Clean structure and memorisation
can coexist; the last checkpoint is not a pure memorisation state. Conclusions
are restricted to this subset, corruption rate, budget and seed.
'''
    s+=fig('memorisation',r'All 150 epochs: clean validation, uncorrupted training, corrupt-target and original-label agreement. The middle panel zooms into the first 20 epochs (dotted 20\% threshold); the right contrasts noisy-training and clean-validation cross-entropy. Threshold crossings are computed directly from these histories.')
    s+=r'''\section{Q2(c): colour cues or intensity geometry?}
\paragraph{Question, prediction and controls.}
Two explanations for the clean CNN advantage are dependence on colour cues and
exploitation of local intensity geometry. Both models are retrained using the
same fixed luminance conversion, $Y=0.2989R+0.5870G+0.1140B$, repeated to three
channels. Exactly the same mapping is used for training, validation and test;
normalisation still uses the common training-channel statistics. Seed17,
architecture, budget and selection criterion are unchanged. If geometry matters,
we expect the CNN advantage to persist after colour removal. This is a
retraining experiment, not a frozen-model distribution-shift test.
'''
    s+=f'\\paragraph{{Outcome.}} The CNN--MLP test gap is {pct(clean_gap)} points for clean RGB and {pct(gray_gap)} for grayscale. '
    if clean_gap > 0 and gray_gap > 0:
        s+='The CNN advantage survives grayscale conversion, consistent with a contribution from intensity geometry. '
    else:
        s+='The clean-versus-grayscale gaps do not establish a persistent CNN advantage across both inputs. '
    s+='Grayscale removes colour cues and can change texture contrast. '
    s+=page()+r'''\section{Q2: errors and confidence across conditions}
\paragraph{Metrics.}
For ten-class softmax probabilities $p$, we report cross-entropy/NLL,
$\frac1n\sum_i\sum_c(p_{ic}-1[y_i=c])^2$ (multiclass Brier), and equal-width
15-bin ECE $\sum_b\frac{n_b}{n}|\mathrm{acc}_b-\mathrm{conf}_b|$. Empty bins
contribute zero. Brier ranges from 0 to 2. Confidence on mistakes is mean maximum
softmax probability among incorrect predictions; it is not an estimate of error
severity. Reliability curves omit empty bins. None of these test measurements
is used for calibration or checkpoint selection.
'''
    rows=[]
    for condition in ['clean','permuted','grayscale']:
        for kind in ['mlp','cnn']:
            m=metrics[f'{kind}_{condition}_17']; rows.append([esc(condition),kind.upper(),num(m['nll'],3),num(m['brier'],3),num(m['ece'],3),num(m['incorrect_confidence'],3)])
    s+=table(['Input','Model','NLL','Brier','ECE','Wrong conf.'],rows,'Official test error/confidence, seed 17. Accuracy is in Table1. ECE depends on binning and is not a proper scoring rule.','tab:confidence')
    s+=fig('confusions','Row-normalised test confusion matrices. Integer class order throughout: 0 airplane,1 automobile,2 bird,3 cat,4 deer,5 dog,6 frog,7 horse,8 ship,9 truck. A darker diagonal indicates better recall.')
    s+=fig('reliability','Test reliability by condition, seed 17. Points are mean confidence and accuracy in nonempty fixed bins; the diagonal is perfect empirical calibration.')
    clean_c=metrics['cnn_clean_17']; perm_c=metrics['cnn_permuted_17']; gray_c=metrics['cnn_grayscale_17']
    s+=f'\\paragraph{{Linked interpretation.}} CNN NLL changes from {num(clean_c["nll"],3)} on clean RGB to {num(perm_c["nll"],3)} after permutation and {num(gray_c["nll"],3)} after grayscale retraining; its confidence on incorrect predictions is {num(clean_c["incorrect_confidence"],3)}, {num(perm_c["incorrect_confidence"],3)}, and {num(gray_c["incorrect_confidence"],3)}, respectively. '
    s+=f'CNN ECE is {num(clean_c["ece"],3)} on clean RGB, {num(perm_c["ece"],3)} on permuted RGB, and {num(gray_c["ece"],3)} on grayscale. '
    s+='The confusion matrices show the class-specific errors behind these aggregate results. '
    s+=page()+r'''\section{Q3(a): bottleneck capacity and reconstruction}
\paragraph{Question and controls.}
Does increasing code capacity improve reconstruction under the same protocol?
The encoder is three stride-2, padded 3$\times$3 convolutions, channels 3--16--32--64,
ReLU after each, ending at 64$\times$4$\times$4. A linear layer gives $d$ latent
coordinates. A linear expansion to 1,024 followed by ReLU and three stride-2
4$\times$4 transposed convolutions, channels 64--32--16--3, reconstructs 32$\times$32
pixels; intermediate layers use ReLU and the last uses sigmoid. Codes have
$d\in\{16,64,256\}$, compared with 3,072 input components. The latent coordinates
are unrestricted real values; dimension ratios are not storage-bit compression
ratios. Only bottleneck-adjacent linear dimensions change. Total parameters grow
by 2,049 per added latent coordinate; encoder/decoder convolutions stay fixed.
\paragraph{Protocol and selection.}
All three use seed 59, AdamW, LR $10^{-3}$, weight decay $10^{-4}$, batch 256,
24 cosine-decayed epochs, and mean squared error over every image/channel/pixel.
The model and checkpoint are chosen using validation reconstruction error only.
Shared seeds do not make different-shaped layers identically initialised. No
class labels are used in AE training or reconstruction selection.
'''
    rows=[]
    for _,a in ae[ae.representation!='dae'].iterrows(): rows.append([int(a.dimension),f'{int(a.parameters):,}',int(a.epoch),num(a.val_mse,5),num(a.clean_test_mse,5)])
    s+=table(['$d$','Parameters','Epoch','Val MSE','Test MSE'],rows,'All three clean AEs. The winner uses validation MSE only. Additional test MSEs for nonwinners are measured for Q3(c), with no reselection.','tab:ae')
    s+=f'The validation winner is $d={selected_dim}$ with MSE {num(sel["validation_mse"],5)}. Its official clean-test MSE is {num(ae[ae.representation==f"ae_{selected_dim}"].clean_test_mse.iloc[0],5)}. '
    s+=r'''A larger code can retain more pixel detail but also changes the number of
bottleneck weights. Consequently this experiment estimates the effect of code
capacity with its unavoidable adjacent-parameter change; it is not an
independent parameter-budget ablation.
'''
    s+=fig('ae_learning','Validation reconstruction MSE for each bottleneck. All three receive the same 24-epoch budget; the selected epoch may differ.',wide=False)
    s+=fig('ae_reconstructions','One reproducible test image per class: first index of that true class, chosen only for post-test illustration. Top: clean target; subsequent rows: clean-input reconstructions at each dimension. Same examples across rows; numeric MSE averages the entire test set.')
    s+=r'''\paragraph{Limits.}
Pixel MSE favours averages and is sensitive to texture and background. Qualitative
sharpness is illustrative, not a substitute for the full-set metric. Validation
reconstruction selection says nothing by itself about which representation best
supports classification. Q3(c) tests that claim separately with frozen encoders.
'''
    s+=page()+r'''\section{Q3(b--c): denoising and class information}
\paragraph{Denoising hypothesis and training.}
Independent additive Gaussian noise with $\sigma=0.2$ is clipped to$[0,1]$.
It can plausibly be reduced by reconstruction, though smoothing may remove class
cues. The DAE uses the validation-selected clean-AE dimension and the same
architecture/24-epoch optimiser protocol, but learns noisy inputs$\to$clean targets
without labels. Training noise has explicit run/epoch/batch seeds; validation
uses seed 590001 plus batch start. Test noise uses 590002 plus batch start with
batch 256. All test conditions share the same stored underlying noisy tensor.
The seed 17 clean CNN is frozen throughout: noisy pixels, clean-AE reconstructions
and DAE reconstructions all receive the classifier's training mean/std after
reconstruction. Sigmoid outputs remain in$[0,1]$; there is no erroneous second
$1/255$ scaling. The clean CNN is reused from Q1(a)/Q2(a), the clean AE from Q3(a).
'''
    rows=[]
    for key,label in [('corrupted','Noisy input'),(f'ae_{selected_dim}','Clean AE'),('dae','Denoising AE')]:
        a=den[key]; rows.append([label,num(a['mse'],5),num(a['psnr_from_mean_mse'],2),pct(a['accuracy']),num(a['nll'],3)])
    s+=table(['Condition','MSE','PSNR',r'Acc.\%','NLL'],rows,r'Paired noisy-image evaluation with the same frozen CNN. PSNR is $10\log_{10}(1/\mathrm{MSE})$ from dataset mean MSE, not the mean of per-image PSNR.','tab:denoise')
    delta=den['dae']['accuracy']-den['corrupted']['accuracy']
    s+=f'DAE classification changes by {pct(delta)} points relative to noisy pixels; it corrects {den["paired_changes"]["noise_wrong_dae_correct"]} previously wrong examples and breaks {den["paired_changes"]["noise_correct_dae_wrong"]} previously correct ones. '
    s+=('Denoising improves classifier accuracy on the noisy test images. ' if delta>0 else 'Denoising does not improve classifier accuracy on balance in this setting. ')
    baseline=den[f'ae_{selected_dim}']
    s+=f'The clean-AE baseline reaches {pct(baseline["accuracy"])}\\% versus {pct(den["dae"]["accuracy"])}\\% for the DAE: the DAE changes classifier accuracy by {pct(den["dae"]["accuracy"]-baseline["accuracy"])} points and changes pixel MSE by {num(den["dae"]["mse"]-baseline["mse"],5)}. This single corruption draw does not establish statistical significance. '
    s+=r'''Lower reconstruction MSE is not sufficient evidence of semantic recovery.
'''
    s+=fig('denoising_reconstructions','Exactly paired clean targets, noisy inputs, clean-AE and DAE reconstructions. Examples match the clean reconstruction panel. Noise is clipped before either reconstruction path.')
    s+=r'''\paragraph{Frozen representation probe.}
For each clean AE and the DAE, extract the frozen encoder's linear bottleneck
from \emph{clean} images. Fit a training-only StandardScaler and multinomial
logistic regression (LBFGS, max 2,000 iterations, tolerance$10^{-4}$); choose
$C\in\{.01,.1,1,10\}$ by validation accuracy with smaller-$C$ tie preference.
No encoder is fine tuned. Identical probe protocol controls the readout family;
different representation dimensions still change probe parameter counts.
DAE reconstruction in the following comparison also uses clean inputs, to match
probe inputs. Its training-selection validation MSE used noisy inputs instead.
'''
    rows=[]
    for _,a in ae.iterrows(): rows.append([esc(a.representation),int(a.dimension),num(a.clean_test_mse,5),a.probe_C,pct(a.probe_test_accuracy)])
    s+=table(['Encoder','$d$','Clean MSE','Probe $C$',r'Test acc.\%'],rows,'Q3(c): separately required test reconstruction/semantic comparison, frozen before test. These extra measurements do not change Q3(a) selection.','tab:probes')
    best_rec=ae.sort_values('clean_test_mse').iloc[0]; best_probe=ae.sort_values('probe_test_accuracy',ascending=False).iloc[0]
    s+=f'The lowest clean-test MSE belongs to {esc(best_rec.representation)}, whereas the strongest probe is {esc(best_probe.representation)}. '
    s+=('Their winners coincide here, but this single agreement does not establish a general equivalence between reconstruction and semantics. ' if best_rec.representation==best_probe.representation else 'The winners disagree, directly showing that reconstruction selection and linear class information need not agree. ')
    discordant=sum((a.clean_test_mse-b.clean_test_mse)*(a.probe_test_accuracy-b.probe_test_accuracy)>0 for a,b in __import__('itertools').combinations(ae.itertuples(),2))
    s+=f'Across the six pairs of encoders, {discordant} have an opposite reconstruction-versus-probe ordering (ties excluded); this is a descriptive comparison of four representations, not a significance test. '
    s+=r'''A linear probe measures information accessible to that readout, not all class
information recoverable by arbitrary nonlinear classifiers.
'''
    s+=page()+r'''\section{Q4(a): a fixed-representation SVM sweep}
\paragraph{Representation, fairness and selection.}
Before outcomes, Q4 fixes the 128-dimensional penultimate ReLU of the seed 17
clean CNN from Q1/Q2. The checkpoint and extraction stay unchanged throughout
Q4. This compact supervised geometry makes kernel fitting feasible and supports
a direct classification-versus-clustering comparison; a shared StandardScaler
fits all 45,000 training representations. There is
no PCA or probe-based representation selection. To bound kernel cost, every
SVM uses the same saved stratified 6,000-example training subset (seed 71,
600 per class), with all 5,000 validation examples. The inherited CNN was
trained on all 45,000 training labels; the subset bounds kernel optimisation
cost and is not a claim of learning from only 6,000 labelled images. Scaling uses only training
images, including those outside the SVM subset; this is declared shared
unsupervised preprocessing. Clustering later uses 45,000 training representations.
We test six $C$ values over five orders of magnitude and linear, RBF and
cubic polynomial kernels. Gamma is \texttt{scale}, degree 3, coef0=1,
libsvm tolerance 0.001, unlimited iterations and no probability fitting.
Actual gamma and support counts are logged. Multiclass training is one-versus-one.
Highest validation accuracy selects the final configuration; ties prefer smaller
$C$ then kernel order linear/RBF/poly. Only that winner is tested.
'''
    s+=f'The common realised gamma is {sweep.gamma.iloc[0]:.8f}, computed from training-subset feature variance; it is unchanged across the grid. '
    rows=[]
    for kernel in ['linear','rbf','poly']:
        for _,a in sweep[sweep.kernel==kernel].sort_values('C').iterrows():
            rows.append([kernel,f'{a.C:g}',pct(a.train_accuracy),pct(a.val_accuracy),pct(a.sv_fraction)])
    s+=table(['Kernel','$C$',r'Train\%',r'Val\%',r'SV\%'],rows,'Complete sweep. Support-vector fraction denominator is6,000 for every setting. Validation and training accuracy refer to the same frozen feature extraction and scaling.','tab:svm')
    s+=f'\\paragraph{{Selected outcome.}} {esc(svm_sel["kernel"])} with $C={svm_sel["C"]:g}$, validation accuracy {pct(svm_sel["val_accuracy"])}\\%, official test accuracy {pct(svm["accuracy"])}\\%. '
    for kernel in ['linear','rbf','poly']:
        g=sweep[sweep.kernel==kernel].sort_values('C'); lo=g.iloc[0]; hi=g.iloc[-1]
        s+=f'For {kernel}, $C=.001\\to100$ changes train accuracy {pct(lo.train_accuracy)}$\\to${pct(hi.train_accuracy)}\\% and validation {pct(lo.val_accuracy)}$\\to${pct(hi.val_accuracy)}\\%; the final train--validation gap is {pct(hi.train_accuracy-hi.val_accuracy)} points. '
        peak=g.sort_values(['val_accuracy','C'],ascending=[False,True]).iloc[0]
        if lo.train_accuracy+.05<hi.train_accuracy and lo.val_accuracy+.02<peak.val_accuracy:
            s+=f'The smallest $C$ therefore underfits relative to this grid. '
        if hi.train_accuracy-hi.val_accuracy>.10 and hi.val_accuracy+.005<peak.val_accuracy:
            s+=f'At $C=100$, validation is below its peak at $C={peak.C:g}$ despite a large training advantage, consistent with overfitting. '
        else:
            s+='There is no clear high-$C$ validation decline of more than half a point; a train--validation gap alone is weaker evidence of deterioration. '
    s+=r'''\paragraph{Interpretation.}
Low training and validation accuracy indicates underfitting; a growing training
advantage without validation improvement indicates overfitting. These measured
trends, rather than $C$ alone or a monotonic-support assumption, determine that
interpretation. Support vectors are active constraints, not mislabeled examples.
Kernel smoothness, regularisation and inherited CNN geometry interact. This
subset-limited SVM is not a claim about full 45,000-example kernel training.
'''
    s+=fig('svm_sweep','Top: train/validation accuracy against $C$. Bottom: support-vector fraction against $C$, separately for each kernel. All six settings per curve are actually fitted.')
    s+=page()+r'''\section{Q4(b): difficult pairs and boundaries}
\paragraph{Post-test selection and score meaning.}
The three pairs below maximise summed bidirectional off-diagonal test confusion
counts, with lexicographic ties. This diagnostic selection is required by Q4(b)
and does not affect the already selected model. For classes $i<j$, the saved
45-column OVO score uses positive for$i$ and negative for$j$; sign convention is
checked against multiclass voting. Scores are functional decision values, not
normalised kernel-space distances or aggregated one-versus-rest confidence.
Only training vectors with nonzero coefficients in that specific binary
classifier are eligible for the support panels. Each panel displays the six
eligible support vectors and six true-$i$/$j$ test examples with smallest absolute
pair score. Small $|f|$ means proximity in decision-value units within that pair,
not comparable metric distance across different pairs. T is the true class and
P the final ten-class prediction; a third class can override a binary vote.
'''
    observations=j(P/'pair_observations.json') if (P/'pair_observations.json').exists() else {}
    for number,(_,a) in enumerate(pairs.iterrows()):
        observation=observations.get(f'{int(a.class_i)}-{int(a.class_j)}','')
        s+=fig(f'pair_{number}',fr'{esc(a.name_i)} versus {esc(a.name_j)}: {int(a.i_to_j)} and {int(a.j_to_i)} directed errors ({int(a.pair_errors)} total). Every training image shown is active for this pair; exact indices, unrounded scores and dual coefficients are saved in pair\_examples.csv. '+esc(observation))
    s+=r'''\paragraph{What the images establish.}
The panels expose ambiguous backgrounds, poses, outlines or textures near the
learned binary surfaces, while training support vectors identify active
constraints. Similar-looking images can occupy overlapping feature regions;
however visual inspection alone cannot prove that a particular cue caused an
error, identify label noise, or establish Euclidean overlap in the original
pixel space. The quantitative pair counts and shared-cluster statistics below
provide complementary evidence about representation geometry.
'''
    s+=page()+r'''\section{Q4(c): clustering the same representation}
\paragraph{Label-free method selection.}
The inherited CNN representation was learned with training labels; this is
unsupervised clustering within a previously supervised feature space, not a
claim of fully unsupervised representation learning. KMeans uses the identical
128-dimensional CNN representation and shared
training-fitted scaler. Fit only on 45,000 training representations, evaluate
$k\in\{8,10,12,15,20\}$ using a fixed 2,000-example validation sample, three
seeds 83/97/109, ten initialisations each and at most 300 Lloyd iterations.
Choose highest mean validation silhouette, ties smaller $k$. Stability is mean
ARI between validation cluster assignments across seeds, without semantic
labels; it is diagnostic, not a test-label selection criterion. Final fit uses
seed 83 at the chosen $k$, and nearest-centroid prediction is a valid out-of-sample
rule. Test assignments are saved before accessing test labels. Thus evaluation
labels neither fit centroids nor choose$k$. KMeans uses more training samples
than the fixed 6,000-example SVM, as declared in advance.
'''
    rows=[]
    for _,a in clusters.iterrows(): rows.append([int(a.k),num(a.mean_val_silhouette,4),num(a.std_val_silhouette,4),num(a.mean_seed_ARI,4)])
    s+=table(['$k$','Mean silhouette','SD','Seed ARI'],rows,'Internal validation selection and stability. ARI here compares seed assignments, not clusters with true classes.','tab:clusterselection')
    s+=f'The selected $k={clusterchoice["k"]}$ gives test ARI {num(clustering["ARI"])} and NMI {num(clustering["NMI"])} (arithmetic normalisation). '
    s+=r'''ARI compares pairwise partitions adjusted for chance; NMI measures label--cluster
information. Neither is directly comparable numerically with supervised accuracy.
The full contingency table shows where their disagreement arises.
'''
    contingency=pd.read_csv(T/'cluster_contingency.csv',index_col=0)
    rows=[[int(ix)]+[int(a) for a in row] for ix,row in contingency.iterrows()]
    s+=table(['Cluster']+[str(i) for i in range(10)],rows,'Test cluster-to-class counts; columns use the class order defined in the confusion-matrix caption. Each class column sums to 1,000; all cells sum to 10,000.','tab:contingency')
    # A class-level disagreement always exists quantitatively; select largest SVM recall minus modal cluster concentration.
    cm=np.array(svm['confusion']); ct=contingency.to_numpy(); recall=np.diag(cm)/cm.sum(1); modal=ct.max(0)/ct.sum(0)
    ix=int(np.argmax(recall-modal)); mode=int(ct[:,ix].argmax()); purity=ct[mode,ix]/ct[mode].sum()
    s+=f'\\paragraph{{Quantitative disagreement.}} For {names[ix]}, supervised SVM recall is {pct(recall[ix])}\\%, while {pct(modal[ix])}\\% of true {names[ix]} examples occupy their largest cluster (cluster{mode}); that cluster has {pct(purity)}\\% {names[ix]} purity. '
    for _,a in pairs.iterrows():
        s+=f'{esc(a.name_i)}/{esc(a.name_j)} have SVM cross-confusion {pct(a.pair_confusion_rate)}\\% among those 2,000 examples, but cross-class same-cluster probability {num(a.cluster_cross_class_coassignment,4)}. '
    s+=r'''The latter statistic is $\sum_k n_{ki}n_{kj}/(n_i n_j)$: draw one member from
each true class, then ask whether their assigned clusters agree. Low supervised
pair confusion need not imply separated unsupervised clusters. These statistics
have different denominators and meanings; no cluster-label mapping is tuned on
the test set. Supervision can learn boundaries inside broad density groups.
\section{Reproducibility and limitations}
\paragraph{Reproduction and reuse.}
Run \texttt{prepare.py}, integrity tests, \texttt{train.py}, \texttt{selection.py},
\texttt{evaluate.py}, \texttt{figures.py}, then \texttt{verify.py}; exact shell
commands and locked dependencies are in README. Per-run metadata records config,
seed, split/source hashes, hardware, epochs, runtime and checkpoint selection;
restart files preserve optimiser/scheduler/RNG states. Saved machine-readable
results generate every table and figure. The clean CNN/MLP first appear in Q1(a), are reused unchanged in Q2(a), and then extend to its interventions;
clean AEs in Q3(a), DAE in Q3(b), then frozen reuse in Q3(c)/Q4 is explicit.
\paragraph{Scope.}
The commit on page 1 identifies the archived experiment code and results. The
pivotal comparisons use two seeds and the memorisation experiment uses one.
A single split, compact architectures, and finite training budgets limit
generalisability.
'''
    s+='\n\\clearpage\n'+r'''{\small
\begin{thebibliography}{9}
\bibitem{cifar} Alex Krizhevsky. Learning Multiple Layers of Features from Tiny
Images. Technical report, University of Toronto, 2009. Official dataset:
\url{https://www.cs.toronto.edu/~kriz/cifar.html}.
\bibitem{torch} PyTorch contributors. PyTorch2.6 documentation and torchvision
CIFAR10 implementation. \url{https://pytorch.org/docs/2.6/} and
\url{https://pytorch.org/vision/0.21/generated/torchvision.datasets.CIFAR10.html}.
\bibitem{sklearn} Scikit-learn contributors. Version1.6 documentation: SVC,
LogisticRegression, KMeans, silhouette, ARI and NMI.
\url{https://scikit-learn.org/1.6/}.
\bibitem{cvpr} CVPR organisers. Official CVPR2026 author kit, tagged
CVPR2026-v1(latex). \url{https://github.com/cvpr-org/author-kit}.
\end{thebibliography}}
\end{document}
'''
    (P/'ML_Assignment_3_P26CS0001.tex').write_text(s)
    print('Generated editable manuscript from saved results; experiment commit',commit)
if __name__=='__main__': main()
