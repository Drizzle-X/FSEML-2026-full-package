# Replay Reconstruction Fidelity: Quantitative Analysis

## Experimental protocol

- Dataset: Omniglot background/meta-training split, classes 0–962.
- Random seed: 9.
- Latent dimension: 128.
- Replay buffer: 1,000 samples.
- Simple replay: replay rate 0.10 and replay gap 960.
- Training duration: 80,000 meta-training iterations.
- Process analysis: all 83 replay events, with 200 extra buffer samples per event. These samples were used only for evaluation and did not participate in gradient updates.
- Final analysis: all 1,000 paired samples in the final replay buffer.
- Pixel MSE and SSIM were evaluated in the common [0,1] image space.
- Semantic metrics used an independently trained and frozen Omniglot classifier shared by both runs.
- Only one seed was evaluated; the results are descriptive rather than a multi-seed significance analysis.

## Final-buffer results

| Method | Pixel MSE ↓ | SSIM ↑ | Frozen feature cosine ↑ | Independent real ACC ↑ | Independent pseudo ACC ↑ | LPR ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | 0.0206 ± 0.0087 | 0.8221 ± 0.0521 | 0.7855 ± 0.0766 | 93.40% | 0.90% | 0.96% |
| Warm-start | 0.0228 ± 0.0100 | 0.8023 ± 0.0608 | 0.7676 ± 0.0809 | 93.40% | 0.70% | 0.64% |

The scratch run is better than warm-start on all five reconstruction-fidelity measures. Both runs recover image structure and moderate feature similarity, but neither preserves sufficient class-discriminative information for the independent classifier.

## Evolution across all replay events

For Scratch, the mean of the last ten events compared with the first ten events changes as follows:

- Pixel MSE: 0.0287 → 0.0204.
- SSIM: 0.7385 → 0.8241.
- Frozen feature cosine: 0.7126 → 0.7867.
- Independent pseudo ACC: 0.15% → 1.60%.
- LPR: 0.16% → 1.64%.

For Warm-start:

- Pixel MSE: 0.0335 → 0.0225.
- SSIM: 0.6814 → 0.8033.
- Frozen feature cosine: 0.6809 → 0.7654.
- Independent pseudo ACC: 0.00% → 1.00%.
- LPR: 0.00% → 1.06%.

Thus, reconstruction quality improves rather than deteriorates with task age/training time. However, the improvement is substantially stronger for coarse structure and feature direction than for class identity.

## Why feature cosine is high while semantic ACC is low

The two metrics test different properties. Frozen feature cosine measures the angle between two continuous feature vectors. It can remain moderately high when the reconstruction retains global stroke density, position, background, and coarse shape. Independent classifier ACC is a strict 963-way decision: small losses of class-specific strokes can move the reconstruction across a decision boundary even when the overall feature vectors remain aligned.

The final prediction distributions provide direct evidence of semantic compression/collapse:

- The final buffer contains 617 distinct true classes.
- Scratch reconstructions produce only 112 distinct predicted classes; the five most frequent predictions account for 43.5% of all samples.
- Warm-start reconstructions produce only 87 distinct predicted classes; the five most frequent predictions account for 52.1% of all samples.
- Event-level feature cosine has only moderate correlation with independent pseudo ACC (Scratch: r=0.489; Warm-start: r=0.484) and LPR (Scratch: r=0.472; Warm-start: r=0.471).

Therefore, feature cosine must not be described as equivalent to semantic preservation. The results indicate preservation of coarse representational structure, alongside severe loss of fine-grained class identity.

## Recommended presentation in the paper

Retain all five pre-specified metrics and report the real-image reference accuracy beside pseudo-image accuracy. Omitting Independent ACC or LPR after observing low results would appear selective and would weaken the response to the reviewer’s request for direct replay-quality evidence.

The main text should make a narrower claim:

1. FSEML replay preserves structural and coarse feature information, as supported by MSE, SSIM, and frozen-feature cosine.
2. Fine-grained semantic fidelity remains limited, as shown by Independent ACC and LPR.
3. The method should not be characterized as producing photorealistic or fully class-faithful reconstructions.
4. The single-seed Omniglot study is an empirical diagnostic and a limitation, not evidence of universal replay fidelity.

Independent ACC and LPR can remain in the main quantitative table because they directly answer the reviewer. The 83-event curves and prediction-collapse analysis are suitable for a supplementary figure or appendix if main-text space is limited.

