# Discrete Latent Representations: A Learning Journey

A personal exploration of discrete latent representations—from foundational autoencoders to modern visual tokenizers.

---

## Philosophy

This repository is designed as a **learning-first** codebase. The goal isn't to have the cleanest abstractions or the most general framework, but to deeply understand each component by implementing it yourself. Feel free to refactor, restructure, and redesign as your understanding evolves.

---

## Learning Roadmap

### Phase 0: Foundations (Optional but Recommended)

Before diving into discrete representations, ensure you're comfortable with continuous latent spaces.

#### Concepts to Understand
- **Autoencoders**: Encoder-decoder architectures, bottleneck representations, reconstruction losses
- **Variational Autoencoders (VAEs)**: The ELBO objective, KL divergence regularization, the reparameterization trick
- **The problem**: Why do we want discrete representations? (compression, interpretability, compatibility with language models, avoiding posterior collapse)

#### Papers
- Kingma & Welling, 2013 - *Auto-Encoding Variational Bayes* ([arXiv:1312.6114](https://arxiv.org/abs/1312.6114))
- Bowman et al., 2015 - *Generating Sentences from a Continuous Space* (posterior collapse discussion)

#### Implementation Challenge
Build a simple VAE on MNIST or CIFAR-10. Visualize the latent space. Notice how the continuous latent space can be "fuzzy"—nearby points decode to similar but not identical outputs.

---

### Phase 1: Vector Quantization — The Core Idea

This is where discrete representations begin. VQ-VAE introduced a simple but powerful idea: instead of continuous latents, map to the nearest vector in a learned codebook.

#### Concepts to Understand
- **Vector Quantization**: Given encoder output `z_e`, find nearest codebook vector `e_k`
- **The gradient problem**: Quantization is non-differentiable. How do we train through it?
- **Straight-through estimator**: Copy gradients from decoder input to encoder output
- **The three losses**: Reconstruction, codebook (moving codebook vectors toward encoder outputs), commitment (keeping encoder outputs close to codebook)
- **Codebook collapse**: When only a few codebook entries get used. Why does it happen?

#### Papers
- van den Oord et al., 2017 - *Neural Discrete Representation Learning* (VQ-VAE) ([arXiv:1711.00937](https://arxiv.org/abs/1711.00937))

#### Implementation Challenge
Implement VQ-VAE from scratch. Key decisions you'll face:
- How to initialize the codebook?
- What's the right balance between commitment loss and codebook loss?
- How to track codebook utilization?

**Diagnostic**: Plot codebook usage histogram. If most entries are dead, you have collapse.

---

### Phase 2: Stabilizing Vector Quantization

The original VQ-VAE training can be unstable. This phase is about understanding and fixing these issues.

#### Concepts to Understand
- **Exponential Moving Average (EMA) updates**: Instead of gradient descent on codebook, use running averages of encoder outputs
- **Codebook reset/reinitialization**: Detecting and replacing dead codes
- **Entropy regularization**: Encouraging uniform codebook usage
- **Increased codebook size**: Trade-offs between expressiveness and utilization

#### Papers
- Razavi et al., 2019 - *Generating Diverse High-Fidelity Images with VQ-VAE-2* ([arXiv:1906.00446](https://arxiv.org/abs/1906.00446)) — introduces hierarchical VQ and better training
- Dhariwal et al., 2020 - *Jukebox* (Appendix A has excellent VQ training tricks)
- Huh et al., 2023 - *Straightening Out the Straight-Through Estimator* ([arXiv:2305.08842](https://arxiv.org/abs/2305.08842))

#### Implementation Challenge
Add EMA codebook updates to your VQ-VAE. Implement dead code detection and reinitialization. Compare training stability and final codebook utilization vs. vanilla VQ-VAE.

---

### Phase 3: Soft Quantization & Relaxations

Hard quantization (argmax to nearest codebook) is one approach. Another family uses soft/probabilistic assignments.

#### Concepts to Understand
- **Gumbel-Softmax / Concrete distribution**: A differentiable approximation to categorical sampling
- **Temperature annealing**: Start soft, gradually harden
- **dVAE (discrete VAE)**: As used in DALL-E's first stage
- **Soft vs. Hard trade-offs**: Soft is easier to train but introduces "blurriness" in the discrete space

#### Papers
- Jang et al., 2016 - *Categorical Reparameterization with Gumbel-Softmax* ([arXiv:1611.01144](https://arxiv.org/abs/1611.01144))
- Maddison et al., 2016 - *The Concrete Distribution* ([arXiv:1611.00712](https://arxiv.org/abs/1611.00712))
- Ramesh et al., 2021 - *Zero-Shot Text-to-Image Generation* (DALL-E) ([arXiv:2102.12092](https://arxiv.org/abs/2102.12092)) — Section 3 on dVAE

#### Implementation Challenge
Implement Gumbel-Softmax quantization as an alternative to hard VQ. Experiment with temperature schedules. When does soft quantization help? When does it hurt?

---

### Phase 4: Hierarchical & Multi-Scale Quantization

Real images have structure at multiple scales. Hierarchical approaches capture this.

#### Concepts to Understand
- **VQ-VAE-2**: Multiple levels of discrete latents (e.g., 32×32 "bottom" and 8×8 "top")
- **Residual Quantization (RQ)**: Quantize, compute residual, quantize residual, repeat
- **Product Quantization (PQ)**: Split dimensions into groups, quantize each group independently
- **Trade-offs**: Hierarchical = more tokens, more expressiveness, harder prior modeling

#### Papers
- Razavi et al., 2019 - *VQ-VAE-2* (hierarchical)
- Lee et al., 2022 - *Autoregressive Image Generation using Residual Quantization* (RQ-VAE) ([arXiv:2203.01941](https://arxiv.org/abs/2203.01941))
- Zeghidour et al., 2021 - *SoundStream* ([arXiv:2107.03312](https://arxiv.org/abs/2107.03312)) — RVQ for audio

#### Implementation Challenge
Extend your VQ-VAE to have two levels of latents. The top level should capture global structure, the bottom level should capture details. How do you condition the bottom level on the top?

---

### Phase 5: Beyond Reconstruction — Perceptual & Adversarial Losses

Pure reconstruction loss (MSE/L1) leads to blurry outputs. Modern tokenizers use perceptual and adversarial losses.

#### Concepts to Understand
- **Perceptual loss**: Compare features from a pretrained network (VGG, LPIPS) rather than raw pixels
- **Adversarial loss**: Add a discriminator to push outputs toward the real image manifold
- **VQGAN**: VQ-VAE + perceptual loss + PatchGAN discriminator
- **Loss balancing**: How to weight reconstruction vs. perceptual vs. adversarial vs. commitment losses?

#### Papers
- Esser et al., 2021 - *Taming Transformers for High-Resolution Image Synthesis* (VQGAN) ([arXiv:2012.09841](https://arxiv.org/abs/2012.09841))
- Zhang et al., 2018 - *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS) ([arXiv:1801.03924](https://arxiv.org/abs/1801.03924))

#### Implementation Challenge
Add a PatchGAN discriminator to your VQ-VAE. Add perceptual loss (LPIPS or VGG features). The training becomes more complex—experiment with loss weights and training schedules.

---

### Phase 6: Modern Tokenizers — Simplifying & Scaling

Recent work has found simpler alternatives to VQ that avoid codebook collapse entirely.

#### Concepts to Understand
- **Finite Scalar Quantization (FSQ)**: Instead of a codebook, just round each dimension to a small set of values
- **Lookup-Free Quantization (LFQ)**: Binary quantization with entropy regularization
- **Why simpler can be better**: No codebook = no collapse, easier to scale

#### Papers
- Mentzer et al., 2023 - *Finite Scalar Quantization: VQ-VAE Made Simple* ([arXiv:2309.15505](https://arxiv.org/abs/2309.15505))
- Yu et al., 2023 - *Language Model Beats Diffusion — Tokenizer is Key to Visual Generation* (MAGVIT-2, LFQ) ([arXiv:2310.05737](https://arxiv.org/abs/2310.05737))

#### Implementation Challenge
Implement FSQ. It's surprisingly simple—just a few lines of code. Compare codebook utilization, reconstruction quality, and training stability vs. your VQ implementations.

---

### Phase 7: State-of-the-Art Visual Tokenizers

Putting it all together: modern tokenizers combine the best ideas with careful engineering.

#### Key Systems to Study
- **VQGAN** (Esser et al., 2021): The baseline that launched a thousand image generators
- **MAGVIT / MAGVIT-2** (Yu et al., 2023): LFQ + careful architecture + 3D convolutions for video
- **Open-MAGVIT2** (Luo et al., 2024): Open reproduction with training recipes
- **Cosmos Tokenizer** (NVIDIA, 2024): State-of-the-art image/video tokenizer
- **TiTok** (Yu et al., 2024): 1D tokenization — images as sequences of ~32 tokens

#### Papers
- Yu et al., 2023 - *MAGVIT-2* ([arXiv:2310.05737](https://arxiv.org/abs/2310.05737))
- Luo et al., 2024 - *Open-MAGVIT2* ([arXiv:2409.04410](https://arxiv.org/abs/2409.04410))
- Yu et al., 2024 - *An Image is Worth 32 Tokens for Reconstruction and Generation* (TiTok) ([arXiv:2406.07550](https://arxiv.org/abs/2406.07550))

#### Implementation Challenge
Pick one modern tokenizer and try to reproduce it. What design decisions matter most? How do the components interact?

---

### Phase 8: Priors Over Discrete Latents

A tokenizer is only half the story. To generate, you need a prior over the discrete codes.

#### Concepts to Understand
- **Autoregressive priors**: GPT-style transformers predicting tokens left-to-right (or raster-scan)
- **Masked prediction**: BERT-style parallel decoding (MaskGIT)
- **Diffusion over discrete tokens**: Discrete diffusion, absorbing states
- **Why discrete helps**: Enables use of powerful language model architectures

#### Papers
- Esser et al., 2021 - *Taming Transformers* (autoregressive prior)
- Chang et al., 2022 - *MaskGIT: Masked Generative Image Transformer* ([arXiv:2202.04200](https://arxiv.org/abs/2202.04200))
- Austin et al., 2021 - *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM) ([arXiv:2107.03006](https://arxiv.org/abs/2107.03006))

#### Implementation Challenge
Train a small transformer prior over your tokenizer's latent codes. Compare autoregressive vs. masked prediction. How does tokenizer quality affect generation quality?

---

## Suggested Project Structure

This is just a suggestion—design what makes sense to you:

```
discrete-latent-representations/
├── configs/           # Experiment configs (yaml/json/python)
├── data/              # Dataset utilities
├── experiments/       # Training scripts, logs, checkpoints
├── notebooks/         # Exploration and visualization
└── src/               # Your modules (or flat structure, up to you)
```

---

## Minimal Dependencies

A starting point for `pyproject.toml`:

```toml
dependencies = [
    "torch>=2.0",
    "torchvision",
    "einops",           # Tensor reshaping sanity
    "wandb",            # Experiment tracking (or tensorboard)
    "pillow",
    "tqdm",
]
```

Add as you need: `lpips`, `timm`, `transformers`, `accelerate`, etc.

---

## Tips for the Journey

1. **Start simple**: MNIST/CIFAR before ImageNet. 32×32 before 256×256.

2. **Visualize everything**: Reconstructions, codebook usage histograms, latent space, dead codes over time.

3. **Track metrics**: Reconstruction loss isn't enough. Track PSNR, SSIM, LPIPS, codebook perplexity.

4. **Read the appendices**: Training details are often hidden there.

5. **Ablate ruthlessly**: When something doesn't work, isolate the problem.

6. **Compare apples to apples**: Same dataset, same architecture, change one thing at a time.

---

## Progress Tracker

- [ ] Phase 0: VAE baseline
- [ ] Phase 1: VQ-VAE
- [ ] Phase 2: Stable VQ (EMA, dead code reset)
- [ ] Phase 3: Gumbel-Softmax / dVAE
- [ ] Phase 4: Hierarchical / RQ-VAE
- [ ] Phase 5: VQGAN (perceptual + adversarial)
- [ ] Phase 6: FSQ / LFQ
- [ ] Phase 7: Modern tokenizer reproduction
- [ ] Phase 8: Prior modeling

---

## References (Quick Links)

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [VQ-VAE](https://arxiv.org/abs/1711.00937) | 2017 | Vector quantization for neural networks |
| [VQ-VAE-2](https://arxiv.org/abs/1906.00446) | 2019 | Hierarchical discrete latents |
| [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) | 2016 | Differentiable categorical sampling |
| [VQGAN](https://arxiv.org/abs/2012.09841) | 2021 | Perceptual + adversarial training |
| [FSQ](https://arxiv.org/abs/2309.15505) | 2023 | Codebook-free quantization |
| [MAGVIT-2](https://arxiv.org/abs/2310.05737) | 2023 | Lookup-free quantization, video |
| [MaskGIT](https://arxiv.org/abs/2202.04200) | 2022 | Parallel decoding with masked prediction |

---

Happy learning! 🎯


