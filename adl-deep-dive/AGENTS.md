
# Agent Guidelines

## Criteria for the Deep Dive on Kolmogorov–Arnold Networks (KANs)

- Discover topics that go beyond the lecture
- Show connections to the lecture content (where applicable)
- Choose a topic in teams of 2 students
- We will have a small “conference” at the end of the term
- Topics are organized in tracks.
- Important
- Connect with teams in your track and coordinate
- Show direct links to lecture content
- 15 minutes (3min questions), equal share of speaking time and preparation work
- Target audience: graduate students that took Advanced Deep Learning
- Coordinate with the teams in your tracks. Goal: No duplication of content.
- When working on a broader task, render and inspect the slides you are working on to ensure correct rendering and formatting.
- Include references to other relevant literature as referenced in the KAN paper. their bibliographic entries can be found in `arXiv-kan-2404.19756/ref.bib`.
- use `$bold(..)$` for vectors, matrices, and tensors.


## Style guidelines

- Aim for clarity, conciseness and high educational value in explanations, assuming the audience has a graduate-level understanding of machine learning and has completed the Advanced Deep Learning course but may not be familiar with the specific techniques used in this project.
- Use equations and figures from the original KAN paper where appropriate to illustrate key concepts.
- Always cite the paper using "@kan-liu2025" when including their figures or equations. Copy used figures from `arXiv-kan-2404.19756/figs` to `slides/figures`.
- Use get-library-docs "/websites/typst_app" to retrieve context about Typst syntax and commands when generating Typst code.
- When making changes to the Typst slides, ensure that any new figures are stored in the `figures/` directory and referenced correctly in the Typst code.
- Provide concise bullet points, avoiding long paragraphs and aiming for slides with two columns where appropriate.
- Always start by reading and digesting the original KAN paper `arXiv-kan-2404.19756/kan.tex`. From there, identify key concepts, methods, and results that can be linked to the lecture content.

## Directory structure of @kan-liu2025

```
arXiv-kan-2404.19756
├── figs
│   ├── al_complex.png
│   ├── best_feynman_kan.pdf
│   ├── best_special_kan.pdf
│   ├── continual_learning.pdf
│   ├── decision_tree.png
│   ├── density.png
│   ├── extend_grid.pdf
│   ├── feynman_pf.pdf
│   ├── fitting_feynman_bonus.pdf
│   ├── fitting_feynman.pdf
│   ├── fitting_special.pdf
│   ├── flowchart.png
│   ├── interpretability_hyperparameters.png
│   ├── interpretable_examples_short.png
│   ├── interpretable_examples.png
│   ├── kan_mlp.pdf
│   ├── knot_unsupervised.pdf
│   ├── knot_unsupervised.png
│   ├── lan_interpretable_examples.png
│   ├── lan_toy_interpretability_evolution.png
│   ├── math_combined.png
│   ├── math.png
│   ├── minimal_feynman_kan.pdf
│   ├── minimal_special_kan.pdf
│   ├── mobility_edge.png
│   ├── model_scale_exp100d.pdf
│   ├── model_scale_xy.pdf
│   ├── model_scaling_4.pdf
│   ├── model_scaling_toy.pdf
│   ├── model_scaling.pdf
│   ├── mosaic_results.png
│   ├── PDE_results.pdf
│   ├── siren.png
│   ├── special_pf.pdf
│   ├── spline_notation.png
│   ├── sr.png
│   ├── theory_model.png
│   ├── toy_interpretability_evolution.png
│   └── unsupervised_toy.png
├── kan.bbl
├── kan.tex
├── neurips_2023.sty
├── neurips_2024.sty
└── ref.bib
```

## Lecture Bridge Context


Here is a proposal for the "Deep Dive on KANs" conference tracks. These topics are designed to bridge the gap between your specific lecture slides and the cutting-edge KAN paper, suitable for graduate students.

### Conference Track 1: Architectural Foundations & Optimization
*Focus: Contrasting the mathematical and structural underpinnings of KANs against the standard MLP and optimization techniques covered in Foundations I & II.*

**Topic 1.1: The Battle of Theorems: UAT vs. KAT**
*   **Abstract:** The lecture `foundations_1_reduced_bib.pdf` introduced the Universal Approximation Theorem (UAT) as the theoretical justification for MLPs,. KANs challenge this by utilizing the Kolmogorov-Arnold Representation Theorem (KAT), shifting learnable non-linearities from nodes to edges,. This talk will mathematically compare UAT and KAT, explaining why KAT was historically discarded for being "pathological" and how the authors’ use of B-splines and arbitrary depth solved this.
*   **Lecture Connection:** Foundations I (History of Neural Networks, UAT); Foundations II (Activation Functions like ReLU/Tanh).
*   **Deep Dive:** Analyze the implications of moving from fixed activation functions (ReLU/Sigmoid) to learnable spline-based activations.

**Topic 1.2: Optimization Dynamics: Grid Extension vs. Learning Rate Schedules**
*   **Abstract:** In `foundations_2_bib.pdf`, we explored Gradient Descent, He-Initialization, and Learning Rate Schedules (e.g., Cosine Annealing) to navigate loss landscapes,. KANs introduce a novel optimization technique called "Grid Extension," where the B-spline grid is fine-grained during training to avoid local minima and improve accuracy,. This talk will compare Grid Extension to traditional Learning Rate Schedules and discuss how the KAN loss landscape differs from the MLP landscape described in the lecture (e.g., saddle points vs. bad valleys),.
*   **Lecture Connection:** Foundations II (Gradient Descent Challenges, Initialization, LR Schedules).
*   **Deep Dive:** How does KAN's "fine-graining" relate to the concept of "overspecification" and "double descent" mentioned in `foundations_3_bib.pdf`?

---

### Track 2: Scalability & Modern Architectures
*Focus: Integrating KANs into the advanced architectures discussed in the CNN, ResNet, and Transformer lectures.*

**Topic 2.1: ConvKANs: Rethinking the Filter Kernel**
*   **Abstract:** The lecture `arch_cnn_bib.pdf` defines the convolution operation as a linear dot product between a kernel $w$ and an input region $x$,. KANs propose replacing linear weights with non-linear functions $\phi(x)$. This topic explores the theoretical construction of a "Convolutional KAN." How does replacing the linear filter kernel with a KAN layer affect the concepts of translation invariance and equivariance discussed in the lecture?
*   **Lecture Connection:** CNNs (Filter kernels, Equivariance); ResNets (Bottleneck blocks).
*   **Deep Dive:** Propose a KAN-based residual block. Does the KAN architecture inherently solve the "shattered gradient" problem discussed in `arch_resnet_bib.pdf`, or does it still require skip connections?

**Topic 2.2: Kansformers: Breaking the Scaling Laws?**
*   **Abstract:** `arch_transformer_1_bib.pdf` introduces the Transformer, where MLPs constitute the bulk of the parameters. The KAN paper suggests that KANs scale faster ($N^{-4}$) than MLPs ($N^{-2}$). This presentation will analyze the feasibility of replacing Transformer MLP blocks with KAN layers ("Kansformers"). It must critically analyze the "Scaling Hypothesis" from the lecture against the "Fast Scaling Laws" claimed by the KAN authors.
*   **Lecture Connection:** Transformers I & II (Architecture, Scaling Laws, GPT-3); Foundations III (Bias-Variance Tradeoff).
*   **Deep Dive:** The lecture mentions that Transformers are efficient on GPUs. The KAN paper admits slow training due to poor batching. Analyze this trade-off: Is the theoretical scaling advantage of KANs worth the hardware inefficiency?

---

### Track 3: Science, Interpretability & Generative Models
*Focus: KANs as "White Box" models and their application in generative tasks vs. the "Black Box" nature of standard DL.*

**Topic 3.1: White-Box Deep Learning: Symbolic Regression vs. Saliency Maps**
*   **Abstract:** `arch_cnn_bib.pdf` discusses interpretability techniques like Saliency Maps and Grad-CAM to peak inside the "Black Box",. KANs offer a paradigm shift towards "White Box" learning, where the model *is* the formula. This talk will demonstrate how KANs use sparsification and pruning (related to regularization in `foundations_3_bib.pdf`) to discover symbolic physical laws, contrasting this with the "post-hoc" interpretability methods taught in the lecture.
*   **Lecture Connection:** CNNs (Interpretation/Grad-CAM); Foundations III (Regularization, Pruning).
*   **Deep Dive:** How does the KAN "Symbolification" step compare to the "Inductive Bias" concepts discussed in Model Selection?

**Topic 3.2: Generative KANs: From PDEs to Diffusion**
*   **Abstract:** `generative_bib.pdf` introduces Diffusion Models and the U-Net architecture used for denoising,. The KAN paper demonstrates high efficacy in solving Partial Differential Equations (PDEs). This topic explores the intersection: Can KANs replace the U-Net in Diffusion models? Since Diffusion is a differential equation process, KANs' superior ability to model derivatives might offer a more efficient generative process.
*   **Lecture Connection:** Generative Models (Diffusion, U-Nets, VAEs).
*   **Deep Dive:** Contrast the "Auto-Regressive" generation of GPT-3 `arch_transformer_2_bib.pdf` with the potential of KANs to learn continuous functions for generation.

---

### Track 4: The Sustainability & Efficiency Track
*Focus: Critical analysis of KANs through the lens of Sustainable AI.*

**Topic 4.1: The Efficiency Paradox: Parameter Count vs. FLOPs**
*   **Abstract:** `sus_ai.pdf` highlights that energy consumption is $Power \times Time$. While KANs are parameter efficient (requiring fewer parameters for the same accuracy), the paper admits they are 10x slower to train due to the lack of optimized CUDA kernels for B-splines. This talk will perform a critical sustainability audit of KANs. Do they solve the "Increased Computing Requirements" or exacerbate them?
*   **Lecture Connection:** Sustainable AI (Energy Demand, Efficient Models, Hardware usage).
*   **Deep Dive:** `sus_ai.pdf` suggests "Model Quantization". Can KANs be quantized? Since they rely on continuous spline functions, is 8-bit quantization even possible without destroying the grid accuracy?

**Topic 4.2: Catastrophic Forgetting & Continual Learning**
*   **Abstract:** `foundations_3_bib.pdf` discusses model training and the need for large datasets. The KAN paper claims KANs possess "local plasticity" due to the local nature of B-splines, allowing them to avoid catastrophic forgetting better than global MLPs,. This talk will explore if KANs could enable more sustainable "Lifelong Learning" models that don't require massive re-training cycles, addressing the "Rebound Effect" mentioned in the Sustainability lecture.
*   **Lecture Connection:** Foundations III (Training Process); Sustainable AI (Reducing compute needs).
*   **Deep Dive:** Analyze the "Locality" of B-splines versus the global receptivity of the standard MLP neurons discussed in Foundations I.

### Instructions for Students
*   **Preparation:** Review the specific slides cited in your chosen track alongside the relevant sections of the KAN paper.
*   **Presentation:** You must explicitly contrast the "Standard Deep Learning Way" (from the slides) with the "KAN Way" (from the paper).
*   **Critical Thinking:** Do not just present KANs as "better." Use the lecture knowledge (e.g., about GPU efficiency or Vanishing Gradients) to critique where KANs might fail.