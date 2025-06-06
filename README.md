# Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race
[Lihao Sun](https://sites.google.com/uchicago.edu/lihao-sun)<sup>1</sup>,
[Chengzhi Mao](https://www.cs.columbia.edu/~mcz/)<sup>2</sup>,
[Valentin Hofmann](https://valentinhofmann.github.io/)<sup>3,4</sup>,
[Xuechunzi Bai](https://www.xuechunzibai.com/)<sup>1</sup>

<sup>1</sup>University of Chicago,
<sup>2</sup>Rutgers University,
<sup>3</sup>Allen Institute for AI,
<sup>4</sup>University of Washington

[Paper Link](https://arxiv.org/abs/2506.00253)

Accepted to ACL 2025 Main Conference

---
### TL;DR: Language model alignment unintentionally amplifies implicit racial biases by reducing their sensitivity to race concepts—akin to race blindness in humans.  

## Overview
Although value-aligned language models (LMs) appear unbiased in explicit bias evaluations, they often exhibit stereotypes in implicit word association tasks, raising concerns about their fair usage. We investigate the mechanisms behind this discrepancy and find that alignment surprisingly amplifies implicit bias in model outputs. Specifically, we show that aligned LMs, unlike their unaligned counterparts, overlook racial concepts in early internal representations when the context is ambiguous. Not representing race likely fails to activate safety guardrails, leading to unintended biases. Inspired by this insight, we propose a new bias mitigation strategy that works by incentivizing the representation of racial concepts in the early model layers. In contrast to conventional mitigation methods of machine unlearning, our interventions find that steering the model to be more aware of racial concepts effectively mitigates implicit bias. Similar to race blindness in humans, ignoring racial nuances can inadvertently perpetuate subtle biases in LMs.

![Figure1](figures/selfie.jpg)

## Using this directory
Corresponding to our experiment structure, the directory is organized into three main components: behavioral (prompt construction), mechanistic interpretability (SelfIE + activation patching), and intervention (activation engineering + LoRA). The SelfIE component requires code from [SelfIE codebase](https://github.com/tonychenxyz/selfie). The main experiments are undertaken with Llama 3 8B and 70B. All results are reproducible using deterministic generation. 

## A quick walkthrough of our findings
### Behavioral
> 📁 Relevant files: /behavioral

![Figure2](figures/behavioral.jpg)

We investigate racial bias in LMs—specifically, the tendency to associate black with negative words and white with positive ones. Bias is tested in two ways:
- Explicit: Likert scale, asking whether the model agrees with a given association such as black is related to negative, white is related to positive.
- Implicit: Word association, let the model freely pair black/white with positive/negative words.

We evaluate both base and aligned models and find:

*While alignment removes explicit bias, it amplifies implicit racial bias*—especially in ambiguous contexts where black and white can refer to either color or race:
- Alignment removes explicit bias: near-zero agreement with harmful stereotypes statements. 
- But it amplifies implicit bias: in over 90% of cases, aligned models associate “black” with negative concepts.

### Mechanistic Insights
#### Activation Patching
> 📁 Relevant files: /mech_interp/activation_patching

![Figure3](figures/activation_patching.jpg)

To explain this behavior, we use activation patching to test whether LMs represent black/white as race or color in ambiguous settings.
- Aligned models are 52.2% less likely to represent race internally.
- This "race blindness" explains why guardrails fail to activate during generation.

#### SelfIE
To explore whether stronger associations beyond the race/color binary might be present, we applied [SelfIE](https://github.com/tonychenxyz/selfie)—an open-ended natural language interpretation method that translates internal embeddings into text.
- SelfIE interpretations of white/black fell into three categories: color, race, or nonsensical outputs (e.g., repeating the instruction)
- Consistent with our activation patching results, the aligned model produced 74.4% fewer race-related interpretations than the base model on implicit prompts. 

### Causal Intevention: Strengthening Race Associations

![Figure4](figures/intervention.jpg)

#### Embedding Intervention via Steering
> 📁 Relevant files: /interventions/activation_steering

Injecting race-aware activations (from disambiguated prompts like “Race: black and white”) reduces implicit bias by 26–42%, especially when applied to early layers. 

#### Weight Intervention via LoRA Fine-tuning
> 📁 Relevant files: /interventions/lora

We fine-tune models using LoRA on prompts where black/white are used ambiguously but labeled with racial meaning. 

Targeted early-layer fine-tuning cuts implicit bias from 97.3% to 42.4% and also reduces explicit bias. 
