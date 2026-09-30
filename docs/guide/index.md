---
layout: docs
title: Guide
nav_order: 2
has_children: true
permalink: /guide/
---

# Guide

Step-by-step documentation for integrating AuraLocal into your app.

## On-device ML, not just LLMs

[On-device ML tools]({{ '/guide/ml-tools' | relative_url }}) covers the Vision, NaturalLanguage, SoundAnalysis, Core ML and
Create ML tools in `AuraCore`: what each one does, where it runs, Swift and `aura ml` examples, a
train → ship → classify walkthrough, and how to run your own Core ML model.

## Will this Hugging Face model run?

[Finding compatible models]({{ '/guide/models' | relative_url }}#finding-compatible-models) checks any Hugging
Face repo against the runtimes AuraLocal pins (mlx-swift-lm 3.31.3, llama.cpp b8851) and a device's memory
budget without downloading the weights: from Swift (`ModelCompatibilityChecker`), the CLI (`aura models`) or
the Model Finder example app.

## Research: distributed inference over USB

[Distributed Inference over USB (research)]({{ '/guide/distributed' | relative_url }}) records an evaluation of
a Mac orchestrating USB-tethered iPhones for speculative decoding and embeddings: measured results, transport
and iOS findings, and what AuraLocal would need. No mesh API ships; the only piece that did is the
multilingual-e5-small embedding provider.
