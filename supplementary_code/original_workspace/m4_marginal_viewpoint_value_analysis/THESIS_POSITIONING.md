# Thesis Positioning For Marginal Viewpoint Value

## Core Research Question

This project can now be framed as:

`What is the marginal information value of additional UAV viewpoints for object detection, and at what point do extra views become mostly redundant rather than genuinely complementary?`

That is stronger than only asking whether:

- one angle is better than another, or
- multiview is better than single-view

because it asks for the shape of the gain curve and the structure of viewpoint complementarity.

## Suggested Thesis Structure

### 1. Single-View Information Landscape

Question:

`Which viewpoints are individually strong or weak?`

Role:

- establishes the baseline viewpoint bias of the detector;
- shows that viewpoint is a central determinant of performance.

### 2. Multi-View Gain And Diminishing Returns

Question:

`How much do 2 views help over 1, and how much does a third help over the best 2-view setup?`

Role:

- shifts the analysis from "multiview beats single-view" to a gain curve;
- supports practical statements about rational swarm size.
- gives a direct answer to the `number of angles` question.

Suggested headline:

`The practically relevant quantity is not only the maximum achievable performance, but the smallest number of views required to capture most of the available multiview gain.`

### 2.5 Size-Conditioned Shapley Progression

Question:

`What is the exact marginal value of the 2nd, 3rd, 4th, ... added angle, and which angle is best at each swarm size?`

Role:

- connects the supervisor's `number of angles` framing to the exact fusion-based Shapley game;
- separates `how many angles are worth adding?` from `which angle is the best teammate at size k?`;
- makes coalition growth and Shapley progression readable in the same framework.

Suggested headline:

`Exact Shapley can be decomposed by coalition size, so swarm-size planning and viewpoint selection become two views of the same multiview game.`

### 3. Complementarity Instead Of Standalone Strength

Question:

`Which viewpoints add information that the others miss?`

Role:

- separates strong standalone views from useful additional views;
- links naturally to agent selection and next-best-view ideas.

Suggested headline:

`A viewpoint is valuable not only because it performs well in isolation, but because it contributes information that is absent from the already selected viewpoints.`

## Recommended Main Claim

`This thesis operationalises the marginal information value of additional UAV viewpoints for object detection. Rather than only comparing single-view and multi-view performance, it measures how detection quality changes as extra views are added, identifies the point of diminishing returns, and characterises which viewpoint combinations are most complementary.`

With the new size-conditioned Shapley view, this can be tightened further to:

`The thesis first asks how many viewpoints are worth adding, and only then which viewpoints are the most valuable teammates at each swarm size.`

## Practical Question This Folder Helps Answer

`How many drones or viewpoints are needed to recover 90% or 95% of the available multiview gain?`

That is a much stronger operational result than:

`these are the three best angles.`

## Bridge To Real Drone Imagery

Before trusted angle estimates exist, the first real-data question should be:

`What is the marginal contribution of adding a 2nd, 3rd, or 4th image from the same target episode?`

That gives a count-first coalition analysis that is already compatible with the
supervisor's new framing. Only after that should the thesis move to the refined
angle-aware question:

`Given estimated or known geometry, which angle is the most valuable teammate?`
