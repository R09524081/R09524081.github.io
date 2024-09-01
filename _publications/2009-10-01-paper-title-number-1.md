---
title: "BEACON: A Bayesian Optimization Strategy for Novelty Search in Expensive Black-Box Systems"
collection: publications
category: manuscripts
permalink: /publication/2009-10-01-paper-title-number-1
excerpt: 'This paper is about the number 1. The number 2 is left for future work.'
date: 2024
venue: 'preprint'
slidesurl: 'https://arxiv.org/pdf/2406.03616'
paperurl: 'https://arxiv.org/pdf/2406.03616'
citation: 'Tang, W. T., Chakrabarty, A., & Paulson, J. A. (2024). BEACON: A Bayesian Optimization Strategy for Novelty Search in Expensive Black-Box Systems. arXiv preprint arXiv:2406.03616.'
---

Novelty search (NS) refers to a class of exploration algorithms that automatically uncover diverse system behaviors through simulations or experiments. Systematically obtaining diverse outcomes is a key component in many real-world design problems such as material and drug discovery, neural architecture search, reinforcement learning, and robot navigation. Since the relationship between the inputs and outputs (i.e., behaviors) of these complex systems is typically not available in closed form, NS requires a black-box perspective. Consequently, popular NS algorithms rely on evolutionary optimization and other meta-heuristics that require intensive sampling of the input space, which is impractical when the system is expensive to evaluate. We propose a Bayesian optimization inspired algorithm for sample-efficient NS that is specifically designed for such expensive black-box systems. Our approach models the input-to-behavior mapping with multi-output Gaussian processes (MOGP) and selects the next point to evaluate by maximizing a novelty metric that depends on a posterior sample drawn from the MOGP that promotes both exploration and exploitation. By leveraging advances in efficient posterior sampling and high-dimensional Gaussian process modeling, we discuss how our approach can be made scalable with respect to both amount of data and number of inputs. We test our approach on ten synthetic benchmark problems and eight real-world problems (with up to 2133 inputs) including new applications such as discovery of diverse metal organic frameworks for use in clean energy technology. We show that our approach greatly outperforms existing NS algorithms by finding substantially larger sets of diverse behaviors under limited sample budgets.
