
## HUSE &mdash; Human Unified with Statistical Evaluation

![Teaser image](./opening_fig.png)
**Picture:** *HUSE is twice the classification error of distinguishing reference and generated text represented as (human judgment, pmodel) pairs. This easily identifies samples with defects in both quality (Sharon has stroke . . .) and diversity (Cleared coach facing ...).*

This repository contains the ready to use evaluation methodology of the following paper:

> **Unifying Human and Statistical Evaluation for Natural Language Generation**<br>
> Tatsunori B. Hashimoto*, Hugh Zhang*, Percy Liang<br>
> *Equal contribution
>
> **Abstract:** *How can we measure whether a natural language generation system produces both high quality and diverse outputs?
Human evaluation captures quality but not diversity, as it does not catch models that simply plagiarize from the training set. On the other hand, statistical evaluation (i.e., perplexity) captures diversity but not quality, as models that occasionally emit low quality samples would be insufficiently penalized. In this paper, we propose a unified framework which evaluates both diversity and quality, based on the optimal error rate of predicting whether a sentence is human- or machine-generated. We demonstrate that this error rate can be efficiently estimated by combining human and statistical evaluation, using an evaluation metric which we call HUSE. On summarization and chit-chat dialogue, we show that HUSE detects diversity defects which fool pure human evaluation and that techniques such as annealing for improving quality actually decrease HUSE due to decreases in diversity.*

A simple working example is given with summarization_example.py. A full walkthrough of with the paper's Mechanical Turk is given in full_summarization_example.py.
