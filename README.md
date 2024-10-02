# Synthetic-data
This is the main repository of the paper: Maximizing the potential of synthetic data: Insights from Random Matrix Theory.

## Abstract
Synthetic data has gained attention for training large language models, but poor-quality data can harm performance. A potential solution is data pruning, which retains only high-quality data based on a score function (human or machine feedback). Previous work analyzed models trained on synthetic data as sample size increases.
Using random matrix theory, we generalize this analysis and derive the performance of a binary classifier trained on a mix of real and pruned synthetic data in a high dimensional setting. Our findings identify conditions where synthetic data could improve performance, focusing on the quality of the generative model and verification strategy. We also show a smooth phase transition in synthetic label noise, contrasting with prior works on sharp transition in infinite sample limits. Our extensive experimental setup validates our theoretical results.

## Paper figures:
All the figures in the paper and more can be found in the folders: 
* [results-plot](results-plot/): contains all the figures shown in the paper
* [study-plot](study-plot/) : for some additonal plots (that are not included in the paper)

## Reproducing figures:
* Figures 1 and 2 can be reproduced by running the notebook:[rmt_laws](rmt_laws.ipynb)
* Run the file [toy_setting](toy_setting.py) to get the plots of Figure 3.
* Run the file [phase_transition](phase_transition.py) to get the phase transition plot in Figure 4.
* Images in Figure 5 can be obtained through the notebook called [mnist](mnist.ipynb). 
* Run the file [train_amazon](train_amazon.py) to get the plots of Figure 6: set $n = 800$.
* Figure 7 can be found in notebook [mnist](mnist.ipynb) and the results are obtained by running the file [train_mnist](train_mnist.py).
* Figure 8 can be reproduced using the file [safety_score](safety_score.py). The numerical results are gotten using files in the folder [Safety-Alignement-experiment](Safety-Alignement-experiment/).
* Figure 9 along with the numerical results can be obtained through the folder [QA_synthetic_safety](QA_synthetic_safety/).
