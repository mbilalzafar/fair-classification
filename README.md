# Fairness in Classification

 
This repository provides a logistic regression implementation in python for the fair classification mechanisms introduced in our <a href="http://arxiv.org/abs/1507.05259" target="_blank">AISTATS'17</a>, <a href="https://arxiv.org/abs/1610.08452" target="_blank">WWW'17</a> and <a href="https://arxiv.org/abs/1707.00010" target="_blank">NIPS'17</a> papers.

Specifically:

1. The <a href="http://arxiv.org/abs/1507.05259" target="_blank">AISTATS'17 paper</a> [1]  proposes mechanisms to make classification outcomes free of disparate impact, that is, to ensure that similar fractions of people from different demographic groups (e.g., males, females) are accepted (or classified as positive) by the classifier. More discussion about the disparate impact notion can be found in Sections 1 and 2 of the paper.


2. The <a href="https://arxiv.org/abs/1610.08452" target="_blank">WWW'17 paper</a> [2]  focuses on making classification outcomes free of disparate mistreatment, that is, to ensure that the misclassification rates for different demographic groups are similar. We discuss this fairness notion in detail, and contrast it to the disparate impact notion, in Sections 1, 2 and 3 of the paper.


2. The <a href="https://arxiv.org/abs/1707.00010" target="_blank">NIPS'17 paper</a> [3]  focuses on making classification outcomes adhere to preferred treatment and preferred impact fairness criteria. For more details on these fairness notions, and how they compare to existing fairness notions used in ML, see Sections 1 and 2 of the paper.

#### Dependencies 
1. [numpy, scipy](https://www.scipy.org/scipylib/download.html) and [matplotlib](http://matplotlib.org/) if you are only using the mechanisms introduced in [1].
2. Additionally, if you are using the mechanisms introduced in [2] and [3], then you also need to install [CVXPY](https://github.com/cvxgrp/cvxpy) and [DCCP](https://github.com/cvxgrp/dccp).

#### Using the code

1. If you want to use the code related to [1], please navigate to the directory "disparate_impact".
2. If you want to use the code related to [2], please navigate to the directory "disparate_mistreatment".
2. If you want to use the code related to [3], please navigate to the directory "preferential_fairness".

Please cite the corresponding paper when using the code.

#### References
1. <a href="http://arxiv.org/abs/1507.05259" target="_blank">Fairness Constraints: Mechanisms for Fair Classification</a> <br>
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi. <br>
20th International Conference on Artificial Intelligence and Statistics (AISTATS), Fort Lauderdale, FL, April 2017.
 
 
2. <a href="https://arxiv.org/abs/1610.08452" target="_blank">Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment</a> <br>
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi. <br>
26th International World Wide Web Conference (WWW), Perth, Australia, April 2017.


3. <a href="https://arxiv.org/abs/1707.00010" target="_blank">From Parity to Preference-based Notions of Fairness in Classification</a> <br>
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi, Adrian Weller. <br>
31st Conference on Neural Information Processing Systems (NIPS), Long Beach, CA, December 2017.