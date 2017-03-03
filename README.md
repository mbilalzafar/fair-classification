# Fairness in Classification

 
This repository provides a logistic regression implementation in python for the fair classification mechanisms introduced in our [AISTATS'17](http://arxiv.org/abs/1507.05259) and [WWW'17](https://arxiv.org/abs/1610.08452) papers.

Specifically: 
1. The [AISTATS'17 paper](http://arxiv.org/abs/1507.05259)  proposes mechanisms to make classification outcomes free of disparate impact, that is, to ensure that similar fractions of people from different social groups (e.g., males, females) are accepted by the classifier. More discussion about the disparate impact notion can be found in Sections 1 and 2 of the paper.


2. The [WWW'17 paper](https://arxiv.org/abs/1610.08452)  focuses on how to make classification outcomes free of disparate mistreatment, that is, to ensure that the misclassification rates for different social groups are similar. We discuss this fairness notion in detail in Sections 1, 2 and 3 of the paper.

Please cite the papers when using the code.

#### Dependencies 
1. [numpy, scipy](https://www.scipy.org/scipylib/download.html) and [matplotlib](http://matplotlib.org/) if you are only using the mechanisms introduced in our AISTATS'17 paper.
2. Additionally, if you are using the mechanisms used in our WWW'17 paper, then you also need to install [CVXPY](https://github.com/cvxgrp/cvxpy) and [DCCP](https://github.com/cvxgrp/dccp).

#### Using the code

1. If you want to use the code related to [1], please navigate to the directory "disparate _impact".
2. If you want to use the code related to [2], please navigate to the directory "disparate_mistreatment".

#### References
1. **Fairness Constraints: Mechanisms for Fair Classification**. <br>
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi. <br>
20th International Conference on Artifical Intelligence and Statistics (AISTATS), Fort Lauderdale, FL, April 2017.
 
 
2. **Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment**. <br>
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi. <br>
26th International World Wide Web Conference (WWW), Perth, Australia, April 2017.