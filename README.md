# COVID-19 Xray Study using CNN from deeplearning4J library.

Using a Convolution Neural Network to examine Covid19 Xrays. Over the last year, I have been working on machine learning with Python.
Like everyone else, I have been following the Covid19 news as well. In Kaggle, I ran across a Python Notebook that examined Xrays of Covid19.
Further research turned up more Python projects such as https://github.com/UCSD-AI4H/COVID-CT. I wanted to do something similar, so I found
a java library called Deeplearning4J. And decided to create a CNN model to determine if an xray is covid19 or not.
This is research only, which it was. The answer is the model was able to determine the difference... 
Accuracy:        0.7143
Precision:       0.7500
Recall:          0.7500
F1 Score:        0.7500  

==================================================================

c.r.o.ResearchXrayCovid19 - ***Fit called on training data.
c.r.o.ResearchXrayCovid19 - 
Example that is labeled NonCOVID the model predicted NonCOVID

saw some 85% too but not consistent
Accuracy:        0.8571
 Precision:       0.8000
 Recall:          1.0000
 F1 Score:        0.8889
Precision, recall & F1: reported for positive class (class 1 - "1") only



images from https://github.com/UCSD-AI4H/COVID-CT


CITE:

@Article{he2020sample,
  author  = {He, Xuehai and Yang, Xingyi and Zhang, Shanghang, and Zhao, Jinyu and Zhang, Yichen and Xing, Eric, and Xie,       Pengtao},
  title   = {Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans},
  journal = {medrxiv},
  year    = {2020},

}
@article{zhang2020artificial,
  title={Artificial Intelligence Distinguishes COVID-19 from Community Acquired Pneumonia on Chest CT},
  author={Li, Lin and Qin, Lixin and Xu, Zeguo and Yin, Youbing and Wang, Xin and Kong, Bin and Bai, Junjie and Lu, Yi and Fang, Zhenghan and Song, Qi and Cao, Kunlin and others},
  journal={Radiology},
  year={2020}
}
https://pubs.rsna.org/doi/10.1148/radiol.2020200905
https://deeplearning4j.konduit.ai/models/multilayernetwork

