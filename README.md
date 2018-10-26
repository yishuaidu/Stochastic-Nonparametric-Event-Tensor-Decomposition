
### [Stochastic Nonparametric Event Tensor Decomposition](https://nips.cc/Conferences/2018/Schedule?showEvent=11661) @ NIPS'2018
 [Shandian Zhe](http://www.cs.utah.edu/~zhe/)   [Yishuai Du](https://www.linkedin.com/in/yishuai-du-583a17b5/)

## Requirement
* MIT Licence
* [**Matlab**](https://www.mathworks.com/products/matlab.html) as the software to run our code

## How to run 
1. Download Stochastic-Nonparametric-Event-Tensor-Decomposition repository
2. For our(rfp-hp), open Matlab, run the  ### "Test" +"_file_name_" + "hybrid.m" ### file for getting models
and ### "Test_data_name_hybrid_more.m" ### for testing log likelihood.
3. For CP-PTF(cp-pp), open Matlab, run the  ### "test" +"_file_name_" +"v2".m" file for getting models
and ### "test" + "_file_name_" + "more".m" ### for testing log likelihood.
4. For CPT-PTF(cpmarkov-pp), open Matlab, run the  ### "test" +"_file_name_" +"v2".m" file for getting models
and ### "test" + "_file_name_" + "more".m" ### for testing log likelihood.
5. For GP-PTF(rfp-pp), open Matlab, run the  ### "test" +"_file_name_" +"v2".m" file for getting models
and ### "test" + "_file_name_" + "more".m" ### for testing log likelihood.
6. For ./data/process-code, "preprocess_file_name_data_hybrid.m" for traning data ,  "preprocess_file_name_data_hybrid_test_more.m" for testing data and "process_file_name_data_hybrid_test_for_baselines.m" for baselines testing data.



## Dataset Intro
There are 5 datasets:
* [Article](https://www.kaggle.com/gspmoreira), The Article data are 12 month logs (Mar. 2016 - Feb. 2017) of CI&T’s Internal Communi-cation platform (DeskDrop). It records users’ operations onthe shared articles of the platform, such as LIKE, FOLLOW
and BOOKMARK. We extracted a three mode event-tensor(user, operation, session id), of size 1895x5x2987.There
are 50,938 entries observed to have events. The length of the longest event sequence in all the entries is 76. The total number of events is 72, 312.
* [UFO](https://www.kaggle.com/NUFORC), The UFO data consist of reported UFO sightings over the last century in the world.
From this dataset, we extracted a two mode event-tensor (UFO shape, city) , of size 28x19,408, with 45, 045 entries observed to have sighting events. The longest event sequence length is 113. There are in total 77,747 events.
* [911](https://www.kaggle.com/mchirico), The 911 data record the emergency (911) calls from 2015-12-10 to 2017-04-10 in Montgomery County, PA. We focused on the Emergence Medical Service (EMS) calls and extracted a two mode even-tensor (EMS title, township), which is 72x69. There are 2,494 entries observed to have events. The length of the longest event sequence is 545. The total number of events are 59,270.











NIPS 2018 

