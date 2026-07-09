# PDF Text Extraction Summary

## 07575-KimH.pdf
- Pages: 9
- Characters extracted: 46774
- Opening text: Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation HyunGi Kim1, Siwon Kim1, Jisoo Mok1, Sungroh Yoon1, 2, 3 † 1Department of Electrical and Computer Engineering, Seoul National University 2Interdisciplinary Program in Artificial Intelligence, Seoul National University 3AIIS, ASRI, and INMC, Seoul National University rlagusrl0128@snu.ac.kr, tuslkk17@gmail.com, magicshop1118@snu.ac.kr, sryoon@snu.ac.kr Abstract Deep Neural Networks have spearheaded remarkable ad- vancements in time series forecasting (TSF), one of the ma- jor tasks in time series modeling. Nonetheless, the non- stationarity of time series undermines the reliability of pre- trained source time series forecasters in mission-critical de- ployment settings. In this study, we introduce a pioneering test-time adaptation framework tailored for TSF (TSF-TTA). TAFAS, the proposed approach to TSF-TTA, 

## 2506.23424v1.pdf
- Pages: 11
- Characters extracted: 35083
- Opening text: arXiv:2506.23424v1 [cs.LG] 29 Jun 2025 Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting Heitor R. Medeiros * 1 2 Hossein Sharifi-Noghabi 1 Gabriel L. Oliveira 1 Saghar Irandoust 1 Abstract Real-world time series often exhibit a non- stationary nature, degrading the performance of pre-trained forecasting models. Test-Time Adap- tation (TTA) addresses this by adjusting models during inference, but existing methods typically update the full model, increasing memory and compute costs. We propose PETSA, a parameter- efficient method that adapts forecasters at test time by only updating small calibration modules on the input and output. PETSA uses low-rank adapters and dynamic gating to adjust representa- tions without retraining. To maintain accuracy de- spite limited adaptation capacity, we introduce a specialized loss combining three components: (1) a robust ter

## 3637528.3671926 (1).pdf
- Pages: 12
- Characters extracted: 72312
- Opening text: Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift Mouxiang Chen∗ Zhejiang University Hangzhou, China chenmx@zju.edu.cn Lefei Shen∗ Zhejiang University Hangzhou, China lefeishen@zju.edu.cn Han Fu Zhejiang University Hangzhou, China 11821003@zju.edu.cn Zhuo Li† State Street Technology (Zhejiang) Ltd. Hangzhou, China lizhuo@zju.edu.cn Jianling Sun Zhejiang University Hangzhou, China sunjl@zju.edu.cn Chenghao Liu† Salesforce Research Asia Singapore chenghao.liu@salesforce.com ABSTRACT Recent years have witnessed the success of introducing deep learning models to time series forecasting. From a data generation perspective, we illustrate that existing models are susceptible to distribution shifts driven by temporal contexts, whether observed or unobserved. Such context-driven distribution shift (CDS) in- troduces biases in predictions within speci

## 3637528.3671926.pdf
- Pages: 12
- Characters extracted: 72312
- Opening text: Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift Mouxiang Chen∗ Zhejiang University Hangzhou, China chenmx@zju.edu.cn Lefei Shen∗ Zhejiang University Hangzhou, China lefeishen@zju.edu.cn Han Fu Zhejiang University Hangzhou, China 11821003@zju.edu.cn Zhuo Li† State Street Technology (Zhejiang) Ltd. Hangzhou, China lizhuo@zju.edu.cn Jianling Sun Zhejiang University Hangzhou, China sunjl@zju.edu.cn Chenghao Liu† Salesforce Research Asia Singapore chenghao.liu@salesforce.com ABSTRACT Recent years have witnessed the success of introducing deep learning models to time series forecasting. From a data generation perspective, we illustrate that existing models are susceptible to distribution shifts driven by temporal contexts, whether observed or unobserved. Such context-driven distribution shift (CDS) in- troduces biases in predictions within speci

## 3690624.3709210.pdf
- Pages: 12
- Characters extracted: 74606
- Opening text: Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting Lifan Zhao Shanghai Jiao Tong University Shanghai, China mogician233@sjtu.edu.cn Yanyan Shen Shanghai Jiao Tong University Shanghai, China shenyy@sjtu.edu.cn Abstract Time series forecasting always faces the challenge of concept drift, where data distributions evolve over time, leading to a decline in forecast model performance. Existing solutions are based on online learning, which continually organize recent time series ob- servations as new training samples and update model parameters according to the forecasting feedback on recent data. However, they overlook a critical issue: obtaining ground-truth future values of each sample should be delayed until after the forecast horizon. This delay creates a temporal gap between the training samples and the test sample. Our empirical analysis reveals that the 

## 4690_Online_time_series_predic (1).pdf
- Pages: 27
- Characters extracted: 96476
- Opening text: Published as a conference paper at ICLR 2026 ONLINE TIME SERIES PREDICTION USING FEATURE ADJUSTMENT Xiannan Huang College of Transportation Tongji University Shanghai, 201804, China huang xn@tongji.edu.cn Shuhan Qiu College of Transportation Tongji University Shanghai, 201804, China qiusuan@tongji.edu.cn Jiayuan Du College of Computer Science Tongji University Shanghai, 201804, China dujiayuan@tongji.edu.cn Chao Yang ∗ College of Transportation Tongji University Shanghai, 201804, China tongjiyc@tongji.edu.cn ABSTRACT Time series forecasting is of significant importance across various domains. How- ever, it faces significant challenges due to distribution shift. This issue becomes particularly pronounced in online deployment scenarios where data arrives se- quentially, requiring models to adapt continually to evolving patterns. Current time series online learning methods focus on two main

## ICLR-2025-fast-and-slow-streams-for-online-time-series-forecasting-without-information-leakage-Paper-Conference.pdf
- Pages: 27
- Characters extracted: 87812
- Opening text: Published as a conference paper at ICLR 2025 FAST AND SLOW STREAMS FOR ONLINE TIME SERIES FORECASTING WITHOUT INFORMATION LEAKAGE Ying-yee Ava Lau, Zhiwen Shao, Dit-Yan Yeung Department of Computer Science and Engineering The Hong Kong University of Science and Technology yyalau@connect.ust.hk, zhiwen@ust.hk, dyyeung@cse.ust.hk ABSTRACT Current research in online time series forecasting (OTSF) faces two significant issues. The first is information leakage, where models make predictions and are then evaluated on historical time steps that have already been used in backprop- agation for parameter updates. The second is practicality: while forecasting in real-world applications typically emphasizes looking ahead and anticipating fu- ture uncertainties, prediction sequences in this setting include only one future step with the remaining being observed time points. This necessitates a redefin

## lee25ag.pdf
- Pages: 29
- Characters extracted: 108062
- Opening text: Lightweight Online Adaption for Time Series Foundation Model Forecasts Thomas L. Lee * 1 2 William Toner* 3 Rajkarn Singh 3 Artjom Joosen 3 Martin Asenov 3 Abstract Foundation models (FMs) have emerged as a promising approach for time series forecasting. While effective, FMs typically remain fixed dur- ing deployment due to the high computational costs of learning them online. Consequently, de- ployed FMs fail to adapt their forecasts to cur- rent data characteristics, despite the availability of online feedback from newly arriving data. This raises the question of whether FM performance can be enhanced by theefficient usage of this feed- back. We propose ELF to answer this question. ELF is a lightweight mechanism for the online adaption of FM forecasts in response to online feedback. ELF consists of two parts: a) the ELF- Forecaster which is used to learn the current data distribution; 
