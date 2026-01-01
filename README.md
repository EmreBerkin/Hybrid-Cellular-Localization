# Hybrid-Cellular-Localization
### Summary
A high-precision hybrid localization system that integrates Machine Learning (ML) models with a Hybrid Particle Filter (PF) and Hybrid Unscented Kalman Filter(UKF) to track user movement using cellular signal metrics (RSRP, RSRQ, and SINR). This project demonstrates the fusion of statistical filtering and predictive modeling for robust outdoor positioning.

### Abstract
Traditional GPS-based positioning systems can experience problems in enclosed spaces, tunnels, or dense urban environments due to signal interruptions, path loss, and environmental factors. Furthermore, GPS systems are not energy efficient and consume a lot of energy. This study aims to achieve location estimation with high accuracy and lower energy consumption, even in environments without GPS access, using the existing GSM infrastructure.

### Data Acquisition and Sampling
In this project, time-dependent RSRP, RSRQ, and SINR measurements of a mobile user were collected using a nonuniform sampling scheme, where measurements are not obtained at equal time intervals. Due to the nonuniform nature of the sampling process, hybrid filtering techniques were preferred in order to ensure stable and reliable tracking performance.

### Hybrid Filtering Approach
To handle irregular time steps and abrupt variations in the signal measurements, Hybrid Particle Filter (PF) and Hybrid Unscented Kalman Filter (UKF) structures were employed. These hybrid filters improve robustness by adapting the prediction and update stages to nonuniform temporal sampling.

### Machine Learning Model Training

Following the filtering stage, the collected signal measurements were divided into training and testing sets, where 70% of the data was used for training the machine learning models. The training set was used to fit the following four regression-based models:

* Random Forest Regressor
* Decision Tree Regressor
* XGBoost Regressor
* Gradient Boosting Regressor
