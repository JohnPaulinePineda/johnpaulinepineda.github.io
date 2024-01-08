## Data Science Project Portfolio

---

### Machine Learning Case Studies

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#)

[Supervised Learning : Learning Hierarchical Features for Predicting Multiclass X-Ray Images using Convolutional Neural Network Model Variations](https://johnpaulinepineda.github.io/Portfolio_Project_44/)
<br><br>
The integration of artificial intelligence (AI) into healthcare has emerged as a transformative force revolutionizing diagnostics and treatment. The urgency of the COVID-19 pandemic has underscored the critical need for rapid and accurate diagnostic tools. One such innovation that holds immense promise is the development of AI prediction models for classifying medical images in respiratory health. This [case study](https://johnpaulinepineda.github.io/Portfolio_Project_44/) aims to develop multiple convolutional neural network (CNN) classification models that could automatically learn hierarchical features directly from raw pixel data of x-ray images (categorized as Normal, Viral Pneumonia, and COVID-19), while delivering accurate predictions when applied to new unseen data. Data quality assessment was conducted on the initial dataset to identify and remove cases noted with irregularities, in addition to the subsequent preprocessing operations to improve generalization and reduce sensitivity to variations most suitable for the downstream analysis. Multiple CNN models were developed with various combinations of regularization techniques namely, Dropout for preventing overfitting by randomly dropping out neurons during training, and Batch Normalization for standardizing the input of each layer to stabilize and accelerate training. CNN With No Regularization, CNN With Dropout Regularization, CNN With Batch Normalization Regularization, and CNN With Dropout and Batch Normalization Regularization were formulated to discover hierarchical and spatial representations for image category prediction. Epoch training was optimized through internal resampling validation using Split-Sample Holdout with F1 Score used as the primary performance metric among Precision and Recall. All candidate models were compared based on internal and external validation performance. Post-hoc exploration of the model results involved Convolutional Layer Filter Visualization and Gradient Class Activation Mapping methods to highlight both low- and high-level features from the image objects that lead to the activation of the different image categories. These results helped provide insights into the important hierarchical and spatial representations for image category differentiation and model prediction.

<img src="images/CaseStudy5_Summary_1.png?raw=true"/>

<img src="images/CaseStudy5_Summary_2.png?raw=true"/>

<img src="images/CaseStudy5_Summary_3.png?raw=true"/>

<img src="images/CaseStudy5_Summary_4.png?raw=true"/>

<img src="images/CaseStudy5_Summary_5.png?raw=true"/>

<img src="images/CaseStudy5_Summary_6.png?raw=true"/>

---

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#)

[Unsupervised Learning : Discovering Global Patterns in Cancer Mortality Across Countries Via Clustering Analysis](https://johnpaulinepineda.github.io/Portfolio_Project_43/)
<br><br>
Age-standardized cancer mortality rates refer to the number of deaths attributed to cancer within a specific population over a given period, usually expressed as the number of deaths per 100,000 people adjusted for differences in age distribution. Monitoring cancer mortality rates allows public health authorities to track the burden of cancer, understand the prevalence of different cancer types and identify variations in different populations. Studying these metrics is essential for making accurate cross-country comparisons, identifying high-risk communities, informing public health policies, and supporting international efforts to address the global burden of cancer. This [case study](https://johnpaulinepineda.github.io/Portfolio_Project_43/) aims to develop a clustering model with an optimal number of clusters that could recognize patterns and relationships among cancer mortality rates across countries, allowing for a deeper understanding of the inherent and underlying data structure when evaluated against supplementary information on lifestyle factors and geolocation. Data quality assessment was conducted on the initial dataset to identify and remove cases noted with irregularities, in addition to the subsequent preprocessing operations most suitable for the downstream analysis. Multiple clustering modelling algorithms with various cluster counts were formulated using K-Means, Bisecting K-Means, Gaussian Mixture Model, Agglomerative and Ward Hierarchical methods. The best model with optimized hyperparameters from each algorithm was determined through internal resampling validation using 5-Fold Cross Validation using the Silhouette Score used as the primary performance metric. Due to the unsupervised learning nature of the analysis, all candidate models were compared based on internal validation and apparent performance. Post-hoc exploration of the model results involved clustering visualization methods using Pair Plots, Heat Maps and Geographic Maps - providing an intuitive method to investigate and understand the characteristics of the discovered cancer clusters. These findings aided in the formulation of insights on the relationship and association of the various descriptors for the clusters identified.

<img src="images/CaseStudy4_Summary_1.png?raw=true"/>

<img src="images/CaseStudy4_Summary_2.png?raw=true"/>

<img src="images/CaseStudy4_Summary_3.png?raw=true"/>

<img src="images/CaseStudy4_Summary_4.png?raw=true"/>

<img src="images/CaseStudy4_Summary_5.png?raw=true"/>

<img src="images/CaseStudy4_Summary_6.png?raw=true"/>

---

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#)

[Supervised Learning : Identifying Contributing Factors for Countries With High Cancer Rates Using Classification Algorithms With Class Imbalance Treatment](https://johnpaulinepineda.github.io/Portfolio_Project_42/)
<br><br>
Age-standardized cancer rates are measures used to compare cancer incidence between countries while accounting for differences in age distribution. They allow for a more accurate assessment of the relative risk of cancer across populations with diverse demographic and socio-economic characteristics - enabling a more nuanced understanding of the global burden of cancer and facilitating evidence-based public health interventions. This [case study](https://johnpaulinepineda.github.io/Portfolio_Project_42/) aims to develop an interpretable classification model which could provide robust and reliable predictions of belonging to a group of countries with high cancer rates from an optimal set of observations and predictors, while addressing class imbalance and delivering accurate predictions when applied to new unseen data. Data quality assessment and model-independent feature selection were conducted on the initial dataset to identify and remove cases or variables noted with irregularities, in adddition to the subsequent preprocessing operations most suitable for the downstream analysis. Multiple classification modelling algorithms with various hyperparameter combinations were formulated using Logistic Regression, Decision Tree, Random Forest and Support Vector Machine. Class imbalance treatment including Class Weights, Upsampling with Synthetic Minority Oversampling Technique (SMOTE) and Downsampling with Condensed Nearest Neighbors (CNN) were implemented. Ensemble Learning Using Model Stacking was additionally explored. Model performance among candidate models was compared through the F1 Score which was used as the primary performance metric (among Accuracy, Precision, Recall and Area Under the Receiver Operating Characterisng Curve (AUROC) measures); evaluated internally (using K-Fold Cross Validation) and externally (using an Independent Test Set). Post-hoc exploration of the model results to provide insights on the importance, contribution and effect of the various predictors to model prediction involved model-specific (Odds Ratios) and model-agnostic (Shapley Additive Explanations) methods.

<img src="images/CaseStudy3_Summary_1.png?raw=true"/>

<img src="images/CaseStudy3_Summary_2.png?raw=true"/>

<img src="images/CaseStudy3_Summary_3.png?raw=true"/>

<img src="images/CaseStudy3_Summary_4.png?raw=true"/>

<img src="images/CaseStudy3_Summary_5.png?raw=true"/>

<img src="images/CaseStudy3_Summary_6.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Characterizing Life Expectancy Drivers Across Countries Using Model-Agnostic Interpretation Methods for Black-Box Models](https://johnpaulinepineda.github.io/Portfolio_Project_35/)
<br><br>
Life expectancy is a statistical measure that represents the average number of years a person is expected to live from birth, assuming current mortality rates remain constant along the entire life course. It provides an estimation of the overall health and well-being of a population and is often reflective of the local conditions encompassing numerous factors including demographic, socio-economic, healthcare access and healthcare quality. This [case study](https://johnpaulinepineda.github.io/Portfolio_Project_35/) aims to develop an interpretable regression model which could provide robust and reliable estimates of life expectancy from an optimal set of observations and predictors, while delivering accurate predictions when applied to new unseen data. Data quality assessment and model-independent feature selection were conducted on the initial dataset to identify and remove cases or variables noted with irregularities, in adddition to the subsequent preprocessing operations most suitable for the downstream analysis. Multiple regression models with optimized hyperparameters were formulated using Stochastic Gradient Boosting, Cubist Regression, Neural Network, Random Forest, Linear Regression and Partial Least Squares Regression. Model performance among candidate models was compared using the Root Mean Square Error (RMSE) and R-Squared metrics, evaluated internally (using K-Fold Cross Validation) and externally (using an Independent Test Set). Post-hoc exploration of the model results to provide insights on the importance, contribution and effect of the various predictors to model prediction involved model agnostic methods including Dataset-Level Exploration using model-level global explanations (Permutated Mean Dropout Loss-Based Variable Importance, Partial Dependence Plots) and Instance-Level Exploration using prediction-level local explanations (Breakdown Plots, Shapley Additive Explanations, Ceteris Paribus Plots, Local Fidelity Plots, Local Stability Plots).

<img src="images/CaseStudy1_Summary_1.png?raw=true"/>

<img src="images/CaseStudy1_Summary_2.png?raw=true"/>

<img src="images/CaseStudy1_Summary_3.png?raw=true"/>

<img src="images/CaseStudy1_Summary_4.png?raw=true"/>

<img src="images/CaseStudy1_Summary_5.png?raw=true"/>

<img src="images/CaseStudy1_Summary_6.png?raw=true"/>

---

### Machine Learning Exploratory Projects

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#)

[Data Preprocessing : Data Quality Assessment, Preprocessing and Exploration for a Classification Modelling Problem](https://johnpaulinepineda.github.io/Portfolio_Project_41/)

This [project](https://johnpaulinepineda.github.io/Portfolio_Project_41/) explores the various methods in assessing data quality, implementing data preprocessing and conducting exploratory analysis for prediction problems with categorical responses. A non-exhaustive list of methods to detect missing data, extreme outlying points, near-zero variance, multicollinearity and skewed distributions were evaluated. Remedial procedures on addressing data quality issues including missing data imputation, centering and scaling transformation, shape transformation and outlier treatment were similarly considered, as applicable.

<img src="images/Project41_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#)

[Supervised Learning : Exploring Penalized Models for Predicting Numeric Responses](https://johnpaulinepineda.github.io/Portfolio_Project_40/)

This [project](https://johnpaulinepineda.github.io/Portfolio_Project_40/) explores the different penalized regression modelling procedures for numeric responses. Using the standard Linear Regression and Polynomial Regression structures as reference, models applied in the analysis which evaluate various penalties for over-confidence in the parameter estimates included the Ridge Regression, Least Absolute Shrinkage and Selection Operator Regression and Elastic Net Regression algorithms. The resulting predictions derived from the candidate models were assessed in terms of their model fit using the r-squared, mean squared error (MSE) and mean absolute error (MAE) metrics..

<img src="images/Project40_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/Python-black?logo=Python)](#) [![](https://img.shields.io/badge/Jupyter-black?logo=Jupyter)](#)

[Data Preprocessing : Data Quality Assessment, Preprocessing and Exploration for a Regression Modelling Problem](https://johnpaulinepineda.github.io/Portfolio_Project_39/)

This [project](https://johnpaulinepineda.github.io/Portfolio_Project_39/) explores the various methods in assessing data quality, implementing data preprocessing and conducting exploratory analysis for prediction problems with numeric responses. A non-exhaustive list of methods to detect missing data, extreme outlying points, near-zero variance, multicollinearity and skewed distributions were evaluated. Remedial procedures on addressing data quality issues including missing data imputation, centering and scaling transformation, shape transformation and outlier treatment were similarly considered, as applicable.

<img src="images/Project39_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Exploring Boosting, Bagging and Stacking Algorithms for Ensemble Learning](https://johnpaulinepineda.github.io/Portfolio_Project_38/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_38/) explores different ensemble learning approaches which combine the predictions from multiple models in an effort to achieve better predictive performance. The ensemble frameworks applied in the analysis were grouped into three classes including boosting models which add ensemble members sequentially that correct the predictions made by prior models and outputs a weighted average of the predictions; bagging models which fit many decision trees on different samples of the same dataset and averaging the predictions; and stacking which consolidate many different models types on the same data and using another model to learn how to best combine the predictions. Boosting models included the Adaptive Boosting, Stochastic Gradient Boosting and Extreme Gradient Boosting algorithms. Bagging models applied were the Random Forest and Bagged Classification and Regression Trees algorithms. Individual base learners including the Linear Discriminant Analysis, Classification and Regression Trees, Support Vector Machine (Radial Basis Function Kernel), K-Nearest Neighbors and Naive Bayes algorithms were evaluated for correlation and stacked together as contributors to the Logistic Regression and Random Forest meta-models. The resulting predictions derived from all ensemble learning models were evaluated based on their discrimination power using the area under the receiver operating characteristics curve (AUROC) metric.

<img src="images/Project38_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Unsupervised Learning : Discovering Latent Variables in High-Dimensional Data using Exploratory Factor Analysis](https://johnpaulinepineda.github.io/Portfolio_Project_36/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_36/) explores different variations of the exploratory factor analysis method for discovering latent patterns in adequately correlated high-dimensional data. Methods applied in the analysis to estimate and identify potential underlying structures from observed variables included Principal Axes Factor Extraction and Maximum Likelihood Factor Extraction. The approaches used to simplify the derived factor structures to achieve a more interpretable pattern of factor loadings included Varimax Rotation and Promax Rotation. Combinations of the factor extraction and rotation methods were separately applied on the original dataset across different numbers of factors, with the model fit evaluated using the standardized root mean square of the residual, Tucker-Lewis fit index, Bayesian information criterion and high residual rate. The extracted and rotated factors were visualized using the factor loading and dandelion plots.

<img src="images/Project36_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Exploring Penalized Models for Handling High-Dimensional Survival Data](https://johnpaulinepineda.github.io/Portfolio_Project_34/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_34/) explores different regularization methods for minimizing model complexity by promoting coefficient sparsity in high-dimensional survival data. Using a Cox Proportional Hazards Regression model structure, penalty functions applied during the coefficient estimation process included the Least Absolute Shrinkage and Selection Operator, Elastic Net, Minimax Concave Penalty, Smoothly Clipped Absolute Deviation and Fused Least Absolute Shrinkage and Selection Operator. The predictive performance for each algorithm was evaluated using the time-dependent area under the receiver operating characteristics curve (AUROC) metric through both internal bootstrap and external validation methods. Model calibration was similarly assessed by plotting the predicted probabilities from the model versus the actual survival probabilities. The differences in survival time for different risk groups determined from the calibration analyses were additionally examined using the  Kaplan-Meier survival curves.

<img src="images/Project34_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Unsupervised Learning : Exploring and Visualizing Extracted Dimensions from Principal Component Algorithms](https://johnpaulinepineda.github.io/Portfolio_Project_33/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_33/) explores the various principal component-based dimensionality reduction algorithms for extracting and visualizing information. Methods applied in the analysis to transform and reduce high dimensional data included the Principal Component Analysis, Correspondence Analysis, Multiple Correspondence Analysis, Multiple Factor Analysis and Factor Analysis of Mixed Data. The algorithms were separately applied on different iterations of the original dataset as appropriate to the given method, with the correlation plots, factorial maps and biplots (as applicable) formulated for a more intuitive visualization of the extracted dimensions.

<img src="images/Project33_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Statistical Evaluation : Sample Size and Power Calculations for Tests Comparing Proportions in Clinical Research](https://johnpaulinepineda.github.io/Portfolio_Project_32/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_32/) explores the various sample size and power calculations for proportion comparison tests in clinical research. The important factors to be assessed prior to determining the appropriate sample sizes were evaluated for the One-Sample, Unpaired Two-Sample, Paired Two-Sample and Multiple-Sample One-Way ANOVA Pairwise Designs. Power analyses were conducted to address the requirements across different study hypotheses including Tests of Equality, Non-Inferiority, Superiority, Equivalence and Categorical Shift.

<img src="images/Project32_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Statistical Evaluation : Sample Size and Power Calculations for Tests Comparing Means in Clinical Research](https://johnpaulinepineda.github.io/Portfolio_Project_31/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_31/) explores the various sample size and power calculations for mean comparison tests in clinical research. The important factors to be assessed prior to determining the appropriate sample sizes were evaluated for the One-Sample, Two-Sample and Multiple-Sample One-Way ANOVA Pairwise Designs. Power analyses were conducted to address the requirements across different study hypotheses including Tests of Equality, Non-Inferiority, Superiority and Equivalence.

<img src="images/Project31_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Data Preprocessing : Comparing Oversampling and Undersampling Algorithms for Class Imbalance Treatment](https://johnpaulinepineda.github.io/Portfolio_Project_30/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_30/) explores the various oversampling and undersampling methods to address imbalanced classification problems. The algorithms applied to augment imbalanced data prior to model training by updating the original data set to minimize the effect of the disproportionate ratio of instances in each class included Near Miss, Tomek Links, Adaptive Synthetic Algorithm, Borderline Synthetic Minority Oversampling Technique, Synthetic Minority Oversampling Technique and Random Oversampling Examples. The derived class distributions were compared to the original data set and those applied with both random undersampling and oversampling methods. Using the Logistic Regression model structure, the corresponding logistic curves estimated from both the original and updated data were subjectively assessed in terms of skewness, data sparsity and class overlap. Considering the differences in their intended applications dependent on the quality and characteristics of the data being evaluated, a comparison of each method's strengths and limitations was briefly discussed.

<img src="images/Project30_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Exploring Performance Evaluation Metrics for Survival Prediction](https://johnpaulinepineda.github.io/Portfolio_Project_29/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_29/) explores the various performance metrics which are adaptable to censoring conditions for evaluating survival model predictions. Using the Survival Random Forest and Cox Proportional Hazards Regression model structures, metrics applied in the analysis to estimate the generalization performance of survival models on out-of-sample data included the Concordance Index, Brier Score, Integrated Absolute Error and Integrated Square Error. The resulting split-sample cross-validated estimations derived from the metrics were evaluated in terms of their performance consistency across the candidate models. Considering the differences in their intended applications and current data restrictions, a comparison of each metric's strengths and limitations was briefly discussed.

<img src="images/Project29_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Exploring Robust Logistic Regression Models for Handling Quasi-Complete Separation](https://johnpaulinepineda.github.io/Portfolio_Project_28/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_28/) explores the various robust alternatives for handling quasi-complete separation during logistic regression modelling. Methods applied in the analysis to evaluate a quasi-complete condition when a covariate almost perfectly predicts the outcome included the Firth's Bias-Reduced Logistic Regression, Firth's Logistic Regression With Added Covariate, Firth's Logistic Regression With Intercept Correction, Bayesian Generalized Linear Model With Cauchy Priors and Ridge-Penalized Logistic Regression algorithms. The resulting predictions derived from the candidate models were evaluated in terms of the stability of their coefficient estimates and standard errors, including the validity of their logistic profiles and the distribution of their predicted points, which were all compared to that of the baseline model without any form of quasi-complete separation treatment.

<img src="images/Project28_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Unsupervised Learning : Estimating Outlier Scores Using Density and Distance-Based Anomaly Detection Algorithms](https://johnpaulinepineda.github.io/Portfolio_Project_27/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_27/) explores the various density and distance-based anomaly detection algorithms for estimating outlier scores. Methods applied in the analysis to identify abnormal points with patterns significantly deviating away from the remaining data included the Connectivity-Based Outlier Factor, Distance-Based Outlier Detection, Influenced Outlierness, Kernel-Density Estimation Outlier Score, Aggregated K-Nearest Neighbors Distance, In-Degree for Observations in a K-Nearest Neighbors Graph, Sum of Distance to K-Nearest Neighbors, Local Density Factor, Local Distance-Based Outlier Factor, Local Correlation Integral, Local Outlier Factor and Natural Outlier Factor algorithms. Using an independent label indicating the valid and outlying points from the data, the different anomaly-detection algorithms were evaluated based on their capability to effectively discriminate both data categories using the area under the receiver operating characteristics curve (AUROC) metric.

<img src="images/Project27_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Unsupervised Learning : Estimating Outlier Scores Using Isolation Forest-Based Anomaly Detection Algorithms](https://johnpaulinepineda.github.io/Portfolio_Project_26/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_26/) explores the various isolation forest-based anomaly detection algorithms for estimating outlier scores. Methods applied in the analysis to identify abnormal points with patterns significantly deviating away from the remaining data included the Isolation Forest, Extended Isolation Forest, Isolation Forest with Split Selection Criterion, Fair-Cut Forest, Density Isolation Forest and Boxed Isolation Forest algorithms. Using an independent label indicating the valid and outlying points from the data, the different anomaly-detection algorithms were evaluated based on their capability to effectively discriminate both data categories using the area under the receiver operating characteristics curve (AUROC) metric.

<img src="images/Project26_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Unsupervised Learning : Identifying Multivariate Outliers Using Density-Based Clustering Algorithms](https://johnpaulinepineda.github.io/Portfolio_Project_25/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_25/) explores the various density-based clustering algorithms for identifying multivariate outliers. Methods applied in the analysis to cluster points and detect outliers from high dimensional data included the Density-Based Spatial Clustering of Applications with Noise, Hierarchical Density-Based Spatial Clustering of Applications with Noise, Ordering Points to Identify the Clustering Structure, Jarvis-Patrick Clustering and Shared Nearest Neighbor Clustering algorithms. The different clustering algorithms were subjectively evaluated based on their capability to effectively capture the latent characteristics between the different resulting clusters. In addition, the values for the outlier detection rate and Rand index obtained for each algorithm were also assessed for an objective comparison of their clustering performance.

<img src="images/Project25_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Exploring Dichotomization Thresholding Strategies for Optimal Classification](https://johnpaulinepineda.github.io/Portfolio_Project_24/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_24/) explores the various dichotomization thresholding strategies for optimally classifying categorical responses. Using a Logistic Regression model structure, threshold criteria applied in the analysis to support optimal class prediction included Minimum Sensitivity, Minimum Specificity, Maximum Product of Specificity and Sensitivity, ROC Curve Point Closest to Point (0,1), Sensitivity Equals Specificity, Youden's Index, Maximum Efficiency, Minimization of Most Frequent Error, Maximum Diagnostic Odds Ratio, Maximum Kappa, Minimum Negative Predictive Value, Minimum Positive Predictive Value, Negative Predictive Value Equals Positive Predictive Value, Minimum P-Value and ROC Curve Point Closest to Observed Prevalence. The optimal thresholds determined for all criteria were compared and evaluated in terms of their relevance to the sensitivity and specificity objectives of the classification problem at hand.

<img src="images/Project24_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Implementing Gradient Descent Algorithm in Estimating Regression Coefficients](https://johnpaulinepineda.github.io/Portfolio_Project_23/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_23/) manually implements the Gradient Descent algorithm and evaluates a range of values for the learning rate and epoch count parameters to optimally estimate the coefficients of a linear regression model. The cost function optimization profiles of the different candidate parameter settings were compared, with the resulting estimated coefficients assessed against those obtained using normal equations which served as the reference baseline values.

<img src="images/Project23_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Unsupervised Learning : Formulating Segmented Groups Using Clustering Algorithms](https://johnpaulinepineda.github.io/Portfolio_Project_22/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_22/) explores the various clustering algorithms for segmenting information. Methods applied in the analysis to cluster high dimensional data included the K-Means, Partitioning Around Medoids, Fuzzy Analysis Clustering, Hierarchical Clustering, Agglomerative Nesting and Divisive Analysis Clustering algorithms. The different clustering algorithms were subjectively evaluated based on their capability to effectively capture the latent characteristics between the different resulting clusters. In addition, the values for the average silhouette width obtained for each algorithm were also assessed for an objective comparison of their clustering performance.

<img src="images/Project22_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Unsupervised Learning : Extracting Information Using Dimensionality Reduction Algorithms](https://johnpaulinepineda.github.io/Portfolio_Project_21/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_21/) explores the various dimensionality reduction algorithms for extracting information. Methods applied in the analysis to transform and reduce high dimensional data included the Principal Component Analysis, Singular Value Decomposition, Independent Component Analysis, Non-Negative Matrix Factorization, t-Distributed Stochastic Neighbor Embedding and Uniform Manifold Approximation and Projection algorithms. The different dimensionality reduction algorithms were subjectively evaluated based on their capability to effectively capture the latent characteristics between the different resulting components.

<img src="images/Project21_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Data Preprocessing : Remedial Procedures for Skewed Data with Extreme Outliers](https://johnpaulinepineda.github.io/Portfolio_Project_20/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_20/) explores the various remedial procedures for handling skewed data with extreme outliers for classification. Using a Logistic Regression model structure, methods applied in the analysis to address data distribution skewness and outlying points included the Box-Cox Transformation, Yeo-Johnson Transformation, Exponential Transformation, Inverse Hyperbolic Sine Transformation, Base-10 Logarithm Transformation, Natural Logarithm Transformation, Square Root Transformation, Outlier Winsorization Treatment and Outlier Spatial Sign Treatment. The resulting predictions derived from the candidate models applying various remedial procedures were evaluated in terms of their discrimination power using the area under the receiver operating characteristics curve (AUROC) metric. The AUROC values were compared to that of the baseline model which made use of data without any form of data transformation and treatment.

<img src="images/Project20_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Feature Selection : Selecting Informative Predictors Using Simulated Annealing and Genetic Algorithms](https://johnpaulinepineda.github.io/Portfolio_Project_19/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_19/) implements Simulated Annealing and Genetic Algorithms in selecting informative predictors for a modelling problem using the Random Forest and Linear Discriminant Analysis structures. The resulting predictions derived from the candidate models applying both Simulated Annealing and Genetic Algorithms were evaluated in terms of their discrimination power using the area under the receiver operating characteristics curve (AUROC) metric. The AUROC values were compared to those of the baseline models which made use of the full data without any form of feature selection, or implemented a model-specific feature selection process.

<img src="images/Project19_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Feature Selection : Selecting Informative Predictors Using Univariate Filters](https://johnpaulinepineda.github.io/Portfolio_Project_18/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_18/) implements Univariate Filters in selecting informative predictors for a modelling problem. Using the Linear Discriminant Analysis, Random Forest and Naive Bayes model structures, feature selection methods applied in the analysis included the P-Value Threshold with Bonferroni Correction and Correlation Cutoff. The resulting predictions derived from the candidate models applying various Univariate Filters were evaluated in terms of their discrimination power using the area under the receiver operating characteristics curve (AUROC) metric. The AUROC values were compared to those of the baseline models which made use of the full data without any form of feature selection, or implemented a model-specific feature selection process.

<img src="images/Project18_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Feature Selection : Selecting Informative Predictors Using Recursive Feature Elimination](https://johnpaulinepineda.github.io/Portfolio_Project_17/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_17/) implements Recursive Feature Elimination in selecting informative predictors for a modelling problem using the Random Forest, Linear Discriminant Analysis, Naive Bayes, Logistic Regression, Support Vector Machine and K-Nearest Neighbors model structures. The resulting predictions derived from the candidate models applying Recursive Feature Elimination were evaluated in terms of their discrimination power using the area under the receiver operating characteristics curve (AUROC) metric. The AUROC values were compared to those of the baseline models which made use of the full data without any form of feature selection, or implemented a model-specific feature selection process.

<img src="images/Project17_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Feature Selection : Evaluating Model-Independent Feature Importance for Predictors with Dichotomous Categorical Responses](https://johnpaulinepineda.github.io/Portfolio_Project_16/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_16/) explores various model-independent feature importance metrics for predictors with dichotomous categorical responses. Metrics applied in the analysis to evaluate feature importance for numeric predictors included the Area Under the Receiver operating characteristics Curve (AUROC), Absolute T-Test Statistic, Maximal Information Coefficient and Relief Values, while those for factor predictors included the Volcano Plot Using Fisher's Exact Test and Volcano Plot Using Gain Ratio.

<img src="images/Project16_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Feature Selection : Evaluating Model-Independent Feature Importance for Predictors with Numeric Responses](https://johnpaulinepineda.github.io/Portfolio_Project_15/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_15/) explores various model-independent feature importance metrics for predictors with numeric responses. Metrics applied in the analysis to evaluate feature importance for numeric predictors included the Locally Weighted Scatterplot Smoothing Pseudo-R-Squared, Pearson's Correlation Coefficient, Spearman's Rank Correlation Coefficient, Maximal Information Coefficient and Relief Values, while that for factor predictors included the Volcano Plot Using T-Test.

<img src="images/Project15_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Cost-Sensitive Learning for Severe Class Imbalance](https://johnpaulinepineda.github.io/Portfolio_Project_14/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_14/) explores the various cost-sensitive procedures for handling imbalanced data for classification. Methods applied in the analysis to address imbalanced data included model structures which support cost-sensitive learning, namely Class-Weighted Support Vector Machine, Cost-Sensitive Classification and Regression Trees and Cost-Sensitive C5.0 Decision Trees. The resulting predictions derived from the candidate models were evaluated in terms of their discrimination power on the minority class using the specificity metric. The specificity values were compared to those of the baseline models without cost-sensitive learning applied.

<img src="images/Project14_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Data Preprocessing : Remedial Procedures in Handling Imbalanced Data for Classification](https://johnpaulinepineda.github.io/Portfolio_Project_13/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_13/) explores the various remedial procedures for handling imbalanced data for classification. Using a Bagged Trees model structure, methods applied in the analysis to address imbalanced data included the Random Undersampling, Random Oversampling, Synthetic Minority Oversampling Technique (SMOTE) and Random Oversampling Examples (ROSE). All procedures were implemented both within and independent to the model internal validation process. The resulting predictions derived from the candidate models applying various remedial procedures were evaluated in terms of their discrimination power using the area under the receiver operating characteristics curve (AUROC) metric. The AUROC values were compared to that of the baseline model without any form of data imbalance treatment.

<img src="images/Project13_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Evaluating Hyperparameter Tuning Strategies and Resampling Distributions](https://johnpaulinepineda.github.io/Portfolio_Project_12/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_12/) implements various evaluation procedures for hyperparameter tuning strategies and resampling distributions. Using Support Vector Machine and Regularized Discriminant Analysis model structures, methods applied in the analysis to implement hyperparameter tuning included the Manual Grid Search, Automated Grid Search and Automated Random Search with the hyperparameter selection process illustrated for each. The resulting predictions derived from the candidate models applying various hyperparameter tuning procedures were evaluated in terms of their discrimination power using the area under the receiver operating characteristics curve (AUROC) metric.

<img src="images/Project12_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Modelling Multiclass Categorical Responses for Prediction](https://johnpaulinepineda.github.io/Portfolio_Project_11/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_11/) implements various predictive modelling procedures for multiclass categorical responses. Models applied in the analysis to predict multiclass categorical responses included the Penalized Multinomial Regression, Linear Discriminant Analysis, Flexible Discriminant Analysis, Mixture Discriminant Analysis, Naive Bayes, Nearest Shrunken Centroids, Averaged Neural Network, Support Vector Machine (Radial Basis Function Kernel, Polynomial Kernel), K-Nearest Neighbors, Classification and Regression Trees (CART), Conditional Inference Trees, C5.0 Decision Trees, Random Forest and Bagged Trees algorithms. The resulting predictions derived from the candidate models were evaluated in terms of their classification performance using the accuracy metric.

<img src="images/Project11_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Modelling Dichotomous Categorical Responses for Prediction](https://johnpaulinepineda.github.io/Portfolio_Project_10/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_10/) implements various predictive modelling procedures for dichotomous categorical responses. Models applied in the analysis to predict dichotomous categorical responses included the Logistic Regression, Linear Discriminant Analysis, Flexible Discriminant Analysis, Mixture Discriminant Analysis, Naive Bayes, Nearest Shrunken Centroids, Averaged Neural Network, Support Vector Machine (Radial Basis Function Kernel, Polynomial Kernel), K-Nearest Neighbors, Classification and Regression Trees (CART), Conditional Inference Trees, C5.0 Decision Trees, Random Forest and Bagged Trees algorithms. The resulting predictions derived from the candidate models were evaluated in terms of their discrimination power using the area under the receiver operating characteristics curve (AUROC) metric.

<img src="images/Project10_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Modelling Numeric Responses for Prediction](https://johnpaulinepineda.github.io/Portfolio_Project_9/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_9/) implements various predictive modelling procedures for numeric responses. Models applied in the analysis to predict numeric responses included the Linear Regression, Penalized Regression (Ridge, Least Absolute Shrinkage and Selection Operator (LASSO), ElasticNet), Principal Component Regression, Partial Least Squares, Averaged Neural Network, Multivariate Adaptive Regression Splines (MARS), Support Vector Machine (Radial Basis Function Kernel, Polynomial Kernel), K-Nearest Neighbors, Classification and Regression Trees (CART), Conditional Inference Trees, Random Forest and Cubist algorithms. The resulting predictions derived from the candidate models were evaluated in terms of their model fit using the r-squared and root mean squred error (RMSE) metrics.

<img src="images/Project9_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Resampling Procedures for Model Hyperparameter Tuning and Internal Validation](https://johnpaulinepineda.github.io/Portfolio_Project_8/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_8/) explores various resampling procedures during model hyperparameter tuning and internal validation. Using a Recursive Partitioning and Regression Trees model structure, resampling methods applied in the analysis for tuning model hyperparameters and internally validating model performance included K-Fold Cross Validation, Repeated K-Fold Cross Validation, Leave-One-Out Cross Validation, Leave-Group-Out Cross Validation, Bootstrap Validation, Bootstrap 0.632 Validation and Bootstrap with Optimism-Estimation Validation. The resulting predictions derived from the candidate models with their respective optimal hyperparameters were evaluated in terms of their classification performance using the accuracy metric, which were subsequently compared to the baseline model's apparent performance values.

<img src="images/Project8_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Clinical Research Prediction Model Development and Evaluation for Prognosis](https://johnpaulinepineda.github.io/Portfolio_Project_7/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_7/) explores the best practices when developing and evaluating prognostic models for clinical research. The general requirements for the clinical study were defined including the formulation of the research question, intended application, outcome, predictors, study design, statistical model and sample size computation. The individual steps involved in model development were presented including the data quality assessment, predictor coding, data preprocessing, as well as the specification, selection, performance estimation, performance validation and presentation of the model used in the study. Additional details on model validity evaluation was also provided. 

<img src="images/Project7_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Data Preprocessing : Missing Data Pattern Analysis, Imputation Method Evaluation and Post-Imputation Diagnostics](https://johnpaulinepineda.github.io/Portfolio_Project_6/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_6/) explores various analysis and imputation procedures for incomplete data. Missing data patterns were visualized using matrix, cluster and correlation plots, with the missing data mechanism evaluated using a Regression-Based Test. Methods applied in the analysis to replace missing data points with substituted values included Random Replacement, Median Imputation, Mean Imputation, Mutivariate Data Analysis Imputation (Regularized, Expectation-Maximization), Principal Component Analysis Imputation (Probabilistic, Bayesian, Support Vector Machine-Based, Non-Linear Iterative Partial Least Squares, Non-Linear Principal Component Analysis), Multivariate Imputation by Chained Equations, Bayesian Multiple Imputation, Expectation-Maximization with Bootstrapping, Random Forest Imputation, Multiple Imputation Using Additive Regression, Bootstrapping and Predictive Mean Matching and K-Nearest Neighbors Imputation. Performance of the missing data imputation methods was evaluated using the Processing Time, Root Mean Squared Error, Mean Absolute Error and Kolmogorov-Smirnov Test Statistic metrics. Post-imputation diagnostics was performed to assess the plausibility of the substituted values in comparison to the complete data.

<img src="images/Project6_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Survival Analysis and Descriptive Modelling for a Three-Group Right-Censored Data with Time-Independent Variables Using Cox Proportional Hazards Model](https://johnpaulinepineda.github.io/Portfolio_Project_5/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_5/) implements the survival analysis and descriptive modelling steps for a three-group right-censored data with time-independent variables using the Cox Proportional Hazards Model. The Kaplan-Meier Survival Curves and Log-Rank Test were applied during the differential analysis of the survival data between groups. All predictors' prognostic significance were individually and simultaneously evaluated using Univariate and Multivariate Cox Proportional Hazards Models, respectively. The discrimination power of the resulting models were assessed using the Harrel's Concordance Index. The final prognostic model was internally validated using Bootstrap Validation with Optimism Estimation and evaluated for compliance on all required model assumptions using the appropriate diagnostics.

<img src="images/Project5_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Supervised Learning : Survival Analysis and Descriptive Modelling for a Two-Group Right-Censored Data with Time-Independent Variables Using Cox Proportional Hazards Model](https://johnpaulinepineda.github.io/Portfolio_Project_4/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_4/) implements the survival analysis and descriptive modelling steps for a two-group right-censored data with time-independent variables using the Cox Proportional Hazards Model. The Kaplan-Meier Survival Curves and Log-Rank Test were applied during the differential analysis of the survival data between groups. All predictors' prognostic significance were individually and simultaneously evaluated using Univariate and Multivariate Cox Proportional Hazards Models, respectively. The discrimination power of the resulting models were assessed using the Harrel's Concordance Index. The final prognostic model was internally validated using Bootstrap Validation with Optimism Estimation and evaluated for compliance on all required model assumptions using the appropriate diagnostics.

<img src="images/Project4_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Statistical Evaluation : Treatment Comparison Tests Between a Single Two-Level Factor Variable and a Single Numeric Response Variable](https://johnpaulinepineda.github.io/Portfolio_Project_3/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_3/) explores the various methods in comparatively evaluating the numeric response data between two treatment groups in a clinical trial. Statistical tests applied in the analysis included the Student’s T-Test, Welch T-Test, Wilcoxon Rank-Sum Test and Robust Rank-Order Test.

<img src="images/Project3_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Data Preprocessing : Data Quality Assessment, Preprocessing and Exploration for a Regression Modelling Problem](https://johnpaulinepineda.github.io/Portfolio_Project_2/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_2/) explores the various methods in assessing data quality, implementing data preprocessing and conducting exploratory analysis for prediction problems with numeric responses. A non-exhaustive list of methods to detect missing data, extreme outlying points, near-zero variance, multicollinearity, linear dependencies and skewed distributions were evaluated. Remedial procedures on addressing data quality issues including missing data imputation, centering and scaling transformation, shape transformation and outlier treatment were similarly considered, as applicable.

<img src="images/Project2_Summary.png?raw=true"/>

---

[![](https://img.shields.io/badge/R-black?logo=R)](#) [![](https://img.shields.io/badge/RStudio-black?logo=RStudio)](#)

[Data Preprocessing : Data Quality Assessment, Preprocessing and Exploration for a Classification Modelling Problem](https://johnpaulinepineda.github.io/Portfolio_Project_1/)
<br><br>
This [project](https://johnpaulinepineda.github.io/Portfolio_Project_1/) explores the various methods in assessing data quality, implementing data preprocessing and conducting exploratory analysis for prediction problems with categorical responses. A non-exhaustive list of methods to detect missing data, extreme outlying points, near-zero variance, multicollinearity, linear dependencies and skewed distributions were evaluated. Remedial procedures on addressing data quality issues including missing data imputation, centering and scaling transformation, shape transformation and outlier treatment were similarly considered, as applicable.

<img src="images/Project1_Summary.png?raw=true"/>

---

### Visual Analytics Projects

[![](https://img.shields.io/badge/Tableau-black?logo=Tableau)](#)

[Data Visualization : Dashboard Development with Slice-and-Dice Exploration Features](https://public.tableau.com/app/profile/john.pauline.pineda/viz/SuperstoreBusinessAnalysisDashboard/BusinessDashboard)
<br><br>
This [project](https://public.tableau.com/app/profile/john.pauline.pineda/viz/SuperstoreBusinessAnalysisDashboard/BusinessDashboard) enables the exploratory and comparative analyses of business indices across product categories and market locations using the Superstore Dataset. Visualization techniques applied in the formulated dashboard included honeycomb map charts to investigate for geographic clustering between high- and low-performing locations; sparkline charts to  study the general trend of the business metrics over time; and bar charts to obtain perspectives on the performance across the various business components. Filtering features applied included a subset analysis based on regions and states.

<img src="images/TableauDashboard_1.png?raw=true"/>

---

[![](https://img.shields.io/badge/Tableau-black?logo=Tableau)](#)

[Data Visualization : Dashboard Development with Dynamic Filtering Features](https://public.tableau.com/app/profile/john.pauline.pineda/viz/IBMHRAttritionAnalysisDashboard/AttritionDashboard)
<br><br>
This [project](https://public.tableau.com/app/profile/john.pauline.pineda/viz/IBMHRAttritionAnalysisDashboard/AttritionDashboard) enables the exploratory and comparative analyses of attrition rates across employee and job profile categories using the Kaggle IBM HR Dataset. Visualization techniques applied in the formulated dashboard included bar and figure charts to investigate the proportions of employees who have left the company in reference to those who stayed. A dynamic filtering feature was applied which allows for the simultaneous subset analyses across all dashboard data components including gender, marital status, age group, education level, field specialization, department, job role, business travel, training, years at company, years in current role, years since last promotion and years with current manager.

<img src="images/TableauDashboard_2.png?raw=true"/>

---

[![](https://img.shields.io/badge/Tableau-black?logo=Tableau)](#)

[Data Visualization : Dashboard Development with Longitudinal Change Tracking Features](https://public.tableau.com/app/profile/john.pauline.pineda/viz/CrunchbaseCompanyFundRaisingTrendDashboard/PeriodicalChangeTrackingDashboard)
<br><br>
This [project](https://public.tableau.com/app/profile/john.pauline.pineda/viz/CrunchbaseCompanyFundRaisingTrendDashboard/PeriodicalChangeTrackingDashboard) enables the exploratory and comparative analyses of company fund-raising periodical performance across round types and market segments using the Tableau Public Crunchbase Dataset. Visualization techniques applied in the formulated dashboard included bar charts to benchmark the number of fundings of the current period as compared to the previous period. Sparkline charts were used to display the funding count trend over the entire time range. Bar charts and markers were utilized to highlight significant changes between adjacent periods. A longitudinal change tracking feature was applied using a slider filter which allows for the periodical subset analyses across all dashboard components.  

<img src="images/TableauDashboard_3.png?raw=true"/>

---

[![](https://img.shields.io/badge/Tableau-black?logo=Tableau)](#)

[Data Visualization : Dashboard Development with What-If Scenario Analysis Features](https://public.tableau.com/app/profile/john.pauline.pineda/viz/WirelessBusinessWhat-IfAnalysisDashboard/WirelessWhatIfAnalysisDashboard)
<br><br>
This [project](https://public.tableau.com/app/profile/john.pauline.pineda/viz/WirelessBusinessWhat-IfAnalysisDashboard/WirelessWhatIfAnalysisDashboard) enables the exploratory what-if scenario planning analyses of various business factors and conditions collectively influencing net earnings using the Kaggle Wireless Company Dataset. Visualization techniques applied in the formulated dashboard included a scatterplot to show the distribution of sales prices and gross profits prior to benchmarking actions. A bar chart was used to dynamically present the count and distribution of items based on the selected benchmark cut-offs. Gantt bar charts were utilized to compare the reference and adjusted net earning levels given the positive or negative business conditions identified. All analysis variables can be adjusted using slider and list filters which allow for the dynamic exploration of different scenarios across all dashboard components. 

<img src="images/TableauDashboard_4.png?raw=true"/>

---

[![](https://img.shields.io/badge/Tableau-black?logo=Tableau)](#)

[Data Visualization : Dashboard Development with Period-To-Date Performance Tracking Features](https://public.tableau.com/app/profile/john.pauline.pineda/viz/SportsStoreBusinessIndexMonitoringDashboard/RunningTotalsMonitoringDashboard)
<br><br>
This [project](https://public.tableau.com/app/profile/john.pauline.pineda/viz/SportsStoreBusinessIndexMonitoringDashboard/RunningTotalsMonitoringDashboard) enables the period-to-date performance tracking and analysis of various business indices using the Kaggle Sports Store Company Dataset. Visualization techniques applied in the formulated dashboard included line charts to demonstrate the running weekly totals of sales, profit and quantity measures for the latest quarter as compared to benchmarked periods including the previous quarter of the same year, and the same quarter from the previous year. Bar charts were used to present the consolidated quarterly indices. Additional customer and product attributes were included as drop-down list filters which allow for the periodical subset analyses across all dashboard components.

<img src="images/TableauDashboard_5.png?raw=true"/>

---

### Scientific Research Papers

[High Diagnostic Accuracy of Epigenetic Imprinting Biomarkers in Thyroid Nodules](https://ascopubs.org/doi/10.1200/JCO.22.00232)
<br><br>
This [paper](https://ascopubs.org/doi/10.1200/JCO.22.00232) published in the Journal of Clinical Oncology is a collaborative study on formulating a thyroid cancer predicton model using epigenetic imprinting biomarkers. Statistical methods were applied accordingly during differential analysis; gene screening study; and diagnostic grading model building, optimization and validation. The area under the receiver operating characteristics curve (AUROC) metric was used to measure the discrimination power of the candidate predictors, with the classification performance of the candidate models evaluated using the sensitivity, specificity, positive predictive value (PPV) and negative predictive value (NPV) metrics. The final prediction model was internally validated using a 500-cycle optimism-adjusted bootstrap and externally validated using an independent cohort.

---

[Epigenetic Imprinting Alterations as Effective Diagnostic Biomarkers for Early-Stage Lung Cancer and Small Pulmonary Nodules](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-021-01203-5)
<br><br>
This [paper](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-021-01203-5) published in the Clinical Epigenetics journal is a collaborative study on formulating a lung cancer predicton model using epigenetic imprinting biomarkers. Statistical methods were applied accordingly during differential analysis; gene screening study; and diagnostic grading model building, optimization and validation. The area under the receiver operating characteristics curve (AUROC) metric was used to measure the discrimination power of the candidate predictors, with the classification performance of the candidate models evaluated using the sensitivity and specificity metrics. The final prediction model was externally validated using an independent cohort.

---

[Novel Visualized Quantitative Epigenetic Imprinted Gene Biomarkers Diagnose the Malignancy of Ten Cancer Types](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-020-00861-1)
<br><br>
This [paper](https://clinicalepigeneticsjournal.biomedcentral.com/articles/10.1186/s13148-020-00861-1) published in the Clinical Epigenetics journal is a collaborative study on malignancy differentiation for bladder, colorectal, gastric, pancreatic, skin, breast, esophagus, lung, prostate and thyoid tumors using epigenetic imprinting biomarkers. Statistical methods were applied accordingly during differential analysis; gene screening study; and diagnostic classification model building. The area under the receiver operating characteristics curve (AUROC) metric was used to measure the discrimination power of the candidate predictors, with the classification performance of the candidate models evaluated using the sensitivity and specificity metrics. The preliminary models presented were exploratory in nature and were not externally validated using an independent cohort.

---

### Conference Abstracts

- [Advancing Malignancy Risk Stratification for Early-Stage Cancers in Lung Nodules by Combined Imaging and Electrical Impedance Analysis](https://www.jto.org/article/S1556-0864(23)01635-0/fulltext#%20)
- [Intronic Noncoding RNA Expression of DCN is Related to Cancer-Associated Fibroblasts and NSCLC Patients’ Prognosis](https://www.jto.org/article/S1556-0864(21)00892-3/fulltext)
- [Epigenetic Imprinted Genes as Biomarkers for the Proactive Detection and Accurate Presurgical Diagnosis of Small Lung Nodules](https://www.jto.org/article/S1556-0864(21)00820-0/fulltext)
- [Effect of Epigenetic Imprinting Biomarkers in Urine Exfoliated Cells (UEC) on the Diagnostic Accuracy of Low-Grade Bladder Cancer](https://ascopubs.org/doi/10.1200/JCO.2020.38.15_suppl.e17027)
- [Epigenetic Imprinted Gene Biomarkers Significantly Improve the Accuracy of Presurgical Bronchoscopy Diagnosis of Lung Cancer](https://ascopubs.org/doi/10.1200/JCO.2020.38.15_suppl.e21055)
- [Quantitative Chromogenic Imprinted Gene In Situ Hybridization (QCIGISH) Technique Could Diagnose Lung Cancer Accurately](https://www.atsjournals.org/doi/10.1164/ajrccm-conference.2020.201.1_MeetingAbstracts.A4452)

---
<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
