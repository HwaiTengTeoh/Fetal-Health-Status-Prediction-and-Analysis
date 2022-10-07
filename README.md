# Fetal Health Status Prediction and Analysis
*Note: This repository shares the group project in fulfillment of the Data Mining class for Master of Data Science coursework.*

## Work Summary
In this era with advanced healthcare and healthcare facilities, most pregnancies would be able to go on without incident. However, there is an approximate of 8% pregnancies involving complications that could possibly cause harms to the mother or baby (4 Common Pregnancy Complications, n.d.). Hence, regular medical checkups and prenatal tests are very crucial throughout the nine-month pregnancy to ensure the fetus grows and develops healthily, at the same time detect any problems as early as possible as a means for early preparation or treatment towards the problems that may arises to the mother or baby (Fetal Health and Development, n.d.). One of the ways to perform medical checkup and prenatal test on pregnant women is cardiotocography (CTG) which measures the fetus’s heart rate and monitors the uterus contractions. With the help of medical experts to analyze the different readings such as baseline value, accelerations, fetal movement, etc. generated from the CTG on the fetus’s heart rate and uterus contractions, the fetus’s health status on whether they are normal, pathologic, or suspect can be determined.

## Objectives
- To handle imbalanced dataset using different resampling techniques
- To predict the fetus health status based on the CTGs reading by developing machine learning classification models using different Decision Tree algorithms
- To evaluate the performance of machine learning classification models built using different Decision Tree algorithms 

## Data Sources
https://archive.ics.uci.edu/ml/datasets/cardiotocography

## Methodology

## Results and Findings

## Conclusion
- CTG data helps examine and detect fetal abnormal health status. This earlier detection can help to decide on earlier medical intervention before too late for the growth of the baby. In this experimental study, we built the model with a different set of data, one with purely CTG records and another with CTG records and pattern diagnostic attributes. 
- The model's performance trained with pattern diagnostic attributes can predict the minority class well for both DT and RF. However, in the actual scenario, these attributes may not exist; thus, we focus on the model trained using merely CTG data. 
- We evaluated the performance of DT and RF (tree-based classifier) with the ID3 algorithm by taking the information gained as the criteria for splitting leave and node. We used the ensembling method of Bagging and Boosting with DT and RF. The results of this study show that the model's performance in predicting the abnormalities of baby with Bagging and Boosting are satisfactory outcomes with both DT and RF. 
- To handle the imbalanced data, we also used several resampling techniques for DT and RF. 
- Random undersampling resulted in worse than the base classifier in both DT and RF, as the instances were removed randomly, which may lead to important information loss. Undersampling techniques OSS and TL with DT combination yielded the best performance for predicting for both the minority classes, S and P. 
- Among the undersampling techniques, CNN, TL, and OSS with RF combination yielded the best performance for predicting the minority class, whereas, among oversampling techniques, random oversampling and SMOTE technique with RF were the best for classifying the minority classes. 
- However, XGBoost is the best model to classify the classes.

## Future works
- Different classifiers, such as Neural Network can be introduced to train the model for the extension of the work. 
- To handle the imbalance issue of the data, we can introduce a cost-based algorithm to apply penalties on wrongly classified outcomes rather than performing resampling techniques. 
