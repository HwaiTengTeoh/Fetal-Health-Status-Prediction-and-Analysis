# Fetal Health Status Prediction and Analysis
*Note: This repository shares the group project in fulfillment of the Data Mining class for Master of Data Science coursework.*

## Work Summary
In this era with advanced healthcare and healthcare facilities, most pregnancies would be able to go on without incident. However, there is an approximate of 8% pregnancies involving complications that could possibly cause harms to the mother or baby (4 Common Pregnancy Complications, n.d.). Hence, regular medical checkups and prenatal tests are very crucial throughout the nine-month pregnancy to ensure the fetus grows and develops healthily, at the same time detect any problems as early as possible as a means for early preparation or treatment towards the problems that may arises to the mother or baby (Fetal Health and Development, n.d.). One of the ways to perform medical checkup and prenatal test on pregnant women is cardiotocography (CTG) which measures the fetus’s heart rate and monitors the uterus contractions. With the help of medical experts to analyze the different readings such as baseline value, accelerations, fetal movement, etc. generated from the CTG on the fetus’s heart rate and uterus contractions, the fetus’s health status on whether they are normal, pathologic, or suspect can be determined.

## Objectives
- To handle imbalanced dataset using different resampling techniques
- To predict the fetus health status based on the CTGs reading by developing machine learning classification models using different Decision Tree algorithms
- To evaluate the performance of machine learning classification models built using different Decision Tree algorithms 

## Data Sources
Cardiotocography (CTG) Dataset from UCI Machine Learning Repository is used to study and analyze the underlying pattern that could be utilized for classification prediction. The dataset has a total count of 40 attributes with basic information and SegFile as the unique identifier, 23 numerical attributes, 11 categorical attributes, which are classification work done by medical experts, and one target label. All the categorical attributes in the dataset are diagnostic attributes labeled by the medical experts, whereas the 23 numerical attributes are the actual CTG readings. It contains 2,126 instances with three different classes of fetal health status, which are *normal, suspect, and pathologic*. 

The distribution of each class in this study is as follows:
- Normal: 1,655 (78%)
- Suspect: 295 (14%)
- Pathologic: 176 (8%)

## Methodology
- Data preprocessing such as normalization, dropping of columns, and resampling will be done to transform, preprocess and cleanse the dataset to prepare it for Modelling. 
- The different pre-processed datasets produced will be used to train a few tree-based models: Decision Tree, Random Forest, and Ensemble, which comprise Bagging and Boosting.

### Resampling techniques
As the data distribution for each 3 class is highly imbalanced, different resampling methods are used alongside other data preprocessing techniques such as normalization and dropping of columns were applied to the dataset to study the changes in the machine learning models’ performance. 

#### Naïve resampling techniques
- Random Oversampling
- Random Undersampling

#### Synthetic data sampling
- Oversampling: SMOTE
- Oversampling: Adaptive synthetic data sampling (ADASYN)
- Oversampling: Borderline SMOTE
- Oversampling: SVM SMOTE

#### Data Cleaning Techniques
- Undersampling: Tomek-link 
- Undersampling: Condensed Nearest Neighbor
- Undersampling: One Sided Selection
- Undersampling: Edited Nearest Neighbour
- Undersampling: Repeated Edited Nearest Neighbour and All KNN
- Undersampling: Neighborhood Cleaning rule

#### Summary of Resampling Techniques Used
| Resampling Techniques with kNN algorithms |	k |
|:-|:-:|
| ***<ins>Undersampling:</ins>*** | |
| Condensed Nearest Neighbor (CNN) |	1 |
| Tomac Link (TL) |	1 |
| One Sided Selection (OSS) |	1 |
| Edited Nearest Neighbour (ENN) |	3 |
| Repeated Edited Nearest Neighbour (RENN) |	3 |
| All KNN |	3 |
| Neighborhood Cleaning rule |	3 |
| ***<ins>Oversampling:</ins>*** |	 |
| SMOTE |	5 |
| BorderlineSMOTE1 |	5 |
| BorderlineSMOTE2 | 5 |

## Results and Findings
### Feature Selection

- We built the model with a different set of training data with complete data (CTG records and diagnostic pattern attributes) and data with CTG data only (without pattern attributes). 
- Figure 1a shows the features the decision tree classifier reported along with their importance score arranged in descending order with the complete data set, including CTG records and pattern diagnostic attributes.
- In contrast, Figure 1b shows the feature's importance with the decision tree model trained with merely CTG records data. It was interesting to note that pattern diagnostic attributes such as suspect pattern and shift pattern deceleration pattern occupied the top range of feature importance in Figure 1a with other CTG-related attributes. 
- The suspect pattern attribute was the most important feature and acted as the deterministic factor for the classification prediction. With another model built with merely CTG record data, the percentage of time with abnormal short-term variability, the mean value of short-term variability (SisPorto), acceleration (SisPorto), the percentage of time with abnormal long-term variability (SisPorto), and mean value of histogram are having importance of more than 10%. The rest are shown in Figure 1b. 

![](<results/Figure1a.png>)
\
**Figure 1a: Feature Importance by decision tree trained with CTG record data and pattern diagnostic attributes**


![](<results/Figure1b.png>)
\
**Figure 1b: Feature Importance by decision tree trained with CTG record data only**

### Modelling
The focus of the study here is to employ a Decision Tree (DT) and other tree-based classifiers, such as Random Forest, using the ID3 algorithm to classify the health status of fetal: normal (N), suspicious (S), and pathologic (P). With the convenience of using the sklearn package from Python in building a classification prediction model, hyperparameter tuning was performed to find the best combination of parameters for DT and RF. The default values of the parameters were used at first, but the performance of the models was improved after hyperparameter tuning with grid search. We have tested different combinations of parameters when building tree-based models with entropy as a criterion in developing nodes and leaves. The selection of parameters was as follows: 

![](<results/Table1.PNG>)

Before training the model, we performed a train-test split in the ratio of 80:20, whereby 80% of the instances were used to train the tree-based model (DT and RF), and for the rest of the dataset, 20% was used to test the performance of the model. Since the data is imbalanced, we employ different resampling methods, as discussed earlier, to balance the proportion of the class distribution in the dataset. The resampled training dataset was used to fit the model, and the model's performance adopting different resampling techniques was evaluated using the hold-out test data. 

#### Performance Evaluation 
- For the evaluation of the model, metrics such as accuracy, precision, recall, and f1-score were calculated and tabulated in Tables 2a and 2b as shown [here](https://github.com/HwaiTengTeoh/Fetal-Health-Status-Prediction-and-Analysis/blob/main/results/Performance%20Evaluation%20Results.pdf). Both tables show the metrics of the performance of models built with the adoption of the ensembling method (Bagging and Boosting) and resampling method (Undersampling and Oversampling) for two different sets of data.
- From Table 2a, we can observe that the base model's performance was outstanding, with an accuracy of more than 98%, and the metrics such as precision, recall, and f1-score represented each class, including the minority class. It was mainly due to the high predictive power of the presence of pattern diagnostic attributes tagged by experts for each instance that enhance the accuracy of the model in classifying fetal health status even though the data is highly imbalance. Adopting Boosting and Bagging with the base classifiers does not improve the model's performance, as there were no drastic changes in the metrics. 
- Since the model built was good enough, the resampling techniques here did not further improve the model but introduced more bias to the model. Hence, if the base model is performing well in classifying minority classes, even if the data is imbalanced, the introduction of resampling techniques may not enhance the performance. It was shown in the result of the model trained with full data with the presence of pattern diagnostic attributes. 
- However, we should also consider the case where we have only the CTG records data without a further diagnosis by the obstetricians to identify the pattern. By merely observing the CTG data, we have built the same sets of models without including the pattern diagnostic attributes. Table 2b shows the performance of each model, which we will discuss further in the following section. 

![](<results/Figure2.PNG>)

![](<results/Figure3.PNG>)

![](<results/Figure4.PNG>)

![](<results/Figure5.PNG>)

- Figures 2a and 2b provide the bar charts which show the accuracy score achieved by different classifiers for DT and RF, respectively. We can observe that the accuracy of DT is enhanced except for random undersampling, CBB, All KNN, SMOTE, Borderline SMOTE 1, and Borderline SMOTE 2. As for RF, the accuracy of the model is enhanced except for random undersampling, ENN, RENN, Borderline SMOTE 1, and Borderline SMOTE 2. However, our main objective is to increase the power of predicting the odd cases of fetal health. Thus, evaluation by accuracy is not sufficient to determine which model performs the best.
- Similarly, we calculated the precision measure of the classifier with different ensembling methods and resampling techniques. The precision has been calculated for each classifier and class of outcome. All techniques used have increased the precision of class N. Since we are focusing on predicting the abnormal status of fetal health, we are looking at the precision of the suspicious (S) and pathologic (P) classes. Figure 3a shows that AdaBoost with DT has the highest precision for class P, but the bagging method does not increase its precision score. Resampling techniques do not increase the precision score for DT, and there is a noticeable drop in the precision score compared with base DT. Only undersampling techniques with TL, OSS, ENN, RENN, and NCR have increased the precision score for class S. For RF, similar results were observed where the precision score for class N for all classifiers was improved. Adaboost improved the performance of precision for class P. Random Undersampling and other oversampling methods decreased the precision performance for class S. 
- When we glimpsed at the performance of the classifier on the recall score, oversampling techniques with Borderline SMOTE1 obtained the highest score, followed by Borderline SMOTE 2, SMOTE and random oversampling for class P. OSS is the only undersampling technique that has greater recall score for class P. Though Bagging and Boosting with DT gave higher recall score than the base DT, BorderlineSMOTE1 still outperform the rest. For class S, the model with ensembling method and resampling techniques were reaching higher recall score with again Borderline SMOTE1 hitting the highest score. In RF, bagging and boosting improved the recall score of the model for both class S and P. Most undersampling techniques do not improve the recall score in RF for class P and S except Random undersampling, CNN and OSS. In this case, oversampling techniques outperform the undersampling techniques as the recall score for class S and P were greatly enhanced as compared to the base RF. 
- The classification performance was also evaluated using the F1-score, taking the weighted harmonic mean of precision and recalls score. The F1 score of DT for all three classes was improved in the bagging and AdaBoost methods. However, the F1 score after applying random undersampling was reported with poor performance as well as all the oversampling methods for class S. In terms of undersampling technique, CNN had the best F1-score, followed by OSS and TL for class S. In contrast, all oversampling technique perform poorly on the model for class S. In terms of class P, undersampling methods such as CNN, TL, OSS have improved the F1 score of P except random undersampling, ENN, RENN, All KNN, and NCR. Borderline SMOTE 1 has improved the F1 score of class P and outperforms the other three oversampling techniques. 
- Overall, the F1-score for class P of all the techniques and methods was improved except for random undersampling. CNN outperformed the other undersampling techniques with the highest F1-score for class S among the undersampling techniques. Other oversampling techniques, such as SMOTE and random oversampling, have improved the F1 score but not by a significant amount. The F1 score of class N did not change much.
- XGBoost is not compared with the base models, DT, and RF performance with ensembling and resampling techniques as it uses another tree-based algorithm. However, when considering all models, XGBoost outperforms all the models discussed above with higher precision, recall and F1 value for all three class instances. Thus, we can conclude that XGBoost is the best among the models we have discussed. 


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
