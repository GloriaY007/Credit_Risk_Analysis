# Predicting-Credit-Risk
Machine Learning to Predict Credict Risk 

## Credit Risk Resampling Analysis

### Summary
This notebook used the LoanStats_2019Q1.csv dataset and implemented multiple machine learning approaches to predict the loan-status for each data point based upon the other provided metrics. The first problem before creating any models is the inherent class imbalance present in the data. There are 68470 high-risk loans and only 347 low-risk loans. This is a significant difference in data can produce a biased model (in this case, better at identifying high-risk than low-risk loans). To solve this problem, three strategies were employed. They are listed below with an anlaysis of each model that was used.

### Results
Oversampling
Balancing the classes chooses more instances of the low-risk loans for training until it is balanced with high-risk loans. The first technique of oversampling is random oversampling in which instances are randomly selected and added to the training set. When using this technique on the loans data and fitting a logisitic regression model to it, it produced an accuracy score of 64.81%. Analyzing the model further shows poor recall with an average of 60% showing that the model is not identifying relevant data consistently. When looking at the precision of the model, the average is 99%; however, further inspection shows that this is largely due to precision of the model to determine low-risk loans (100% precision). The model produced 1% precision when dealing with high-risk loans highlighting the models extremely poor ability to predict high-risk loans.

SMOTE (Synthetic Minority Oversampling Technique) oversampling was the other methos used here. SMOTE generates new instances for the minority class based upon existing data points. These new instances are then added to the minority class to balance the data. Again, a logistic regression model was greated with the SMOTE generated data and produced an accuracy score of 66.26%. Employing SMOTE produced similar precision results to random oversampling and it's the recall of the model that shows the largest difference. This model was slightly better at recognizing relevant data points with an average recall score of 69.0%.

Undersampling
Undersampling does the opposite of oversampling by decreasing the number of instances in the majority class to create a balanced dataset. The undersampling technique used is Cluster Centroid Undersampling in which the algorithm identifies clusters within the majority class and then generates data points to represent the clusters. Fitting a logistic regression model to this data and analyzing its performance, produced an accuracy of 54.43%. The model obtained the same precision scores as the oversampling methods but dropped in recall significantly. Most notably, the model scored 40.0% on recall of low-risk loans.

Combination Sampling
In an attempt to balance the trade-offs associated with over- and under-sampling, SMOTEEN (SMOTE and Edited Nearest Neighbors) combines them. It first oversamples the minority class with SMOTE and then cleans the data with an edited undersampling technique, EEN. When used as the backend of a logisitic regression model, the model received an accuracy score of 63.69%. It achieved the same precision results as all the other models and landed in the middle in recall. With scores of 69% for high-risk loan and 58% for low-risk loan recall, the model is weak at identifying relevant data.

### Conclusion
None of these algorithms performs consistently enough to be usable as a credit-risk predictor. They all share the weakness of extremely low precision for high-risk loans and none achieved scores higher than 75% for recall.This means the models are very bad at correctly identifying high-risk loans and is not consistent in identifying relevant data.

## Credit Risk Ensemble Analysis

### Summary
Ensemble learning uses multiple models and combines them so the final prediction is based on the accumulated predictions from each model. The goal of ensemble learning is to increase performance compared to a singular model by decreasing variance and improving accuracy. This notebook uses the LoanStats_2019Q1.csv dataset as the base to compare two ensemble learning methods. In order to handle the class imbalance in the dataset, we used a random oversampling technique prior to the training of each model.

### Results
Balanced Random Forest Classifier
This first algorithm balances the data by random undersampling each bootstrap. It uses multiple decision trees and trains each on a sample of the overall training data. It then combines the predictions of these individual trees to produce the final output of the model. 

On the loans dataset, the balanced accuracy score was 78.85%. Further analysis shows that the recall scores were also relatively high with 70% recall for high-risk loans and 87% recall for low-risk loans. It also achieved 100% precision for low-risk loans but only 3% precision for high-risk ones. The combination of these scores shows that the model is very good at recognizing and labeling low-risk loans; however, it has a harder time recognizing high_risk loans and is very bad at labeling them.

Easy Ensemble AdaBoost Classifier
The second algorithm is an adaptive boosting model that sequentially trains, evaluates, and creates a new model taking into account the errors of the previous one. This algorithm also achieves a balanced dataset through random undersampling and trains multiple AdaBoost learners on different bootstrap samples. This model scored significantly higher than all others with an accuracy score of 93.17%. It also achieved recall scores of 92% for high-risk and 94% for low-risk loans. The drawback of the model is its 9% precision for high-risk loans. This shows that this model still performs poorly at labeling high_risk loans correctly.

### Conclusion
Overall, the Easy Ensemble AdaBoost Classifier outperformed the Balanced Random Forest Classifier, achieving significantly better recall and accuracy scores. The main limitation comes with the low precision for high-risk loans so there are lots of instances where the model incorrectly labeled the sample as high-risk. In conclusion, further exploration into improving the precision of the model is necessary.


