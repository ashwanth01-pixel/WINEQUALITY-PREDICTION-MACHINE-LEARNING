import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix,roc_curve, roc_auc_score,auc
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import numpy as np  
from sklearn.multiclass import OneVsRestClassifier

random.seed(100)
wine = pd.read_csv(r"C:\Users\KC\Downloads\winequality.csv")

#data samples for 5
print("DATA SAMPLES:-\n")
print(wine.head())

#plotting features vs quality
feature_names = wine.columns[:-1]
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))
axs = axs.flatten()
color_palette = plt.cm.get_cmap('Set3', len(wine['quality'].unique()))
for i, feature in enumerate(feature_names):
    feature_quality_mean = wine.groupby('quality')[feature].mean()
    num_ratings = len(feature_quality_mean)
    bars = axs[i].bar(feature_quality_mean.index, feature_quality_mean.values, color=color_palette(np.arange(num_ratings)))
    axs[i].set_title(feature)
    axs[i].set_xlabel('Quality')
    axs[i].set_ylabel('Mean ' + feature)
    axs[i].legend(bars, feature_quality_mean.index)
plt.tight_layout()
plt.show()

#descriptive statistics
print('DESCRIPTIVE STATISTICS:-') 
descriptivestatistics = wine.describe().loc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
print(descriptivestatistics)

#correlation matrix
correlation_matrix = wine.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - Wine Quality', fontsize=16)
plt.show()

# Checking null values in all features
for column in wine.columns:
    has_null = wine[column].isnull().any()
    print(f"{column}: {has_null}")

# Split the dataset into features (X) and target variable (y)
X = wine.drop('quality', axis=1)
y = wine['quality'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
sc = StandardScaler()
X_train2 = pd.DataFrame(sc.fit_transform(X_train))
X_test2 = pd.DataFrame(sc.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

#logistic regression
print("\n LOGISTIC REGRESSION")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred)
lr_precision = precision_score(y_test, y_pred,average='macro')
lr_recall = recall_score(y_test, y_pred,average='macro')
print(f"Accuracy: {lr_accuracy}")
print(f"Precision: {lr_precision}")
print(f"Recall: {lr_recall}")

#random forest
print("\n RANDOM FOREST")
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
rf_precision = precision_score(y_test, y_pred,average='macro')
rf_recall = recall_score(y_test, y_pred,average='macro')
print(f"Accuracy: {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")

#svm
print("\n SVM")
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred)
svm_precision = precision_score(y_test, y_pred,average='macro')
svm_recall = recall_score(y_test, y_pred,average='macro')
print(f"Accuracy: {svm_accuracy}")
print(f"Precision: {svm_precision}")
print(f"Recall: {svm_recall}")

#decision tree
print("\nDECISION TREE")
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred)
dt_precision = precision_score(y_test, y_pred,average='macro')
dt_recall = recall_score(y_test, y_pred,average='macro')
print(f"Accuracy: {dt_accuracy}")
print(f"Precision: {dt_precision}")
print(f"Recall: {dt_recall}")

#knn
print("\nKNN")
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred,average='macro')
knn_recall = recall_score(y_test, y_pred,average='macro')
print(f"Accuracy: {knn_accuracy}")
print(f"Precision: {knn_precision}")
print(f"Recall: {knn_recall}")

#naive bayes
print("\nNAIVE BAYE'S")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

#quality vs each quality ratings
print("quality vs each quality ratings")
quality_counts = wine['quality'].value_counts().sort_index()
print(quality_counts)
plt.bar(quality_counts.index, quality_counts.values)
plt.xlabel('Quality Ratings')
plt.ylabel('Count')
plt.title('Quality Ratings Distribution')
plt.show()

#BINNING
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
print("class distribution according to quality ratings")
print(wine['quality'].value_counts())
sns.barplot(x=(wine['quality'].value_counts().index), y=(wine['quality'].value_counts().values))
plt.xlabel('quality')
plt.ylabel('counts')
plt.show()

#confusion matrix(post processing)
y = np.where(y > 5, 1, 0)              
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model.fit(X_train,y_train)
svm_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
nb_pred = nb_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred=rf_model.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_pred)
lr_cm = confusion_matrix(y_test, lr_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
nb_cm = confusion_matrix(y_test, nb_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
rf_cm=confusion_matrix(y_test,rf_pred)
labels = ['0', '1']
classifiers = ['SVM', 'Logistic Regression', 'K-NN', 'Naive Bayes', 'Decision Tree','randomforest']
cms = [svm_cm, lr_cm, knn_cm, nb_cm, dt_cm,rf_cm]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    sns.heatmap(cms[i], annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {classifiers[i]}')
plt.tight_layout()
plt.show()

#cross-validation accuracy value and plotting
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}
color_palette = plt.cm.get_cmap('Set3', len(models))
accuracy_scores = {}
for i, (model_name, model) in enumerate(models.items()):
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{model_name}")
    print(f"  Cross-validated Accuracy: {scores.mean()}")
    print()
    accuracy_scores[model_name] = np.mean(scores)
    plt.bar(i, accuracy_scores[model_name], color=color_palette(i))
plt.xticks(range(len(models)), models.keys(), rotation=45)
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy for Different Models')
plt.legend(models.keys())
plt.show()

#roc curve
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
y_prob = classifier.predict_proba(X_test)
label_mapping = {label: i for i, label in enumerate(np.unique(y_test))}
y_test_mapped = np.array([label_mapping[label] for label in y_test])
n_classes = len(label_mapping)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_mapped == i, y_prob[:, i])
    roc_auc[i] = roc_auc_score(y_test_mapped == i, y_prob[:, i])
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], label='ROC curve (area = {:.2f})'.format(roc_auc[i]))
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

#input
print("INPUT FEATURES")
feature_names = X.columns
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier()
model.fit(X_scaled, y)
input_values = []
for feature in feature_names:
    value = float(input(f"Enter {feature}: "))
    input_values.append(value)
input_data = [input_values]
input_scaled = scaler.transform(input_data)
predicted_quality = model.predict(input_scaled)
print("Predicted wine quality:", predicted_quality)
if(predicted_quality==1):
    print("GOOD QUALITY")
else:
    print("BAD QUALITY")



