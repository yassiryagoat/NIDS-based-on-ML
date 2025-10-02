# --------------------- importing libraries ---------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------------- importing class for label encoding ------------

from sklearn.preprocessing import LabelEncoder

# --------- importing method to split the dataset -------------------

from sklearn.model_selection import train_test_split

# ------------- importing class for feature scaling --------------------

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# ------------ importing class for cross validation --------------

from sklearn.model_selection import StratifiedKFold

# ------------ importing XGBoost classifier ----------

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    print("XGBoost imported successfully!")
except ImportError:
    print("XGBoost not found. Please install it using: pip install xgboost")
    exit()

# ----------- importing methods for performance evaluation ------------

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

# --------------------- importing dataset -----------------------

try:
    train_dataset = pd.read_csv('NSL-KDD__Multiclass_Classification_Dataset.csv')
    test_dataset = pd.read_csv('KDDTest+.csv')
    print("Datasets loaded successfully!")
    print(f"Training dataset shape: {train_dataset.shape}")
    print(f"Test dataset shape: {test_dataset.shape}")
except FileNotFoundError as e:
    print(f"Error loading datasets: {e}")
    print("Please ensure the CSV files are in the correct directory")
    exit()

# -------------------- taking care of inconsistent data -----------------

print("\n--- Data Cleaning ---")
train_dataset = train_dataset.replace([np.inf, -np.inf], np.nan) 
train_dataset = train_dataset.dropna()
train_dataset = train_dataset.drop_duplicates()
print('Training set - No of null values : ', train_dataset.isnull().sum().sum())

test_dataset = test_dataset.replace([np.inf, -np.inf], np.nan) 
test_dataset = test_dataset.dropna()
test_dataset = test_dataset.drop_duplicates()
print('Test set - No of null values : ', test_dataset.isnull().sum().sum())

# --------------- generalising different types of attacks in test set --------------------------

print("\n--- Standardizing Attack Categories ---")
DoS = ['apache2','mailbomb','neptune','teardrop','smurf','pod','back','land','processtable']
test_dataset = test_dataset.replace(to_replace = DoS, value = 'DoS')

U2R = ['httptunnel','ps','xterm','sqlattack','rootkit','buffer_overflow','loadmodule','perl']
test_dataset = test_dataset.replace(to_replace = U2R, value = 'U2R')

R2L = ['udpstorm','worm','snmpgetattack','sendmail','named','snmpguess','xsnoop','xlock','warezclient','guess_passwd','ftp_write','multihop','imap','phf','warezmaster','spy']
test_dataset = test_dataset.replace(to_replace = R2L, value = 'R2L')

Probe = ['mscan','saint','ipsweep','portsweep','nmap','satan']
test_dataset = test_dataset.replace(to_replace = Probe, value = 'Probe')

# --------------- creating dependent variable vector ----------------

y_train = train_dataset.iloc[:, -2].values
y_test = test_dataset.iloc[:, -2].values

print("\nTraining set class distribution:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
for cls, count in zip(unique_train, counts_train):
    print(f"{cls}: {count}")

print("\nTest set class distribution:")
unique_test, counts_test = np.unique(y_test, return_counts=True)
for cls, count in zip(unique_test, counts_test):
    print(f"{cls}: {count}")

# --------------- onehotencoding the categorical variables with dummy variables ----------------

print("\n--- One-Hot Encoding ---")
train_dataset.drop(train_dataset.iloc[:, [41, 42]], inplace = True, axis = 1) 
train_dataset = pd.get_dummies(train_dataset)
test_dataset.drop(test_dataset.iloc[:, [41, 42]], inplace = True, axis = 1) 
test_dataset = pd.get_dummies(test_dataset)

print(f"After encoding - Training features: {train_dataset.shape[1]}")
print(f"After encoding - Test features: {test_dataset.shape[1]}")

# ----- inserting columns of zeroes for categorical variables that are not common in train & test set -------

# Get column names to check which ones exist
train_cols = set(train_dataset.columns)
test_cols = set(test_dataset.columns)

# Add missing columns to make datasets compatible
missing_in_train = test_cols - train_cols
missing_in_test = train_cols - test_cols

for col in missing_in_train:
    train_dataset[col] = 0

for col in missing_in_test:
    test_dataset[col] = 0

# Ensure same column order
train_dataset = train_dataset.reindex(sorted(train_dataset.columns), axis=1)
test_dataset = test_dataset.reindex(sorted(test_dataset.columns), axis=1)

print(f"Final - Training features: {train_dataset.shape[1]}")
print(f"Final - Test features: {test_dataset.shape[1]}")

# ----------------- creating matrix of features ----------------

X_train = train_dataset.iloc[:, :].values
X_test = test_dataset.iloc[:, :].values

# -------------- encoding categorical (dependent) variable -----------------

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("\nAfter label encoding:")
print("Training set class distribution:", np.asarray(np.unique(y_train, return_counts=True)))
print("Test set class distribution:", np.asarray(np.unique(y_test, return_counts=True)))

# Store class names for later reference
class_names = le.classes_
print("Class mapping:", {i: name for i, name in enumerate(class_names)})

# --------------------------- feature scaling --------------------------------

print("\n--- Feature Scaling ---")
# XGBoost can handle features on different scales, but scaling can help
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed!")

# ---------------------- Define XGBoost classifier ---------------------------

# XGBoost Configuration with optimized parameters
classifier = XGBClassifier(
    objective='multi:softprob',        # Multi-class classification with probability output
    n_estimators=300,                  # Number of boosting rounds
    max_depth=6,                       # Maximum depth of each tree
    learning_rate=0.1,                 # Step size shrinkage to prevent overfitting
    subsample=0.8,                     # Subsample ratio of the training instances
    colsample_bytree=0.8,             # Subsample ratio of columns when constructing each tree
    colsample_bylevel=0.8,            # Subsample ratio of columns for each level
    colsample_bynode=0.8,             # Subsample ratio of columns for each node
    random_state=42,                   # Random seed for reproducibility
    n_jobs=-1,                        # Use all available cores
    reg_alpha=0.1,                    # L1 regularization term on weights
    reg_lambda=1.0,                   # L2 regularization term on weights
    gamma=0.1,                        # Minimum loss reduction required to make a split
    min_child_weight=1,               # Minimum sum of instance weight needed in a child
    scale_pos_weight=1,               # Controls the balance of positive and negative weights
    tree_method='auto',               # Tree construction algorithm
    validate_parameters=True,          # Validate input parameters
    early_stopping_rounds=50,         # Stop if no improvement for 50 rounds
    eval_metric='mlogloss',           # Evaluation metric for multiclass
    verbosity=1                       # Verbosity of printing messages
)

print(f"\nUsing XGBoost classifier with configuration:")
print(f"Objective: {classifier.objective}")
print(f"N_estimators: {classifier.n_estimators}")
print(f"Max_depth: {classifier.max_depth}")
print(f"Learning_rate: {classifier.learning_rate}")
print(f"Subsample: {classifier.subsample}")
print(f"Colsample_bytree: {classifier.colsample_bytree}")

# ---------------------- stratified k fold ---------------------------

print("\n--- Cross Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
true_negative_rates = []
false_positive_rates = []
false_negative_rates = []
false_alarm_rates = []

fold = 1
for train, test in cv.split(X_train, y_train):
    print(f"Processing fold {fold}/5...")
    
    # ------------------------ splitting the train set into NEW train set and validation set ------------------------
    X_train_fold, X_validation = X_train[train], X_train[test]
    y_train_fold, y_validation = y_train[train], y_train[test]
    
    # ------------------ fitting the XGBoost classifier to the NEW train set ------------------------
    print(f"  Training XGBoost on fold {fold}...")
    
    # Create evaluation set for early stopping
    eval_set = [(X_validation, y_validation)]
    
    classifier.fit(
        X_train_fold, 
        y_train_fold,
        eval_set=eval_set,
        verbose=False  # Set to True to see training progress
    )
    
    print(f"  Training completed. Best iteration: {classifier.best_iteration}")
    
    # --------------------- predicting the validation set result ------------------------
    y_pred = classifier.predict(X_validation)
    
    # ------------------------ performance evaluation using methods from scikit learn --------------------------------
    accuracy_scores.append(accuracy_score(y_validation, y_pred))
    precision_scores.append(np.mean(precision_score(y_validation, y_pred, average=None, zero_division=0)))
    recall_scores.append(np.mean(recall_score(y_validation, y_pred, average=None, zero_division=0)))
    f1_scores.append(np.mean(f1_score(y_validation, y_pred, average=None, zero_division=0)))
    
    # ---------------------- making confusion matrix -----------------------------
    cm = multilabel_confusion_matrix(y_validation, y_pred)
    
    # --------------------- performance evaluation using the confusion matrix ---------------
    TNR = []
    FPR = []
    FNR = []
    FAR = []
    
    for i in range(len(class_names)):
        if i < len(cm):
            tn = cm[i][0][0]
            fn = cm[i][1][0]
            tp = cm[i][1][1]
            fp = cm[i][0][1]
            
            # Avoid division by zero
            if (fp + tn) > 0:
                TNR.append(tn / (fp + tn))
                FPR.append(fp / (fp + tn))
            else:
                TNR.append(0)
                FPR.append(0)
                
            if (fn + tp) > 0:
                FNR.append(fn / (fn + tp))
            else:
                FNR.append(0)
                
            if (fp + fn + tp + tn) > 0:
                FAR.append((fp + fn) / (fp + fn + tp + tn))
            else:
                FAR.append(0)

    # ------------- getting average values (of all classes) for each fold (each validation set) ----------------    
    true_negative_rates.append(np.mean(TNR))
    false_alarm_rates.append(np.mean(FAR))
    
    fold += 1

# ------------------------ printing results of performance evaluation -------------------
    
print('\n================== XGBoost Validation Set Results (Cross-Validation) =====================')
print(f'Accuracy Score : {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}')
print(f'Precision Score : {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}')
print(f'Recall Score : {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}')
print(f'F1 Score : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}') 
print(f'Specificity or True Negative Rate : {np.mean(true_negative_rates):.4f} ± {np.std(true_negative_rates):.4f}')
print(f'False Positive Rate : {np.mean(false_positive_rates):.4f} ± {np.std(false_positive_rates):.4f}')
print(f'False Negative Rate : {np.mean(false_negative_rates):.4f} ± {np.std(false_negative_rates):.4f}')
print(f'False Alarm Rate : {np.mean(false_alarm_rates):.4f} ± {np.std(false_alarm_rates):.4f}')   

# ------------------ Train final model on full training set and predict test set --------------------

print("\n--- Final XGBoost Model Training ---")
print("Training XGBoost on full training set...")

# Calculate class weights for imbalanced data
class_weights = {}
unique_classes, class_counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
n_classes = len(unique_classes)

for cls, count in zip(unique_classes, class_counts):
    class_weights[cls] = total_samples / (n_classes * count)

print("Class weights:")
for cls in unique_classes:
    print(f"{class_names[cls]}: {class_weights[cls]:.4f}")

# Apply sample weights
sample_weights = np.array([class_weights[cls] for cls in y_train])

# Create final evaluation set
X_train_final, X_eval, y_train_final, y_eval = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

eval_set_final = [(X_eval, y_eval)]

classifier.fit(
    X_train_final, 
    y_train_final,
    sample_weight=sample_weights[:len(y_train_final)],
    eval_set=eval_set_final,
    verbose=True
)

print(f"Training completed. Best iteration: {classifier.best_iteration}")
print(f"Best score: {classifier.best_score:.6f}")

y_pred = classifier.predict(X_test)

# --------------- making confusion matrix -----------------

cm = multilabel_confusion_matrix(y_test, y_pred)

# ---------------- performance evaluation using confusion matrix -----------------

TNR = []
FPR = []
FNR = []
FAR = []

for i in range(len(class_names)):    
    if i < len(cm):
        tn = cm[i][0][0]
        fn = cm[i][1][0]
        tp = cm[i][1][1]
        fp = cm[i][0][1]
        
        # Avoid division by zero
        if (fp + tn) > 0:
            TNR.append(tn / (fp + tn))
            FPR.append(fp / (fp + tn))
        else:
            TNR.append(0)
            FPR.append(0)
            
        if (fn + tp) > 0:
            FNR.append(fn / (fn + tp))
        else:
            FNR.append(0)
            
        if (fp + fn + tp + tn) > 0:
            FAR.append((fp + fn) / (fp + fn + tp + tn))
        else:
            FAR.append(0)

# --------------- printing results of performance evaluation --------------------

print('\n================ XGBoost Test Set Results ================')
print(f'Accuracy Score : {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision Score : {np.mean(precision_score(y_test, y_pred, average=None, zero_division=0)):.4f}')
print(f'Recall Score : {np.mean(recall_score(y_test, y_pred, average=None, zero_division=0)):.4f}')
print(f'F1 Score : {np.mean(f1_score(y_test, y_pred, average=None, zero_division=0)):.4f}')
print(f'Specificity or True Negative Rate : {np.mean(TNR):.4f}')
print(f'False Positive Rate : {np.mean(FPR):.4f}')
print(f'False Negative Rate : {np.mean(FNR):.4f}')
print(f'False Alarm Rate : {np.mean(FAR):.4f}')

# --------------- Detailed classification report --------------------

print('\n================ XGBoost Detailed Classification Report ================')
print(classification_report(y_test, y_pred, target_names=class_names))

# --------------- Per-class metrics --------------------

print('\n================ XGBoost Per-Class Metrics ================')
precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

for i, class_name in enumerate(class_names):
    if i < len(precision_per_class):
        print(f"{class_name}:")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1-Score: {f1_per_class[i]:.4f}")
        if i < len(TNR):
            print(f"  Specificity: {TNR[i]:.4f}")
        print()

# --------------- XGBoost specific information --------------------

print('\n================ XGBoost Model Information ================')
print(f"Number of boosting rounds: {classifier.n_estimators}")
print(f"Best iteration: {classifier.best_iteration}")
print(f"Objective: {classifier.objective}")
print(f"Number of features: {classifier.n_features_in_}")

# Feature importance analysis
print('\n================ Feature Importance Analysis ================')
feature_importance = classifier.feature_importances_
top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
print("Top 20 most important features:")
for i, idx in enumerate(reversed(top_features_idx)):
    print(f"{i+1:2d}. Feature {idx:3d}: {feature_importance[idx]:.6f}")

# Plot feature importance
try:
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(classifier, max_num_features=20, importance_type='gain')
    plt.title('XGBoost Feature Importance (Top 20)')
    plt.tight_layout()
    plt.show()
    print("Feature importance plot displayed.")
except Exception as e:
    print(f"Could not plot feature importance: {e}")

# Plot training history if available
try:
    if hasattr(classifier, 'evals_result_') and classifier.evals_result_:
        plt.figure(figsize=(12, 6))
        
        # Plot training curve
        if 'validation_0' in classifier.evals_result_:
            eval_results = classifier.evals_result_['validation_0']
            metric_name = list(eval_results.keys())[0]
            plt.plot(eval_results[metric_name], label=f'Validation {metric_name}')
            plt.xlabel('Boosting Round')
            plt.ylabel(metric_name)
            plt.title('XGBoost Training Curve')
            plt.legend()
            plt.grid(True)
            plt.show()
            print("Training curve plotted.")
except Exception as e:
    print(f"Could not plot training curve: {e}")

print("XGBoost model training and evaluation completed successfully!")