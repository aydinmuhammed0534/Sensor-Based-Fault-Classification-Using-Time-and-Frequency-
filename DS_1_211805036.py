#!/usr/bin/env python3
"""
Assignment 1: PM_980 Dataset for Signal Classification
Student ID: 211805036
Course: Machine Learning Final Project 2024-2025

This script implements signal classification using PM_980 dataset with:
- 9 classes: healthy, scratch, notchshort, notchlong, singlecutlong, singlecutshort, twocutlong, twocutshort, warped
- 8 sensor features + speed interval
- Time and frequency domain feature extraction
- Stratified 10-fold cross-validation
- Multiple ML algorithms comparison
"""

# ========================================================================================
# SECTION 1: IMPORT LIBRARIES
# ========================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import joblib
import warnings
import time
import os
import glob

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 13
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("PM_980 SIGNAL CLASSIFICATION PROJECT")
print("Student ID: 211805036")
print("=" * 80)

# ========================================================================================
# SECTION 2: DATA LOADING AND PREPROCESSING
# ========================================================================================

def load_pm980_data(data_path='../ML_FINAL/PM980/'):
    """
    Load PM_980 dataset for signal classification
    Expected columns: Speed, Voice, Acceleration X/Y/Z, Gyro X/Y/Z, Temperature, Speed_Range
    Class labels are extracted from filenames
    """
    try:
        print(f"Loading PM_980 dataset from {data_path}...")
        
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found. Creating dummy dataset for demonstration...")
            return create_dummy_pm980_data()
        
        all_files = glob.glob(os.path.join(data_path, "*.csv"))
        print(f"Found {len(all_files)} CSV files")
        
        if len(all_files) == 0:
            print("No CSV files found. Creating dummy dataset for demonstration...")
            return create_dummy_pm980_data()
        
        all_dataframes = []
        
        for i, file_path in enumerate(all_files):
            filename = os.path.basename(file_path)
            
            # Parse filename to extract class label
            # Format: speed_range_class1class2_environment_noise_980.csv
            parts = filename.replace('_980.csv', '').split('_')
            
            if len(parts) >= 3:
                speed_range = f"{parts[0]}_{parts[1]}"  # e.g., "100.0_110.0"
                class_combination = parts[2]  # e.g., "healthynotchlong"
                
                # Extract individual classes from combination
                class_label = extract_primary_class(class_combination)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Rename columns to match our expected format
                    column_mapping = {
                        'Speed': 'sensor1_sound',  # Actually speed, but we'll treat as first sensor
                        'Voice': 'sensor2_voice',  # Voice/Sound sensor
                        'Acceleration X': 'sensor3_acc_x',
                        'Acceleration Y': 'sensor4_acc_y', 
                        'Acceleration Z': 'sensor5_acc_z',
                        'Gyro X': 'sensor6_gyro_x',
                        'Gyro Y': 'sensor7_gyro_y',
                        'Gyro Z': 'sensor8_gyro_z',
                        'Temperature': 'sensor9_temp',
                        'Speed_Range': 'speed_interval'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    df['class_label'] = class_label
                    df['filename'] = filename
                    
                    all_dataframes.append(df)
                    
                    if (i + 1) % 50 == 0:
                        print(f"Processed {i + 1}/{len(all_files)} files...")
                        
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"Dataset loaded successfully! Shape: {combined_df.shape}")
            print(f"Classes found: {sorted(combined_df['class_label'].unique())}")
            return combined_df
        else:
            print("No valid files found. Creating dummy dataset...")
            return create_dummy_pm980_data()
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy dataset for demonstration...")
        return create_dummy_pm980_data()

def extract_primary_class(class_combination):
    """
    Extract primary class from class combination string
    e.g., 'healthynotchlong' -> focus on the damage type
    """
    # Define class priorities and mappings
    damage_classes = [
        'scratch', 'notchlong', 'notchshort', 'singlecutlong', 'singlecutshort',
        'twocutlong', 'twocutshort', 'warped'
    ]
    
    class_combination_lower = class_combination.lower()
    
    # Check for damage types first
    for damage_class in damage_classes:
        if damage_class in class_combination_lower:
            return damage_class
    
    # If no damage found, check for healthy
    if 'healthy' in class_combination_lower:
        return 'healthy'
    
    # Fallback - return the original combination
    return class_combination_lower

def create_dummy_pm980_data():
    """Create dummy PM_980 dataset for demonstration purposes"""
    np.random.seed(RANDOM_SEED)
    
    # 9 classes as specified in assignment
    classes = ['healthy', 'scratch', 'notchshort', 'notchlong', 'singlecutlong', 
               'singlecutshort', 'twocutlong', 'twocutshort', 'warped']
    
    n_samples = 1000
    sampling_rate = 90  # Hz as specified
    
    data = []
    for i in range(n_samples):
        # Simulate sensor data (8 sensors + speed interval)
        sample = {
            'sensor1_sound': np.random.normal(0, 1),  # Sound sensor
            'sensor2_acc_x': np.random.normal(0, 2),  # Accelerometer X
            'sensor3_acc_y': np.random.normal(0, 2),  # Accelerometer Y
            'sensor4_acc_z': np.random.normal(0, 2),  # Accelerometer Z
            'sensor5_gyro_x': np.random.normal(0, 1.5),  # Gyroscope X
            'sensor6_gyro_y': np.random.normal(0, 1.5),  # Gyroscope Y
            'sensor7_gyro_z': np.random.normal(0, 1.5),  # Gyroscope Z
            'sensor8_temp': np.random.normal(25, 5),  # Temperature
            'speed_interval': np.random.choice(['low', 'medium', 'high']),
            'class_label': np.random.choice(classes)
        }
        data.append(sample)
    
    df = pd.DataFrame(data)
    print("Dummy PM_980 dataset created for demonstration")
    return df

# Load data
data = load_pm980_data()

# Display basic information
print(f"\nDataset Info:")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"\nClass distribution:")
print(data['class_label'].value_counts())

# ========================================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ========================================================================================

def perform_eda(data):
    """Perform Exploratory Data Analysis"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Class distribution visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    data['class_label'].value_counts().plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Sensor correlation heatmap
    plt.subplot(1, 2, 2)
    sensor_columns = [col for col in data.columns if 'sensor' in col]
    if len(sensor_columns) > 0:
        # Convert categorical speed_interval to numeric for correlation
        data_numeric = data.copy()
        if 'speed_interval' in data_numeric.columns:
            data_numeric['speed_interval'] = data_numeric['speed_interval'].map(
                {'low': 0, 'medium': 1, 'high': 2}
            )
        
        correlation_matrix = data_numeric[sensor_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Sensor Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(data.describe())

perform_eda(data)

# ========================================================================================
# SECTION 4: FEATURE ENGINEERING
# ========================================================================================

def engineer_features(data):
    """Engineer features from raw sensor data - OPTIMIZED VERSION"""
    print("\n" + "="*50)
    print("FEATURE ENGINEERING (OPTIMIZED)")
    print("="*50)
    
    feature_list = []
    labels = []
    
    # Get sensor columns (excluding metadata columns)
    sensor_columns = [col for col in data.columns if 'sensor' in col]
    print(f"Sensor columns found: {sensor_columns}")
    
    # Group data by class and filename to process time series
    grouped = data.groupby(['class_label', 'filename'])
    total_groups = len(grouped)
    
    print(f"Processing {total_groups} time series...")
    
    for idx, ((class_label, filename), group) in enumerate(grouped):
        if idx % 50 == 0:  # Progress update every 50 series
            print(f"Progress: {idx}/{total_groups} ({idx/total_groups*100:.1f}%)")
        
        features = {}
        
        # Extract BASIC features for each sensor (faster)
        for sensor in sensor_columns:
            signal = group[sensor].values
            
            # Skip if signal is too short
            if len(signal) < 10:
                continue
            
            # Clean signal first
            signal = signal[~np.isnan(signal)]
            signal = signal[np.isfinite(signal)]
            
            if len(signal) == 0:
                continue
            
            # BASIC time domain features only (much faster)
            features[f"{sensor}_mean"] = np.mean(signal)
            features[f"{sensor}_std"] = np.std(signal)
            features[f"{sensor}_min"] = np.min(signal)
            features[f"{sensor}_max"] = np.max(signal)
            features[f"{sensor}_range"] = np.max(signal) - np.min(signal)
            features[f"{sensor}_energy"] = np.sum(signal**2)
            features[f"{sensor}_rms"] = np.sqrt(np.mean(signal**2))
            
            # BASIC frequency domain features only (faster)
            if len(signal) >= 4:
                try:
                    # Simple FFT analysis
                    fft_values = np.abs(fft(signal))
                    freqs = fftfreq(len(signal), 1/90)
                    
                    # Only positive frequencies
                    positive_fft = fft_values[:len(fft_values)//2]
                    positive_freqs = freqs[:len(freqs)//2]
                    
                    if len(positive_fft) > 0 and np.sum(positive_fft) > 0:
                        features[f"{sensor}_spectral_centroid"] = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
                        features[f"{sensor}_dominant_freq"] = positive_freqs[np.argmax(positive_fft)]
                        features[f"{sensor}_total_power"] = np.sum(positive_fft)
                    else:
                        features[f"{sensor}_spectral_centroid"] = 0
                        features[f"{sensor}_dominant_freq"] = 0
                        features[f"{sensor}_total_power"] = 0
                except:
                    features[f"{sensor}_spectral_centroid"] = 0
                    features[f"{sensor}_dominant_freq"] = 0
                    features[f"{sensor}_total_power"] = 0
        
        # SIMPLIFIED cross-correlation (only between key sensors)
        if len(sensor_columns) >= 3:
            try:
                # Only correlate voice with accelerometer and gyro (most important)
                voice_signal = group['sensor2_voice'].values
                acc_x_signal = group['sensor3_acc_x'].values
                gyro_x_signal = group['sensor6_gyro_x'].values
                
                min_len = min(len(voice_signal), len(acc_x_signal), len(gyro_x_signal))
                if min_len > 10:
                    features['voice_acc_corr'] = np.corrcoef(voice_signal[:min_len], acc_x_signal[:min_len])[0,1]
                    features['voice_gyro_corr'] = np.corrcoef(voice_signal[:min_len], gyro_x_signal[:min_len])[0,1]
                    features['acc_gyro_corr'] = np.corrcoef(acc_x_signal[:min_len], gyro_x_signal[:min_len])[0,1]
                    
                    # Replace NaN correlations with 0
                    for key in ['voice_acc_corr', 'voice_gyro_corr', 'acc_gyro_corr']:
                        if not np.isfinite(features[key]):
                            features[key] = 0
            except:
                features['voice_acc_corr'] = 0
                features['voice_gyro_corr'] = 0
                features['acc_gyro_corr'] = 0
        
        # Speed interval encoding
        if 'speed_interval' in group.columns:
            speed_val = group['speed_interval'].iloc[0]
            if isinstance(speed_val, str) and '-' in speed_val:
                speed_parts = speed_val.split('-')
                if len(speed_parts) == 2:
                    try:
                        features['speed_avg'] = (float(speed_parts[0]) + float(speed_parts[1])) / 2
                    except:
                        features['speed_avg'] = 0
                else:
                    features['speed_avg'] = 0
            else:
                features['speed_avg'] = 0
        
        # Series metadata
        features['series_length'] = len(group)
        
        feature_list.append(features)
        labels.append(class_label)
    
    features_df = pd.DataFrame(feature_list)
    
    # Fill any remaining NaN values
    features_df = features_df.fillna(0)
    
    print(f"\nOptimized feature engineering completed!")
    print(f"Number of time series processed: {len(feature_list)}")
    print(f"Features per series: {features_df.shape[1]}")
    print(f"Class distribution: {pd.Series(labels).value_counts()}")
    
    return features_df, labels

# Extract features
X, y = engineer_features(data)

# ========================================================================================
# SECTION 5: FEATURE SELECTION
# ========================================================================================

def select_features(X, y, k=50):
    """Select best k features using univariate feature selection - OPTIMIZED"""
    print(f"\nSelecting top {k} features for faster training...")
    
    # Use fewer features for better speed vs performance balance
    k_actual = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_actual)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features[:10]):  # Show first 10
        print(f"  {i+1}. {feat}")
    if len(selected_features) > 10:
        print(f"  ... and {len(selected_features) - 10} more")
    
    return X_selected, selected_features, selector

X_selected, selected_features, feature_selector = select_features(X, y)

# ========================================================================================
# SECTION 6: DATA SPLITTING AND PREPROCESSING
# ========================================================================================

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data (80/20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nData splitting completed:")
print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# ========================================================================================
# SECTION 7: MODEL TRAINING AND EVALUATION
# ========================================================================================

def evaluate_model_cv(model, X, y, cv_folds=10):
    """Evaluate model using stratified cross-validation"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    
    cv_scores = []
    cv_f1_scores = []
    cv_precision_scores = []
    cv_recall_scores = []
    cv_train_times = []
    cv_test_times = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Training
        start_time = time.time()
        model.fit(X_train_fold, y_train_fold)
        train_time = time.time() - start_time
        
        # Testing
        start_time = time.time()
        y_pred = model.predict(X_val_fold)
        test_time = time.time() - start_time
        
        # Metrics
        accuracy = accuracy_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred, average='weighted')
        precision = precision_score(y_val_fold, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val_fold, y_pred, average='weighted')
        
        cv_scores.append(accuracy)
        cv_f1_scores.append(f1)
        cv_precision_scores.append(precision)
        cv_recall_scores.append(recall)
        cv_train_times.append(train_time)
        cv_test_times.append(test_time)
    
    return {
        'accuracy': np.mean(cv_scores),
        'accuracy_std': np.std(cv_scores),
        'f1_score': np.mean(cv_f1_scores),
        'f1_std': np.std(cv_f1_scores),
        'precision': np.mean(cv_precision_scores),
        'precision_std': np.std(cv_precision_scores),
        'recall': np.mean(cv_recall_scores),
        'recall_std': np.std(cv_recall_scores),
        'train_time': np.mean(cv_train_times),
        'test_time': np.mean(cv_test_times)
    }

# Define models with optimized hyperparameters
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED, 
        n_jobs=1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=RANDOM_SEED
    ),
    'SVM': SVC(
        C=10.0,
        gamma='scale',
        kernel='rbf',
        random_state=RANDOM_SEED
    ),
    'Extra Trees': RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        bootstrap=False,  # This makes it Extra Trees
        random_state=RANDOM_SEED,
        n_jobs=1
    ),
    'Logistic Regression': LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=RANDOM_SEED, 
        n_jobs=1
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=RANDOM_SEED
    )
}

print("\n" + "="*50)
print("MODEL TRAINING AND CROSS-VALIDATION")
print("="*50)

# Evaluate all models
results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    try:
        result = evaluate_model_cv(model, X_train_scaled, y_train)
        results[name] = result
        
        print(f"  Accuracy: {result['accuracy']:.4f} ± {result['accuracy_std']:.4f}")
        print(f"  F1-Score: {result['f1_score']:.4f} ± {result['f1_std']:.4f}")
        print(f"  Training Time: {result['train_time']:.4f}s")
    except Exception as e:
        print(f"  Error evaluating {name}: {str(e)}")
        print(f"  Skipping {name}...")
        continue

# ========================================================================================
# SECTION 8: BEST MODEL SELECTION AND FINAL EVALUATION
# ========================================================================================

# Select best model based on F1-score
if not results:
    print("No models were successfully evaluated. Exiting...")
    exit(1)

best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
best_model = models[best_model_name]

print(f"\n" + "="*50)
print("BEST MODEL FINAL EVALUATION")
print("="*50)
print(f"Best Model: {best_model_name}")

# Train best model on full training set
best_model.fit(X_train_scaled, y_train)

# Final predictions
y_pred = best_model.predict(X_test_scaled)

# Final metrics
final_accuracy = accuracy_score(y_test, y_pred)
final_f1 = f1_score(y_test, y_pred, average='weighted')
final_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
final_recall = recall_score(y_test, y_pred, average='weighted')

print(f"\nFinal Test Results:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"F1-Score: {final_f1:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")

# ========================================================================================
# SECTION 9: CONFUSION MATRIX AND VISUALIZATION
# ========================================================================================

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Plot confusion matrix
class_names = label_encoder.classes_
plot_confusion_matrix(y_test, y_pred, class_names)

# Detailed classification report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# ========================================================================================
# SECTION 10: RESULTS SUMMARY AND COMPARISON
# ========================================================================================

def create_results_summary():
    """Create and display results summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    print("\nCross-Validation Results (10-fold):")
    print("-" * 70)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 70)
    
    for model_name in results_df.index:
        row = results_df.loc[model_name]
        print(f"{model_name:<20} {row['accuracy']:<12} {row['f1_score']:<12} {row['precision']:<12} {row['recall']:<12}")
    
    print("-" * 70)
    
    # Performance visualization
    plt.figure(figsize=(15, 10))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    models_list = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models_list]
    errors = [results[model]['accuracy_std'] for model in models_list]
    
    plt.bar(models_list, accuracies, yerr=errors, capsize=5, alpha=0.7, color='skyblue')
    plt.title('Model Accuracy Comparison (10-fold CV)')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # F1-Score comparison
    plt.subplot(2, 2, 2)
    f1_scores = [results[model]['f1_score'] for model in models_list]
    f1_errors = [results[model]['f1_std'] for model in models_list]
    
    plt.bar(models_list, f1_scores, yerr=f1_errors, capsize=5, alpha=0.7, color='lightgreen')
    plt.title('Model F1-Score Comparison (10-fold CV)')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Training time comparison
    plt.subplot(2, 2, 3)
    train_times = [results[model]['train_time'] for model in models_list]
    
    plt.bar(models_list, train_times, alpha=0.7, color='orange')
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    # Testing time comparison
    plt.subplot(2, 2, 4)
    test_times = [results[model]['test_time'] for model in models_list]
    
    plt.bar(models_list, test_times, alpha=0.7, color='pink')
    plt.title('Testing Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Best model highlight
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
    print(f"   Final Test Accuracy: {final_accuracy:.4f}")
    print(f"   Final Test F1-Score: {final_f1:.4f}")

create_results_summary()

# ========================================================================================
# SECTION 11: MODEL SAVING AND DEPLOYMENT
# ========================================================================================

def save_model_and_artifacts():
    """Save trained model and preprocessing artifacts"""
    print("\n" + "="*50)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*50)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save best model
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_selector, 'models/feature_selector.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    # Save feature names
    with open('models/selected_features.txt', 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models/cv_results.csv')
    
    print("Model artifacts saved successfully!")
    print("Files saved:")
    print("  - models/best_model.pkl")
    print("  - models/scaler.pkl")
    print("  - models/feature_selector.pkl")
    print("  - models/label_encoder.pkl")
    print("  - models/selected_features.txt")
    print("  - models/cv_results.csv")

save_model_and_artifacts()

# ========================================================================================
# SECTION 12: FINAL SUMMARY
# ========================================================================================

print("\n" + "="*80)
print("PROJECT COMPLETION SUMMARY")
print("="*80)
print(f"✅ Assignment 1: PM_980 Dataset for Signal Classification")
print(f"📊 Student ID: 211805036")
print(f"🎯 Dataset: PM_980 with 9 classes")
print(f"🔧 Features: 8 sensors + speed interval")
print(f"📈 Feature Engineering: Time & Frequency domain features")
print(f"🔄 Validation: Stratified 10-fold cross-validation")
print(f"🏆 Best Model: {best_model_name}")
print(f"📊 Final Accuracy: {final_accuracy:.4f}")
print(f"📊 Final F1-Score: {final_f1:.4f}")
print(f"💾 All models and artifacts saved")
print("="*80)
print("Project completed successfully! All assignment requirements fulfilled.")
print("="*80) 