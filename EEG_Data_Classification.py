# ============================================================================
# EEG DATA CLASSIFICATION - MACHINE LEARNING PIPELINE
# ============================================================================
# Author: Kiran Firdous
# Date: October 2023
# Description: Classification of EEG data for stress detection using multiple ML models
# ============================================================================

# ==================== IMPORTS ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import warnings
warnings.filterwarnings("ignore")

# Data visualization
%matplotlib inline

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn import metrics
from sklearn.metrics import (mean_absolute_error, mean_squared_error, accuracy_score,
                             classification_report, confusion_matrix, precision_score,
                             recall_score, f1_score)

# Machine Learning Models
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                             AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==================== DATA LOADING ====================
# Note: Uncomment and modify the path to your actual data file
# EEG = pd.read_csv("Data_30_EC/EC_EEG_cleanData_30.csv")
# print("Data shape:", EEG.shape)
# print("Data columns:", EEG.columns.tolist())

# For this template, we'll create a sample structure
print("=" * 70)
print("EEG DATA CLASSIFICATION PIPELINE")
print("=" * 70)
print("\nNote: This is a template. Please load your actual data file.")
print("Expected data format: CSV with EEG features and 'state' column (0/1)")
print("\n0 = Non-Stressed, 1 = Stressed")

# ==================== DATA PREPROCESSING FUNCTIONS ====================
def str_features_to_numeric(data):
    """
    Transform all string features in the dataframe to numeric features
    """
    # Determine categorical features
    categorical_columns = []
    numerics = ['int8', 'int16', 'int32', 'int64', 
                'float16', 'float32', 'float64']
    features = data.columns.values.tolist()
    
    for col in features:
        if data[col].dtype in numerics:
            continue
        categorical_columns.append(col)
    
    # Encoding categorical features
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            le.fit(list(data[col].astype(str).values))
            data[col] = le.transform(list(data[col].astype(str).values))
    
    return data

def prepare_data(data_path=None):
    """
    Prepare EEG data for modeling
    """
    if data_path:
        EEG = pd.read_csv(data_path)
    else:
        # Sample data structure - replace with actual data loading
        print("\nLoading sample data structure...")
        # Create sample data for demonstration
        n_samples = 54914
        n_features = 27
        
        # Sample EEG feature names based on your data
        feature_names = [
            'EEG_mean()', 'EEG_std()', 'EEG_mad()', 'EEG_max()', 'EEG_min()',
            'EEG_energy()', 'EEG_iqr()', 'EEG_entropy()', 'EEG_skew()',
            'EEG_kurtosis()', 'EEG_psd_alpha()', 'EEG_psd_beta()',
            'EEG_psd_theta()', 'EEG_psd_delta()', 'EEG_psd_gamma()',
            'EEG_hjorth_mobility()', 'EEG_hjorth_complexity()',
            'EEG_spectral_entropy()', 'EEG_svd_entropy()',
            'EEG_approximate_entropy()', 'EEG_sample_entropy()',
            'EEG_permutation_entropy()', 'EEG_fisher_info()',
            'EEG_detrended_fluctuation()', 'EEG_hurst_exponent()',
            'EEG_zero_crossing()', 'EEG_nonlinear_energy()'
        ]
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        EEG = pd.DataFrame(X, columns=feature_names)
        EEG['state'] = y
        EEG['subject'] = np.random.randint(1, 31, n_samples)
    
    print(f"\nData shape: {EEG.shape}")
    print(f"Columns: {EEG.columns.tolist()}")
    
    # Check class distribution
    if 'state' in EEG.columns:
        print(f"\nClass distribution:")
        print(EEG['state'].value_counts())
        print(f"Stressed (1): {EEG['state'].value_counts().get(1, 0):,}")
        print(f"Non-Stressed (0): {EEG['state'].value_counts().get(0, 0):,}")
    
    # Convert string features to numeric
    EEG = str_features_to_numeric(EEG)
    
    return EEG

# ==================== MODEL TRAINING & EVALUATION ====================
class EEGClassifier:
    """
    Main class for EEG data classification
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_features_labels(self, data):
        """
        Prepare features (X) and labels (y) from data
        """
        # Separate labels
        Labels = data[["state"]]
        
        # Create feature dataframe (remove label and subject columns)
        features_df = data.drop(['state', 'subject'], axis=1, errors='ignore')
        
        X = features_df
        y = Labels
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.1, random_state=42):
        """
        Split data into train and test sets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def standardize_data(self, X_train, X_test):
        """
        Standardize features using StandardScaler
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, model, model_name, X_train=None, y_train=None):
        """
        Train a single model
        """
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        
        print(f"\nTraining {model_name}...")
        
        # Convert y_train to 1D array if needed
        if hasattr(y_train, 'values'):
            y_train_flat = y_train.values.ravel()
        else:
            y_train_flat = y_train
        
        # Train the model
        model.fit(X_train, y_train_flat)
        
        # Store the model
        self.models[model_name] = model
        
        return model
    
    def evaluate_model(self, model, model_name, X_test=None, y_test=None):
        """
        Evaluate a trained model
        """
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions) * 100
        precision = precision_score(y_test, predictions, average='weighted') * 100
        recall = recall_score(y_test, predictions, average='weighted') * 100
        f1 = f1_score(y_test, predictions, average='weighted') * 100
        
        # Classification report
        report = classification_report(y_test, predictions, target_names=['Non-Stressed', 'Stressed'])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions,
            'confusion_matrix': cm,
            'report': report
        }
        
        return accuracy, cm, report
    
    def plot_confusion_matrix(self, cm, model_name, ax=None):
        """
        Plot confusion matrix
        """
        if ax is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()
        
        sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Reds', ax=ax)
        ax.set_title(f'{model_name} - Confusion Matrix\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        ax.xaxis.set_ticklabels(['Stressed', 'Non-Stressed'])
        ax.yaxis.set_ticklabels(['Stressed', 'Non-Stressed'])
        
        return ax
    
    def run_all_models(self):
        """
        Run and evaluate all classification models
        """
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        # Define all models to train
        model_configs = [
            ('Random Forest', RandomForestClassifier(random_state=self.random_state, n_estimators=100)),
            ('Decision Tree', DecisionTreeClassifier(criterion="entropy", max_depth=14, random_state=self.random_state)),
            ('Logistic Regression', LogisticRegression(random_state=self.random_state, max_iter=1000)),
            ('SVM', SVC(random_state=self.random_state)),
            ('K-Nearest Neighbors', KNeighborsClassifier()),
            ('Gradient Boosting', GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, random_state=self.random_state)),
            ('AdaBoost', AdaBoostClassifier(random_state=self.random_state)),
            ('XGBoost', XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')),
            ('LightGBM', LGBMClassifier(random_state=self.random_state)),
            ('Extra Trees', ExtraTreesClassifier(random_state=self.random_state)),
            ('Bernoulli Naive Bayes', BernoulliNB())
        ]
        
        results_summary = []
        
        for model_name, model in model_configs:
            # Train model
            trained_model = self.train_model(model, model_name)
            
            # Evaluate model
            accuracy, cm, report = self.evaluate_model(trained_model, model_name)
            
            # Store summary
            results_summary.append({
                'Model': model_name,
                'Accuracy': f"{accuracy:.2f}%",
                'Precision': f"{self.results[model_name]['precision']:.2f}%",
                'Recall': f"{self.results[model_name]['recall']:.2f}%",
                'F1-Score': f"{self.results[model_name]['f1']:.2f}%"
            })
            
            # Print results
            print(f"\n{model_name}:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Precision: {self.results[model_name]['precision']:.2f}%")
            print(f"  Recall: {self.results[model_name]['recall']:.2f}%")
            print(f"  F1-Score: {self.results[model_name]['f1']:.2f}%")
        
        # Create results dataframe
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY (Ranked by Accuracy)")
        print("="*70)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_model_comparison(self):
        """
        Plot comparison of all models
        """
        if not self.results:
            print("No results to plot. Run models first.")
            return
        
        # Extract accuracies
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        models_sorted = [models[i] for i in sorted_indices]
        accuracies_sorted = [accuracies[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(models_sorted, accuracies_sorted, color='steelblue')
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlim([0, 100])
        
        # Add value labels
        for bar, acc in zip(bars, accuracies_sorted):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{acc:.2f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_all_confusion_matrices(self):
        """
        Plot confusion matrices for all models
        """
        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if model_name in self.results:
                cm = self.results[model_name]['confusion_matrix']
                self.plot_confusion_matrix(cm, model_name, ax=axes[idx])
            else:
                axes[idx].text(0.5, 0.5, f"No results for {model_name}",
                              ha='center', va='center', fontsize=12)
                axes[idx].set_title(model_name)
                axes[idx].axis('off')
        
        # Hide empty subplots
        for idx in range(len(self.models), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()

# ==================== NEURAL NETWORK MODEL ====================
def create_neural_network(input_dim, layers=[64, 32, 16], dropout_rate=0.3):
    """
    Create a neural network model for EEG classification
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    
    return model

def train_neural_network(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """
    Train and evaluate neural network
    """
    input_dim = X_train.shape[1]
    model = create_neural_network(input_dim)
    
    print("\nTraining Neural Network...")
    history = model.fit(X_train, y_train.values.ravel(),
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.1,
                       verbose=0)
    
    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test.values.ravel(), verbose=0)
    accuracy_percent = accuracy * 100
    
    print(f"Neural Network Accuracy: {accuracy_percent:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, accuracy_percent

# ==================== MAIN EXECUTION ====================
def main():
    """
    Main execution function
    """
    print("="*70)
    print("EEG STRESS CLASSIFICATION - COMPLETE PIPELINE")
    print("="*70)
    
    # Initialize classifier
    classifier = EEGClassifier(random_state=42)
    
    # Load and prepare data
    # Replace with your actual data path
    # data_path = "your_data.csv"
    # EEG_data = prepare_data(data_path)
    
    # For demonstration, create sample data
    print("\n1. PREPARING DATA...")
    EEG_data = prepare_data()
    
    # Prepare features and labels
    X, y = classifier.prepare_features_labels(EEG_data)
    
    # Split data
    print("\n2. SPLITTING DATA...")
    X_train, X_test, y_train, y_test = classifier.split_data(X, y, test_size=0.1)
    
    # Standardize data
    print("\n3. STANDARDIZING DATA...")
    X_train_scaled, X_test_scaled = classifier.standardize_data(X_train, X_test)
    classifier.X_train = X_train_scaled
    classifier.X_test = X_test_scaled
    
    # Train and evaluate all models
    print("\n4. TRAINING MACHINE LEARNING MODELS...")
    results_df = classifier.run_all_models()
    
    # Plot model comparison
    print("\n5. VISUALIZING RESULTS...")
    classifier.plot_model_comparison()
    
    # Plot confusion matrices
    classifier.plot_all_confusion_matrices()
    
    # Train Neural Network
    print("\n6. TRAINING NEURAL NETWORK...")
    nn_model, nn_accuracy = train_neural_network(
        X_train_scaled, y_train, X_test_scaled, y_test,
        epochs=30, batch_size=64
    )
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    # Add neural network to results
    results_summary = results_df.copy()
    neural_network_row = pd.DataFrame({
        'Model': ['Neural Network'],
        'Accuracy': [f"{nn_accuracy:.2f}%"],
        'Precision': ['-'],
        'Recall': ['-'],
        'F1-Score': ['-']
    })
    
    all_results = pd.concat([results_summary, neural_network_row], ignore_index=True)
    all_results = all_results.sort_values('Accuracy', ascending=False)
    
    print("\nAll Models Performance (Including Neural Network):")
    print(all_results.to_string(index=False))
    
    # Best model
    best_model_name = all_results.iloc[0]['Model']
    best_accuracy = all_results.iloc[0]['Accuracy']
    print(f"\n✓ Best Model: {best_model_name} with {best_accuracy} accuracy")
    
    # Save results to CSV
    all_results.to_csv('eeg_classification_results.csv', index=False)
    print("✓ Results saved to 'eeg_classification_results.csv'")
    
    return classifier, all_results, nn_model

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("EEG Classification Pipeline")
    print("-" * 50)
    print("This script provides a complete pipeline for EEG data classification.")
    print("Please follow these steps:")
    print("1. Replace the sample data with your actual EEG data")
    print("2. Modify the feature names if different from expected")
    print("3. Adjust model parameters as needed")
    print("4. Run the script to train and evaluate all models")
    
    # Uncomment the following line to run the complete pipeline
    # classifier, results, nn_model = main()
    
    print("\nTo run the pipeline, uncomment 'classifier, results, nn_model = main()'")
    print("and modify the data loading section with your actual data path.")

# ==================== SAMPLE RESULTS FROM YOUR DATA ====================
print("\n" + "="*70)
print("SAMPLE RESULTS FROM ACTUAL RUN (Based on your PDF)")
print("="*70)

sample_results = pd.DataFrame({
    'Model': [
        'Extra Trees Classifier',
        'Random Forest',
        'LightGBM',
        'AdaBoost',
        'Gradient Boosting',
        'Decision Tree',
        'K-Nearest Neighbors',
        'Logistic Regression',
        'SVM',
        'Bernoulli Naive Bayes'
    ],
    'Accuracy': [
        '85.40%',
        '84.40%',
        '83.50%',
        '77.51%',
        '75.62%',
        '79.99%',
        '63.31%',
        '58.54%',
        '59.36%',
        '59.36%'
    ],
    'Precision': [
        '85.00%',
        '84.00%',
        '83.00%',
        '77.00%',
        '75.00%',
        '80.00%',
        '63.00%',
        '54.00%',
        '35.00%',
        '35.00%'
    ],
    'Recall': [
        '85.00%',
        '84.00%',
        '84.00%',
        '78.00%',
        '76.00%',
        '80.00%',
        '63.00%',
        '59.00%',
        '59.00%',
        '59.00%'
    ],
    'F1-Score': [
        '85.00%',
        '84.00%',
        '83.00%',
        '77.00%',
        '75.00%',
        '80.00%',
        '63.00%',
        '49.00%',
        '44.00%',
        '44.00%'
    ]
})

print("\nSample Performance Metrics (From your actual run):")
print(sample_results.to_string(index=False))

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
Based on the analysis:
1. Ensemble methods (Extra Trees, Random Forest, LightGBM) performed best
2. Tree-based models generally outperform linear models for EEG data
3. Deep Learning (Neural Network) shows competitive performance
4. The best model achieves ~85% accuracy in stress detection
""")
