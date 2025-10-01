# Ablation Experiment: Without Wasserstein
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# Ensure the imbalanced-learn library is installed
# If not installed, run: pip install -U imbalanced-learn
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    print("Error: 'imbalanced-learn' library not found.")
    print("Please install it using 'pip install -U imbalanced-learn' and try again.")
    exit()


class MarkovFeatureExtractor: # Class name changed to reflect its modified functionality
    """
    This feature extractor now only calculates and returns first, second, and third-order Markov features.
    All functionality related to Wasserstein distance has been removed.
    """
    def __init__(self,
                 amino_acids='ACDEFGHIKLMNPQRSTVWY',
                 max_sequence_length=2000,
                 batch_size=500):
        self.amino_acids = amino_acids
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self):
        """
        The feature name generation function now only includes names for Markov features.
        """
        markov_names_1 = [f'markov_1_{i}' for i in range(len(self.amino_acids)**1)]
        markov_names_2 = [f'markov_2_{i}' for i in range(len(self.amino_acids)**2)]
        markov_names_3 = [f'markov_3_{i}' for i in range(len(self.amino_acids)**3)]
        # Removed wasserstein_names
        return markov_names_1 + markov_names_2 + markov_names_3

    def _compute_markov_features(self, sequence, order=2):
        # This internal function remains unchanged
        sequence = sequence[:self.max_sequence_length]
        matrix_size = len(self.amino_acids) ** order
        transition_matrix = np.zeros(matrix_size, dtype=np.float32)
        for i in range(max(1, len(sequence) - order)):
            try:
                current_state = sequence[i:i+order]
                current_index = self._state_to_index(current_state, order)
                transition_matrix[current_index] += 1.0
            except Exception:
                continue
        total = np.sum(transition_matrix)
        if total > 0:
            transition_matrix = np.log1p(transition_matrix + 1) / (total + matrix_size)
        return transition_matrix

    def _state_to_index(self, state, order):
        # This internal function remains unchanged
        try:
            return sum(self.amino_acids.index(aa) * (len(self.amino_acids) ** i) for i, aa in enumerate(reversed(state[:order])))
        except ValueError:
            return 0

    # --- Key Modification ---
    # The _advanced_wasserstein_features function has been completely removed

    def extract_features(self, sequences):
        """
        Feature extraction logic updated to no longer compute and concatenate Wasserstein features.
        """
        all_features = []
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i+self.batch_size]
            markov_features_1 = np.array([self._compute_markov_features(seq, order=1) for seq in batch_sequences])
            markov_features_2 = np.array([self._compute_markov_features(seq, order=2) for seq in batch_sequences])
            markov_features_3 = np.array([self._compute_markov_features(seq, order=3) for seq in batch_sequences])
            
            # --- Key Modification ---
            # The original wasserstein_features calculation is removed
            # Now, we directly stack the three Markov feature matrices horizontally
            batch_combined_features = np.hstack([markov_features_1, markov_features_2, markov_features_3])
            
            all_features.append(batch_combined_features)
        return np.vstack(all_features)


def preprocess_sequences(sequences, min_length=10, max_length=2000):
    # This function remains unchanged
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    processed_data = []
    original_indices = []
    for idx, seq in enumerate(sequences):
        if isinstance(seq, str) and min_length <= len(seq) <= max_length and all(aa in valid_amino_acids for aa in seq):
            processed_data.append(seq[:max_length])
            original_indices.append(idx)
    return processed_data, original_indices


def create_pipeline(): # Function name simplified
    """
    Creates the machine learning pipeline for GridSearchCV.
    The pipeline structure remains the same.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            max_features=54)), # Keep the number of selected features the same
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))
    ])
    return pipeline


def main():
    # 1. Read and preprocess data
    target_column = 'Mitochondrion'
    df = pd.read_csv('Mitochondrion均衡样本.csv').dropna(subset=['Sequence', target_column])
    valid_sequences, valid_indices = preprocess_sequences(df['Sequence'])
    valid_labels = df.iloc[valid_indices][target_column]

    # 2. Feature extraction (using the modified extractor)
    print("Starting feature extraction (Markov features only)...")
    feature_extractor = MarkovFeatureExtractor(batch_size=500) # Use the new feature extractor class
    X = feature_extractor.extract_features(valid_sequences)
    y = valid_labels.values.astype(int)
    print("Feature extraction complete.")

    # 3. Split dataset and resample the training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nOversampling the training set to balance the data...")
    ros = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
    print(f"Number of samples in the balanced training set: {len(y_train_balanced)}")

    # 4. === GridSearchCV Integration ===
    # Use the same pipeline and parameter grid as before
    pipeline = create_pipeline()

    # Define the parameter grid to search (remains unchanged)
    param_grid = {
        'classifier__n_estimators': [200, 300],
        'classifier__max_depth': [25, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    # Set up GridSearchCV (remains unchanged)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )

    print("\nStarting GridSearchCV to find the best parameters...")
    grid_search.fit(X_train_balanced, y_train_balanced)

    # 5. === Results Output and Evaluation ===
    print("\nGridSearchCV finished.")
    print(f"Best parameter combination found: {grid_search.best_params_}")
    print(f"Best accuracy score in cross-validation: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nFinal performance evaluation on the test set (Markov features only):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

    # (Optional) Print feature importances (logic remains unchanged)
    if hasattr(best_model.named_steps['feature_selection'], 'estimator_'):
        feature_selector = best_model.named_steps['feature_selection']
        selected_feature_mask = feature_selector.get_support()
        
        # Note: feature_importances_ comes from the RF model in the feature selection step,
        # not from the final classifier's RF model.
        feature_importances = feature_selector.estimator_.feature_importances_
        feature_names = np.array(feature_extractor.feature_names)

        # Ensure the mask and feature name array lengths are consistent
        if len(feature_importances) != len(feature_names):
             print("\nWarning: The number of features in the feature selector does not match the number of original feature names. Skipping feature importance printing.")
        else:
            selected_importances = feature_importances[selected_feature_mask]
            selected_names = feature_names[selected_feature_mask]

            top_features = sorted(zip(selected_names, selected_importances), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop {len(top_features)} features selected by the best model:")
            for idx, (feature, importance) in enumerate(top_features[:100], 1):
                print(f"{idx}. {feature}: {importance:.6f}")


if __name__ == "__main__":
    main()
