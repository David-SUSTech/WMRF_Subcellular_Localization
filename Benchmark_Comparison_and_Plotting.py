#
# =============================================================================
# 0. Library Imports
# =============================================================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix)
from scipy.stats import wasserstein_distance
import collections
import seaborn as sns # Import seaborn for beautifying the confusion matrix

# Ensure necessary libraries are installed
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    print("Error: 'imbalanced-learn' library not found.")
    print("Please install it using 'pip install -U imbalanced-learn' and try again.")
    exit()

try:
    from hmmlearn import hmm
except ImportError:
    print("Error: 'hmmlearn' library not found.")
    print("Please install it using 'pip install hmmlearn' and try again.")
    exit()


# =============================================================================
# 1. Feature Extractor Definitions (Unchanged)
# =============================================================================

# --- Your Feature Extractor (AdvancedMarkovFeatureExtractor) ---
class AdvancedMarkovFeatureExtractor:
    def __init__(self, amino_acids='ACDEFGHIKLMNPQRSTVWY', max_sequence_length=2000, batch_size=500):
        self.amino_acids = amino_acids
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self):
        markov_names_1 = [f'markov_1_{i}' for i in range(len(self.amino_acids)**1)]
        markov_names_2 = [f'markov_2_{i}' for i in range(len(self.amino_acids)**2)]
        markov_names_3 = [f'markov_3_{i}' for i in range(len(self.amino_acids)**3)]
        wasserstein_names = ['standard_distance', 'median_distance', 'relative_distance', 'kl_divergence']
        return (markov_names_1 + markov_names_2 + markov_names_3 + wasserstein_names)

    def extract_features(self, sequences):
        all_features = []
        for i in range(0, len(sequences), self.batch_size):
            batch_sequences = sequences[i:i+self.batch_size]
            markov_features_1 = np.array([self._compute_markov_features(seq, order=1) for seq in batch_sequences])
            markov_features_2 = np.array([self._compute_markov_features(seq, order=2) for seq in batch_sequences])
            markov_features_3 = np.array([self._compute_markov_features(seq, order=3) for seq in batch_sequences])
            wasserstein_features = self._advanced_wasserstein_features(markov_features_2)
            batch_combined_features = np.hstack([markov_features_1, markov_features_2, markov_features_3, wasserstein_features])
            all_features.append(batch_combined_features)
        return np.vstack(all_features)
    
    def _compute_markov_features(self, sequence, order=2):
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
        try:
            return sum(self.amino_acids.index(aa) * (len(self.amino_acids) ** i) for i, aa in enumerate(reversed(state[:order])))
        except ValueError:
            return 0

    def _advanced_wasserstein_features(self, transition_matrices):
        try:
            reference_dist_mean = np.mean(transition_matrices, axis=0)
            reference_dist_median = np.median(transition_matrices, axis=0)
            wasserstein_features = []
            for matrix in transition_matrices:
                standard_distance = wasserstein_distance(matrix, reference_dist_mean)
                median_distance = wasserstein_distance(matrix, reference_dist_median)
                relative_distance = np.sum(np.abs(matrix - reference_dist_mean)) / (np.sum(reference_dist_mean) + 1e-10)
                kl_divergence = np.sum(matrix * np.log((matrix + 1e-10) / (reference_dist_mean + 1e-10)))
                wasserstein_features.append([standard_distance, median_distance, relative_distance, kl_divergence])
            return np.array(wasserstein_features, dtype=np.float32)
        except Exception as e:
            print(f"Wasserstein feature calculation error: {e}")
            return np.zeros((len(transition_matrices), 4), dtype=np.float32)

# --- Manually Implemented AAC and PseAAC Feature Extractors ---
class ManualFeatureExtractor:
    def __init__(self, method='AAC', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        self._AA_HYDRO_VALUES = {'A': 0.62, 'C': 0.29, 'D': -0.9, 'E': -0.74, 'F': 1.19, 'G': 0.48, 'H': -0.4, 'I': 1.38, 'K': -1.5, 'L': 1.06, 'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53, 'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26}
        self._AA_HYDROPHILICITY = {'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5, 'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8, 'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0, 'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3}
        self._AA_SIDE_CHAIN_MASS = {'A': 15, 'C': 47, 'D': 59, 'E': 73, 'F': 91, 'G': 1, 'H': 82, 'I': 57, 'K': 73, 'L': 57, 'M': 75, 'N': 58, 'P': 42, 'Q': 72, 'R': 100, 'S': 31, 'T': 45, 'V': 43, 'W': 130, 'Y': 107}
        self._AA_ISOELECTRIC_POINTS = {'A': 6.02, 'C': 5.02, 'D': 2.98, 'E': 3.08, 'F': 5.91, 'G': 6.06, 'H': 7.64, 'I': 6.04, 'K': 9.47, 'L': 6.04, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 6.02, 'W': 5.88, 'Y': 5.63}
        self._AA_POLARITY = {'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.3, 'F': 5.2, 'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9, 'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5, 'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2}

        self.physicochemical_properties = [self._AA_HYDRO_VALUES, self._AA_HYDROPHILICITY, self._AA_SIDE_CHAIN_MASS, self._AA_ISOELECTRIC_POINTS, self._AA_POLARITY]
        self.normalized_properties = self._normalize_properties()

    def _normalize_properties(self):
        normalized = []
        for prop_dict in self.physicochemical_properties:
            values = np.array([prop_dict[aa] for aa in self.amino_acids])
            mean = np.mean(values)
            std = np.std(values)
            norm_dict = {aa: (prop_dict[aa] - mean) / std for aa in self.amino_acids}
            normalized.append(norm_dict)
        return normalized

    def _calculate_aac(self, sequence):
        counts = collections.Counter(sequence)
        feature_vector = [counts.get(aa, 0) for aa in self.amino_acids]
        total = len(sequence)
        return [0.0] * 20 if total == 0 else [count / total for count in feature_vector]

    def _calculate_pse_aac(self, sequence, lamda=5, w=0.05):
        num_properties = len(self.normalized_properties)
        seq_len = len(sequence)
        if seq_len < lamda + 1: return [0.0] * (20 + lamda * num_properties)
        aac_freq = self._calculate_aac(sequence)
        thetas = []
        for i in range(1, lamda + 1):
            for prop_idx in range(num_properties):
                prop_dict = self.normalized_properties[prop_idx]
                theta = 0.0
                for j in range(seq_len - i):
                    aa1, aa2 = sequence[j], sequence[j+i]
                    if aa1 in prop_dict and aa2 in prop_dict:
                        correlation = (prop_dict[aa1] - prop_dict[aa2]) ** 2
                        theta += correlation
                thetas.append(theta / (seq_len - i))
        denominator = 1.0 + w * sum(thetas)
        pseaac_part1 = [freq / denominator for freq in aac_freq]
        pseaac_part2 = [(w * theta) / denominator for theta in thetas]
        return pseaac_part1 + pseaac_part2

    def extract_features(self, sequences):
        if self.method == 'AAC':
            return np.array([self._calculate_aac(seq) for seq in sequences])
        elif self.method == 'PseAAC':
            lamda, w = self.kwargs.get('lamda', 5), self.kwargs.get('w', 0.05)
            return np.array([self._calculate_pse_aac(seq, lamda=lamda, w=w) for seq in sequences])
    
# --- HMM Feature Extractor ---
class HMMExtractor:
    def __init__(self, n_components=3, n_iter=100, random_state=42):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_map = {aa: i for i, aa in enumerate(self.amino_acids)}

    def extract_features(self, sequences, labels):
        pos_sequences = [s for s, l in zip(sequences, labels) if l == 1]
        neg_sequences = [s for s, l in zip(sequences, labels) if l == 0]
        pos_model = self._train_hmm(pos_sequences)
        neg_model = self._train_hmm(neg_sequences)
        features = []
        for seq in sequences:
            seq_num = self._sequence_to_numerical(seq)
            if seq_num is None: pos_score, neg_score = 0.0, 0.0
            else:
                try: pos_score, neg_score = pos_model.score(seq_num), neg_model.score(seq_num)
                except Exception: pos_score, neg_score = 0.0, 0.0
            features.append([pos_score, neg_score])
        return np.array(features)

    def _sequence_to_numerical(self, sequence):
        seq_num = [self.aa_map[aa] for aa in sequence if aa in self.aa_map]
        return np.array(seq_num).reshape(-1, 1) if seq_num else None

    def _train_hmm(self, sequences):
        sequences_numerical = [self._sequence_to_numerical(s) for s in sequences if s]
        sequences_numerical = [s for s in sequences_numerical if s is not None]
        model = hmm.GaussianHMM(n_components=self.n_components, n_iter=self.n_iter, random_state=self.random_state, covariance_type="diag")
        if not sequences_numerical:
            model.fit(np.array([[0.0]]*self.n_components))
            return model
        lengths = [len(s) for s in sequences_numerical]
        all_sequences_concat = np.concatenate(sequences_numerical)
        model.fit(all_sequences_concat, lengths)
        return model

# =============================================================================
# 2. Helper Functions and Workflow Control
# =============================================================================
def preprocess_sequences(sequences, min_length=10, max_length=2000):
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    processed_data, original_indices = [], []
    for idx, seq in enumerate(sequences):
        if isinstance(seq, str) and min_length <= len(seq) <= max_length and all(aa in valid_amino_acids for aa in seq):
            processed_data.append(seq[:max_length])
            original_indices.append(idx)
    return processed_data, original_indices

def run_experiment(features, y_labels, method_name):
    """
    A unified experiment function that now returns a dictionary with all necessary results.
    """
    print(f"\n{'='*25}\nRunning experiment for: {method_name}\n{'='*25}")
    start_time = time.time()

    X_train, X_test, y_train, y_test = train_test_split(features, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    ros = RandomOverSampler(random_state=42)
    X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

    if method_name == "WMRF":
        pipeline = Pipeline([('scaler', StandardScaler()), ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), max_features=54)), ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))])
        param_grid = {'classifier__n_estimators': [200, 300], 'classifier__max_depth': [25, 30], 'classifier__min_samples_split': [2, 5], 'classifier__min_samples_leaf': [1, 2]}
    else:
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1))])
        param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, 20, None]}
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)
    best_model = grid_search.best_estimator_
    
    print(f"Best params found: {grid_search.best_params_}")
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    total_time = time.time() - start_time

    acc, prec, rec, f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n--- Results for {method_name} ---\nTime: {total_time:.2f}s | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC AUC: {roc_auc:.4f}")

    return {
        'method': method_name, 'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'time': total_time,
        'metrics': {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}
    }


# =============================================================================
# 3. Visualization Functions (Restoring the full suite)
# =============================================================================

def set_pub_style():
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'], 'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'axes.titleweight': 'bold', 'axes.labelweight': 'bold', 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'figure.titlesize': 20, 'figure.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5})
    
def get_color_palette(num_colors):
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
    return colors[:num_colors] if num_colors <= len(colors) else plt.cm.get_cmap('viridis', num_colors).colors
    
def plot_curves(results):
    set_pub_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    colors = get_color_palette(len(results))
    results.sort(key=lambda x: x['method'] != 'WMRF') # Ensure WMRF is plotted first/prominently
    for i, result in enumerate(results):
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba']); roc_auc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"{result['method']} (AUC = {roc_auc_val:.3f})")
        precision, recall, _ = precision_recall_curve(result['y_test'], result['y_pred_proba']); pr_auc_val = auc(recall, precision)
        ax2.plot(recall, precision, color=colors[i], lw=2.5, label=f"{result['method']} (AUPRC = {pr_auc_val:.3f})")
    ax1.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--', label='Random Chance'); ax1.set_xlim([-0.05, 1.05]); ax1.set_ylim([-0.05, 1.05]); ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate'); ax1.set_title('a. Receiver Operating Characteristic (ROC)', loc='left'); ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.8, edgecolor='black'); ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_xlim([-0.05, 1.05]); ax2.set_ylim([-0.05, 1.05]); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('b. Precision-Recall (PR)', loc='left'); ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.8, edgecolor='black'); ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=2.0); plt.savefig("ROC_PR_Curves_Membrane.png", dpi=600, bbox_inches='tight'); plt.savefig("ROC_PR_Curves_Membrane.pdf", bbox_inches='tight'); plt.show()

def plot_confusion_matrices(results):
    set_pub_style()
    method_order = ["WMRF", "PseAAC", "AAC", "HMM"]; results_map = {res['method']: res for res in results}
    if not all(method in results_map for method in method_order):
        print("Error: Missing results for some methods, cannot plot confusion matrices."); return
    ordered_results = [results_map[method] for method in method_order]
    fig, axes = plt.subplots(2, 2, figsize=(12, 11)); axes = axes.flatten()
    for i, result in enumerate(ordered_results):
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], annot_kws={"size": 16}, cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        axes[i].set_title(result['method'], loc='center', fontsize=18); axes[i].set_xlabel('Predicted Label'); axes[i].set_ylabel('True Label')
    fig.suptitle('Confusion Matrices of Different Methods', fontsize=22, y=1.02); plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0); plt.savefig("Confusion_Matrices_Membrane.png", dpi=600, bbox_inches='tight'); plt.savefig("Confusion_Matrices_Membrane.pdf", bbox_inches='tight'); plt.show()
    
def plot_radar_chart(results):
    set_pub_style()
    metrics, method_order = ['Accuracy', 'Precision', 'Recall', 'F1 Score'], ["WMRF", "PseAAC", "AAC", "HMM"]
    results_map = {res['method']: res for res in results}
    if not all(method in results_map for method in method_order):
        print("Error: Missing results for some methods, cannot plot radar chart."); return
    ordered_results = [results_map[method] for method in method_order]
    data = [[res['metrics'][m] for m in metrics] for res in ordered_results]; data_closed = [d + [d[0]] for d in data]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True)); ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, weight='bold'); ax.set_ylim(0, 1.0); ax.set_rlabel_position(22.5); ax.yaxis.set_tick_params(labelsize=10)
    colors = get_color_palette(len(ordered_results))
    for i, res in enumerate(ordered_results):
        ax.plot(angles, data_closed[i], color=colors[i], linewidth=2, linestyle='solid', label=res['method']); ax.fill(angles, data_closed[i], color=colors[i], alpha=0.2)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)); ax.set_title('Performance Comparison Radar Chart', size=20, y=1.1); plt.tight_layout(pad=2.0); plt.savefig("Radar_Chart_Membrane.png", dpi=600, bbox_inches='tight'); plt.savefig("Radar_Chart_Membrane.pdf", bbox_inches='tight'); plt.show()


# =============================================================================
# 4. Main Program Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        # *********** Modification Point: Dataset Switch ***********
        target_column = 'Membrane'
        df = pd.read_csv('Membrane均衡样本.csv').dropna(subset=['Sequence', target_column]) #Alter 'Membrane' with other names, such as Mitochondrion, Cytoplasm, etc
    except FileNotFoundError:
        print(f"\nError: Could not find 'Membrane均衡样本.csv' file. Please ensure it is in the same directory as the script."); exit()
        
    print("Step 0: Preprocessing all sequences...")
    valid_sequences, valid_indices = preprocess_sequences(df['Sequence'])
    valid_labels = df.iloc[valid_indices][target_column].values.astype(int)
    print(f"Found {len(valid_sequences)} valid sequences.")

    print("\nPre-extracting features for all methods...")
    extractors = {"WMRF": AdvancedMarkovFeatureExtractor(batch_size=500), "PseAAC": ManualFeatureExtractor(method='PseAAC', lamda=5, w=0.05), "AAC": ManualFeatureExtractor(method='AAC'), "HMM": HMMExtractor(n_components=3)}
    features_dict = {}
    for name, extractor in extractors.items():
        print(f"Extracting features for {name}...")
        features_dict[name] = extractor.extract_features(valid_sequences, valid_labels) if name == "HMM" else extractor.extract_features(valid_sequences)
        print(f"Done. Feature shape: {features_dict[name].shape}")

    results_list = []
    method_run_order = ["WMRF", "PseAAC", "AAC", "HMM"] 
    for name in method_run_order:
        if name in features_dict:
            results_list.append(run_experiment(features_dict[name], valid_labels, name))

    print(f"\n{'='*30}\nExecution Time Summary\n{'='*30}"); results_list.sort(key=lambda x: x['method'])
    for res in results_list: print(f"{res['method']:<10}: {res['time']:.2f} seconds")

    if results_list:
        print("\nPlotting ROC and PR curves..."); plot_curves(results_list)
        print("\nPlotting confusion matrices..."); plot_confusion_matrices(results_list)
        print("\nPlotting performance radar chart..."); plot_radar_chart(results_list)
        print("\nAll tasks complete. Charts saved as PNG and PDF files.")
    else:
        print("\nNo results were generated, skipping plotting.")
