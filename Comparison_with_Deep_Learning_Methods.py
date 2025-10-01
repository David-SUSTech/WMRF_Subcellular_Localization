import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Embedding, Flatten, Bidirectional, LSTM,
    Attention, Conv1D, MaxPooling1D, Input, Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import time

# 1. Load data
data_path = "Cytoplasm均衡样本.csv"
data = pd.read_csv(data_path)

# 2. Extract sequences and labels
sequences = data['Sequence']
labels = data['Cytoplasm']

# 3. Label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(sequences, encoded_labels, test_size=0.2, random_state=42)

# 5. Define sequence length
max_length = 1000

# 6. Define models

# FFN model
def create_ffn_model(input_length, num_classes):
    inputs = Input(shape=(input_length,))
    x = Embedding(input_dim=256, output_dim=64)(inputs)  # Adjusted input_dim for ASCII range
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add Dropout for regularization
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# BLSTM model
def create_blstm_model(input_length, num_classes):
    inputs = Input(shape=(input_length,))
    x = Embedding(input_dim=256, output_dim=64)(inputs)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.5)(x)  # Add Dropout for regularization
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# A-BLSTM model
def create_ab_lstm_model(input_length, num_classes):
    inputs = Input(shape=(input_length,))
    x = Embedding(input_dim=256, output_dim=64)(inputs)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Attention layer
    query = x
    key_value = x
    attention_out = Attention()([query, key_value])
    flatten = Flatten()(attention_out)
    flatten = Dropout(0.5)(flatten)  # Add Dropout for regularization
    outputs = Dense(num_classes, activation='softmax')(flatten)

    model = Model(inputs, outputs)
    return model

# Conv A-BLSTM model
def create_conv_ab_lstm_model(input_length, num_classes):
    inputs = Input(shape=(input_length,))
    x = Embedding(input_dim=256, output_dim=64)(inputs)
    x = Conv1D(64, 5, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Attention layer
    query = x
    key_value = x
    attention_out = Attention()([query, key_value])
    flatten = Flatten()(attention_out)
    flatten = Dropout(0.5)(flatten)  # Add Dropout for regularization
    outputs = Dense(num_classes, activation='softmax')(flatten)

    model = Model(inputs, outputs)
    return model

# 7. Train and evaluate model function
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Pad sequences to ensure consistent shape
    def pad_sequences(sequences, max_length):
        padded_sequences = np.zeros((len(sequences), max_length), dtype=np.float32)
        for i, seq in enumerate(sequences):
            for j, char in enumerate(seq[:max_length]):
                padded_sequences[i, j] = ord(char)
        return padded_sequences
    
    X_train_padded = pad_sequences(X_train, max_length)
    X_test_padded = pad_sequences(X_test, max_length)

    # One-hot encode labels
    y_train_categorical = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test_categorical = to_categorical(y_test, num_classes=len(label_encoder.classes_))

    # Compile the model with a lower learning rate
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model with increased epochs
    start_time = time.time()
    model.fit(X_train_padded, y_train_categorical, epochs=100, batch_size=32, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    end_time = time.time()

    print(f"Model: {model.name} training time: {end_time - start_time:.2f} seconds")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test_categorical)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 8. Create and evaluate each model
models = {
    "FFN": create_ffn_model(max_length, len(label_encoder.classes_)),
    "BLSTM": create_blstm_model(max_length, len(label_encoder.classes_)),
    "A-BLSTM": create_ab_lstm_model(max_length, len(label_encoder.classes_)),
    "Conv-A-BLSTM": create_conv_ab_lstm_model(max_length, len(label_encoder.classes_)),
}

for model_name, model in models.items():
    print(f"Training model: {model_name}")
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
