import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Reshape, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import os

import pyedflib
import numpy as np



def _read_py_function(filename, num_channels=64):
    # Open the EDF file
    f = pyedflib.EdfReader(filename)

    # Get the number of channels and the signal labels
    n_channels = f.signals_in_file
    #print(f"Total channels in file: {n_channels}")

    # Ensure that we are fetching only the desired number of channels (16 in this case)
    if num_channels > n_channels:
        raise ValueError(f"The file contains only {n_channels} channels, but {num_channels} were requested.")

    # Initialize eeg_data to store only the first `num_channels`
    eeg_data = np.zeros((num_channels, f.getNSamples()[0]), dtype=np.float32)

    # Read only the first `num_channels` channels
    for i in range(num_channels):
        eeg_data[i, :] = f.readSignal(i)

    n_samples = f.getNSamples()[0]
    reminder = int(n_samples % 160)

    # Print statement to check values
    print(f"Original n_samples: {n_samples}")
    print(f"Reminder: {reminder}")

    n_samples -= reminder
    seconds = int(n_samples / 160)

    # Extract person_id from the last directory name
    try:
        person_id = int(os.path.basename(os.path.dirname(filename))[1:])  # Skip 'S' and convert to int
    except ValueError as e:
        raise ValueError(f"Invalid person ID extracted from filename: {filename}. Error: {e}")

    # Create one-hot encoded labels
    label = np.zeros(109, dtype=bool)  # Adjust size according to the number of subjects
    if person_id - 1 < len(label):
        label[person_id - 1] = 1
    else:
        raise ValueError(f"Person ID {person_id} exceeds the number of defined classes.")


    labels = np.tile(label, (seconds, 1))

    # Normalization step
    for i in range(num_channels):
        channel_data = eeg_data[i, :]
        mean_i = np.mean(channel_data)  # Compute mean of channel i
        std_i = np.std(channel_data)    # Compute standard deviation of channel i

        if std_i != 0:  # Avoid division by zero
            eeg_data[i, :] = (channel_data - mean_i) / std_i  # Normalize channel i

    # Transpose the data to shape (n_samples, n_channels)
    eeg_data = eeg_data.T

    if reminder > 0:
        eeg_data = eeg_data[:-reminder, :]

    # Split the data into 160-sample chunks
    intervals = np.linspace(0, n_samples, num=seconds, endpoint=False, dtype=int)
    eeg_data = np.split(eeg_data, intervals)
    del eeg_data[0]  # Remove the first element which is an empty array
    eeg_data = np.array(eeg_data)

    return eeg_data, labels


import os

import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_eeg_data_with_pyedflib(data_path, max_subjects=109):
    # Initialize the global variables
    global subject_data, all_train_data, all_train_labels, all_test_data, all_test_labels
    global train_data, val_data, train_labels, val_labels

    # Initialize the variables before using them
    subject_data = {}
    all_train_data = []
    all_train_labels = []
    all_test_data = []
    all_test_labels = []

    subject_count = 0  # Counter to track the number of subjects processed

    for subject_dir in os.listdir(data_path):
        subject_path = os.path.join(data_path, subject_dir)
        if os.path.isdir(subject_path):
            print(f"Processing subject: {subject_dir}")
            subject_data[subject_dir] = {}
            for recording_file in os.listdir(subject_path):
                recording_path = os.path.join(subject_path, recording_file)
                if recording_file.endswith('.edf'):
                    recording_id = os.path.splitext(recording_file)[0]
                    print(f"Loading recording: {recording_file}")

                    try:
                        eeg_data, labels = _read_py_function(recording_path)
                        print(f"Data shape after processing: {eeg_data.shape}")
                        print(f"Labels shape: {labels.shape}")

                        train_data, test_data, train_labels, test_labels = train_test_split(
                            eeg_data, labels, test_size=0.1, random_state=42
                        )

                        all_train_data.append(train_data)
                        all_train_labels.append(train_labels)
                        all_test_data.append(test_data)
                        all_test_labels.append(test_labels)

                    except Exception as e:
                        print(f"Error loading {recording_file}: {e}")

            subject_count += 1  # Increment the subject counter

            # Stop processing once the desired number of subjects is reached
            if subject_count >= max_subjects:
                print(f"Processed {max_subjects} subjects. Stopping further processing.")
                break

    print(f"Train data list length: {len(all_train_data)}")
    print(f"Train labels list length: {len(all_train_labels)}")
    print(f"Test data list length: {len(all_test_data)}")
    print(f"Test labels list length: {len(all_test_labels)}")

    if all_train_data and all_train_labels:
        all_train_data = np.concatenate(all_train_data, axis=0)
        all_train_labels = np.concatenate(all_train_labels, axis=0)
    else:
        print("No training data loaded.")
        return None, None, None, None, None, None

    if all_test_data and all_test_labels:
        all_test_data = np.concatenate(all_test_data, axis=0)
        all_test_labels = np.concatenate(all_test_labels, axis=0)
    else:
        print("No test data loaded.")
        return None, None, None, None, None, None

    if len(all_train_data) == 0 or len(all_train_labels) == 0:
        print("No data available for splitting into training and validation sets.")
        return None, None, None, None, None, None

    # Split train data further into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        all_train_data, all_train_labels, test_size=0.25, random_state=42
    )

    return train_data, train_labels, val_data, val_labels, all_test_data, all_test_labels


data_path = '/mnt/beegfs/home/akhileshsing2024/mywork/eeg-motor-movementimagery-dataset-1.0.0/files'
train_data, train_laels, val_data, val_labels, test_data, test_labels = load_eeg_data_with_pyedflib(data_path)
# Access global vaables directly if n
print(f"Train data shape: {train_data.shape if train_data is not None else 'None'}")
print(f"Validation data shape: {val_data.shape if val_data is not None else 'None'}")
print(f"Test data shape: {test_data.shape if test_data is not None else 'None'}")



import tensorflow as tf

def eeg_biometric_identification_model(input_shape, n_classes, lstm_size=192, keep_prob=0.5):
    inputs = tf.keras.Input(shape=input_shape)

    # 1D Convolutional Layers
    conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same', activation='relu')(inputs)
    conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=2, strides=1, padding='same', activation='relu')(conv1)
    conv3 = tf.keras.layers.Conv1D(filters=512, kernel_size=2, strides=1, padding='same', activation='relu')(conv2)
    conv4 = tf.keras.layers.Conv1D(filters=1024, kernel_size=2, strides=1, padding='same', activation='relu')(conv3)

    # Flatten the convolution output before feeding into fully connected layers
    flatten = tf.keras.layers.Flatten()(conv4)

    # Fully connected layer with 192 neurons and dropout
    fc1 = tf.keras.layers.Dense(units=192, activation='relu')(flatten)
    dropout_fc1 = tf.keras.layers.Dropout(rate=1 - keep_prob)(fc1)

    # Reshape to (batch_size, timesteps=1, features=192) for LSTM input
    lstm_input = tf.keras.layers.Reshape((1, 192))(dropout_fc1)  

    # Add LSTM layers
    lstm_out = tf.keras.layers.LSTM(units=lstm_size, return_sequences=True)(lstm_input)  # First LSTM layer
    lstm_out = tf.keras.layers.LSTM(units=lstm_size, return_sequences=False)(lstm_out)  # Second LSTM layer

    # Fully connected layers after LSTM
    fc2 = tf.keras.layers.Dense(units=192, activation='relu')(lstm_out)
    fc3 = tf.keras.layers.Dense(units=192, activation='relu')(fc2)

    # Output layer (Softmax for classification)
    output = tf.keras.layers.Dense(n_classes, activation='softmax')(fc3)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

# Example usage
input_shape = (160, 64)  # 160 samples, 64 channels (1-second EEG recording)
n_classes = 109  # Number of subjects
keep_prob = 0.5  # Dropout rate

model = eeg_biometric_identification_model(input_shape, n_classes, lstm_size=192, keep_prob=keep_prob)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Summary
model.summary()

# model = eeg_biometric_identification_model(input_shape, n_classes, lstm_size=192, lstm_layers=2, keep_prob=keep_prob)

# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
#               loss='categorical_crossentropy', 
#               metrics=['accuracy'])

# # Display the model summary
# model.summary()



import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Model Initialization (assuming CNN_LSTM model is defined elsewhere)
# model = CNN_LSTM()

# Optimizer and Model Compilation

# Checkpoint configuration
checkpoint_path = "cp-{epoch:04d}.weights.h5"  # Save checkpoints in the current directory
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Save the initial weights before training (optional)
model.save_weights(checkpoint_path.format(epoch=0))

# Training the model
with tf.device('/GPU:0'):  # Ensure training uses the GPU
    history = model.fit(train_data, 
                        train_labels, 
                        epochs=220, 
                        validation_data=(val_data, val_labels), 
                        batch_size=80,  
                        callbacks=[cp_callback])

# Save the training history as a numpy file
np.save("history.npy", history.history)

# Load the training history
history = np.load("history.npy", allow_pickle=True).item()

# Ensure history is a dictionary
if type(history) is not dict:
    history = history.history

# Find the epoch with the highest validation accuracy
max_value = max(history['val_accuracy'])
print(f"Max Validation Accuracy: {max_value}")

max_index = history['val_accuracy'].index(max_value)
print(f"Best Epoch Index: {max_index}")
print(f"Corresponding Training Accuracy: {history['accuracy'][max_index]}")

# Load the best checkpoint based on validation accuracy
best_checkpoint_path = "cp-{:04d}.weights.h5".format(max_index)
model.load_weights(best_checkpoint_path)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x=test_data, y=test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.savefig("accuracy_plot.png")  # Save the accuracy plot
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("loss_plot.png")  # Save the loss plot
plt.show()
