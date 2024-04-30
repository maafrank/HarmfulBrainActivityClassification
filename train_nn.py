# Imports
import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras_cv
import glob

from random import randint
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
import joblib



#tf.debugging.set_log_device_placement(False)
#MAX_GPU_MEM_MB = 6144
#gpus = tf.config.experimental.list_physical_devices('GPU')
#print(f"Identified GPUs on system: {gpus}")
#if gpus:
#  try:
#    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MAX_GPU_MEM_MB)])
#    print(f"Limited GPU memory to {MAX_GPU_MEM_MB}")
#  except RuntimeError as e:
#    print(e)

#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
#tf.config.experimental.set_virtual_device_configuration(tf.config.list_physical_devices('GPU')[0],
#[tf.config.LogicalDeviceConfiguration(memory_limit=4096)])


# Dataset directories
# NOTE: make sure to change these based on where the data is stored
BASE_PATH = "../hms-harmful-brain-activity-classification"
TRAIN_EEG_PATH = os.path.join(BASE_PATH, "train_eegs", "")
TRAIN_SPECT_PATH = os.path.join(BASE_PATH, "train_spectrograms", "")
TEST_EEG_PATH = os.path.join(BASE_PATH, "test_eegs", "")
TEST_SPECT_PATH = os.path.join(BASE_PATH, "test_spectrograms", "")

# Reading the CSV for train and test set labels from the competition
df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

# Showing the head and size of the training dataset to check if it loaded properly
print(df.head())
print(f"Training labels dataframe shape: {df.shape}")

# Configuration for classifiers
class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = "efficientnetv2_b2_imagenet"  # Name of pretrained classifier
    image_size = [400, 300]  # Input image size
    epochs = 100 # Training epochs
    batch_size = 32  # Batch size
    lr_mode = "cos" # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6 # Number of classes in the dataset
    fold = 0 # Which fold to set as validation data
    class_names = ['Seizure', 'LPD', 'GPD', 'LRDA','GRDA', 'Other']
    label2name = dict(enumerate(class_names))
    name2label = {v:k for k, v in label2name.items()}


BASE_PATH = "../hms-harmful-brain-activity-classification"

# Train + Valid
df = pd.read_csv(f'{BASE_PATH}/train.csv')
df['eeg_path'] = f'{BASE_PATH}/train_eegs/'+df['eeg_id'].astype(str)+'.parquet'
df['spec_path'] = f'{BASE_PATH}/train_spectrograms/'+df['spectrogram_id'].astype(str)+'.parquet'
df['spec2_path'] = f'{BASE_PATH}/SPEC_DIR/train_spectrograms/'+df['spectrogram_id'].astype(str)+'.npy'
df['class_name'] = df.expert_consensus.copy()
df['class_label'] = df.expert_consensus.map(CFG.name2label)
#print(df.head(2))

# Test
test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
test_df['eeg_path'] = f'{BASE_PATH}/test_eegs/'+test_df['eeg_id'].astype(str)+'.parquet'
test_df['spec_path'] = f'{BASE_PATH}/test_spectrograms/'+test_df['spectrogram_id'].astype(str)+'.parquet'
test_df['spec2_path'] = f'{BASE_PATH}/SPEC_DIR/test_spectrograms/'+test_df['spectrogram_id'].astype(str)+'.npy'
#print(test_df.head(2))


# Define a function to process a single eeg_id
SPEC_DIR = BASE_PATH + "/SPEC_DIR"

if not os.path.exists(SPEC_DIR):
    os.makedirs(SPEC_DIR, exist_ok=True)

def process_spec(spec_id, split="train"):
    # Ensure the subdirectory exists
    spec_dir_path = f"{SPEC_DIR}/{split}_spectrograms"
    if not os.path.exists(spec_dir_path):
        os.makedirs(spec_dir_path, exist_ok=True)
    
    # Process the spectrogram
    spec_path = f"{BASE_PATH}/{split}_spectrograms/{spec_id}.parquet"
    spec = pd.read_parquet(spec_path)
    spec = spec.fillna(0).values[:, 1:].T  # fill NaN values with 0, transpose for (Freq, Time)
    spec = spec.astype("float32")
    
    # Save the processed spectrogram
    np.save(f"{spec_dir_path}/{spec_id}.npy", spec)

# Example usage with df
spec_ids = df["spectrogram_id"].unique()
test_spec_ids = test_df["spectrogram_id"].unique()

# NOTE: uncomment when re-creating NPYs
# for spec_id in tqdm(spec_ids):
#    process_spec(spec_id, split="train")

# for spec_id in tqdm(test_spec_ids):
#    process_spec(spec_id, split="test")


def build_augmenter(dim=CFG.image_size):
    augmenters = [
        keras_cv.layers.MixUp(alpha=2.0),
        keras_cv.layers.RandomCutout(height_factor=(1.0, 1.0),
                                     width_factor=(0.06, 0.1)), # freq-masking
        keras_cv.layers.RandomCutout(height_factor=(0.06, 0.1),
                                     width_factor=(1.0, 1.0)), # time-masking
    ]
    
    def augment(img, label):
        data = {"images":img, "labels":label}
        for augmenter in augmenters:
            if tf.random.uniform([]) < 0.5:
                data = augmenter(data, training=True)
        return data["images"], data["labels"]
    
    return augment


def build_decoder(with_labels=True, target_size=CFG.image_size, dtype=32):
    def decode_signal(path, offset=None):
        # Read .npy files and process the signal
        # Using tf to load is because its prob faster
        file_bytes = tf.io.read_file(path)
        sig = tf.io.decode_raw(file_bytes, tf.float32)
        sig = sig[1024//dtype:]  # Remove header tag
        sig = tf.reshape(sig, [400, -1])
        
        # Extract labeled subsample from full spectrogram using "offset"
        if offset is not None: 
            offset = offset // 2  # Only odd values are given
            sig = sig[:, offset:offset+300]
            
            # Pad spectrogram to ensure the same input shape of [400, 300]
            pad_size = tf.math.maximum(0, 300 - tf.shape(sig)[1])
            sig = tf.pad(sig, [[0, 0], [0, pad_size]])
            sig = tf.reshape(sig, [400, 300])
        
        # Log spectrogram 
        sig = tf.clip_by_value(sig, tf.math.exp(-4.0), tf.math.exp(8.0)) # avoid 0 in log
        sig = tf.math.log(sig)
        
        # Normalize spectrogram
        sig -= tf.math.reduce_mean(sig)
        sig /= tf.math.reduce_std(sig) + 1e-6
        
        # Mono channel to 3 channels to use "ImageNet" weights
        sig = tf.tile(sig[..., None], [1, 1, 3])
        return sig
    
    def decode_label(label):
        label = tf.one_hot(label, CFG.num_classes)
        label = tf.cast(label, tf.float32)
        label = tf.reshape(label, [CFG.num_classes])
        return label
    
    def decode_with_labels(path, offset=None, label=None):
        sig = decode_signal(path, offset)
        label = decode_label(label)
        return (sig, label)
    
    return decode_with_labels if with_labels else decode_signal

import shutil

def build_dataset(paths, offsets=None, labels=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=False, repeat=True, shuffle=1024, 
                  cache_dir="", drop_remainder=False):
    # Reset cache dir
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter()
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = (paths, offsets) if labels is None else (paths, offsets, labels)
    
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds


from sklearn.model_selection import StratifiedGroupKFold

#sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=CFG.seed)
#
#df["fold"] = -1
#df.reset_index(drop=True, inplace=True)
#for fold, (train_idx, valid_idx) in enumerate(sgkf.split(df, y=df["class_label"], groups=df["patient_id"])):
#    df.loc[valid_idx, "fold"] = fold
#
#
## Sample from full data
#train_folds = [0,1,2] #[0, 1, 2]  # Folds used for training
#valid_fold = 3           # Fold used for validation
#test_fold = 4            # Fold used for testing
#
## Split the DataFrame into train, valid, and test sets based on folds
#train_df = df[df.fold.isin(train_folds)]
#valid_df = df[df.fold == valid_fold]
#test_df = df[df.fold == test_fold]


# Desired sizes of your datasets
train_size = 10000
valid_size = 5000
test_size = 5000

#train_size = 75000
#valid_size = 10000
#test_size = 10000

# Initialize lists to hold data for each set
train_data = []
valid_data = []
test_data = []

# Number of classes
num_classes = df['class_label'].nunique()

# Calculate samples per class for each dataset
samples_per_class_train = train_size // num_classes
samples_per_class_valid = valid_size // num_classes
samples_per_class_test = test_size // num_classes

# For reproducibility
np.random.seed(CFG.seed)

# Split and sample
for _, group in df.groupby('class_label'):
    # Shuffle the group to randomize selection
    group = group.sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
    
    # Append data for train
    train_data.append(group.iloc[:samples_per_class_train])
    
    # Update index for valid
    valid_start_idx = samples_per_class_train
    valid_end_idx = valid_start_idx + samples_per_class_valid
    valid_data.append(group.iloc[valid_start_idx:valid_end_idx])
    
    # Update index for test
    test_start_idx = valid_end_idx
    test_end_idx = test_start_idx + samples_per_class_test
    test_data.append(group.iloc[test_start_idx:test_end_idx])

# Concatenate lists into DataFrames
train_df = pd.concat(train_data).sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
valid_df = pd.concat(valid_data).sample(frac=1, random_state=CFG.seed).reset_index(drop=True)
test_df = pd.concat(test_data).sample(frac=1, random_state=CFG.seed).reset_index(drop=True)

print("Training Data Class Distribution:")
print(train_df['class_label'].value_counts())

print("\nValidation Data Class Distribution:")
print(valid_df['class_label'].value_counts())

print("\nTest Data Class Distribution:")
print(test_df['class_label'].value_counts())


# Display number of samples in each set
print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)} | Num Test: {len(test_df)}")

# Train
train_paths = train_df.spec2_path.values
train_offsets = train_df.spectrogram_label_offset_seconds.values.astype(int)
train_labels = train_df.class_label.values
train_ds = build_dataset(train_paths, train_offsets, train_labels, batch_size=CFG.batch_size,
                         repeat=True, shuffle=1024, augment=False, cache=True, cache_dir='cache_train/')
# Valid
valid_paths = valid_df.spec2_path.values
valid_offsets = valid_df.spectrogram_label_offset_seconds.values.astype(int)
valid_labels = valid_df.class_label.values
valid_ds = build_dataset(valid_paths, valid_offsets, valid_labels, batch_size=CFG.batch_size,
                         repeat=True, shuffle=False, augment=False, cache=True, cache_dir='cache_valid/')

# Test
test_paths = test_df.spec2_path.values
test_offsets = test_df.spectrogram_label_offset_seconds.values.astype(int)
test_labels = test_df.class_label.values
test_ds = build_dataset(test_paths, test_offsets, test_labels, batch_size=CFG.batch_size,
                        repeat=True, shuffle=False, augment=False, cache=True, cache_dir='cache_test/')


from tensorflow.keras import layers, models

# Set the seed for numpy and TensorFlow so we initialize the model the same everytime
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras import Model
def build_transfer_model(backbone='ResNet50', input_shape=CFG.image_size, num_classes=CFG.num_classes, freeze_base=False):
    if backbone == "BaseLine":
        model = models.Sequential([
            layers.InputLayer(shape=input_shape),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
    else:
        BaseModel = getattr(tf.keras.applications, backbone)
        base_model = BaseModel(include_top=False, weights='imagenet', input_shape=input_shape)
    
        # Optionally freeze the layers of the base_model
        if freeze_base:
            for layer in base_model.layers:
                layer.trainable = False
                
        inputs = tf.keras.Input(shape=input_shape)
        
        # Pass inputs through the base model
        x = base_model(inputs, training=not freeze_base)  # training=False ensures that the base_model runs in inference mode
        
        # Add additional layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # Construct the final model
        model = Model(inputs=inputs, outputs=outputs)

    return model

import math

def get_lr_callback(batch_size=CFG.batch_size, mode='cos', epochs=CFG.epochs):
    lr_start, lr_max, lr_min = 5e-5, 6e-6 * batch_size, 1e-5
    #lr_start, lr_max, lr_min = 0.00016308507842895508, 6e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep: lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep: lr = lr_max
        elif mode == 'exp': lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        elif mode == 'step': lr = lr_max * lr_decay**((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == 'cos':
            decay_total_epochs, decay_epoch_index = epochs - lr_ramp_ep - lr_sus_ep + 3, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import gc

num_train_samples = len(train_df)
num_valid_samples = len(valid_df)
num_test_samples = len(test_df)

train_steps = math.ceil(num_train_samples / CFG.batch_size)
valid_steps = math.ceil(num_valid_samples / CFG.batch_size)
test_steps = math.ceil(num_test_samples / CFG.batch_size)

def load_existing_results(results_path):
    try:
        results_df = pd.read_csv(results_path)
        return results_df.to_dict('records')  # Convert DataFrame to list of dicts
    except FileNotFoundError:
        return []  # If the file does not exist, start with an empty list

def is_model_config_trained(existing_results, bb, fb, aug, lr):
    for res in existing_results:
        if res['BackBone'] == bb:
            if res['FreezeBase'] == fb:
                if res['Augment'] == aug:
                    if res['LearningRate'] == lr:
                        return True
    return False


from tensorflow.keras.models import load_model

# Load the existing results
results_list = load_existing_results('results/Results.csv')
print(pd.DataFrame(results_list))

backbone = ['BaseLine', 'ResNet50', 'MobileNetV3Small', 'EfficientNetV2S', 'EfficientNetB3', 'VGG16']
freeze_base = [True, False]
augment = [True, False]
learning_rate = ['0.01', '0.001', '0.0001', '0.00001', 'lr_callback', ] #'lr_callback_all']
    
for fb in freeze_base:
    for bb in backbone:
        for aug in augment:
            for lr in learning_rate:

                tf.keras.backend.clear_session()
                
                print(f'{bb=} {fb=} {aug=} {lr=}')

                # Check if this configuration has been trained before
                if is_model_config_trained(results_list, bb, fb, aug, lr):
                    print(f"Skipping already trained configuration: {bb}, {fb}, {aug}, {lr}")
                    continue

                train_ds = build_dataset(train_paths, train_offsets, train_labels, batch_size=CFG.batch_size,
                                         repeat=True, shuffle=1024, augment=aug, cache=True, cache_dir='cache_train/')

                # Model re-initialized here for each learning rate
                model = build_transfer_model(backbone=bb,
                                             input_shape=CFG.image_size + [3],
                                             num_classes=CFG.num_classes,
                                             freeze_base=fb)


                if not os.path.exists("weights"): os.makedirs("weights")
                patience = CFG.epochs / 10
                callbacks = [
                    EarlyStopping(patience=patience, verbose=1, monitor='val_loss', mode='min', restore_best_weights=True),
                    ModelCheckpoint(f'weights/{bb}.{fb}.{aug}.{lr}.keras', verbose=1, save_best_only=True, monitor='val_loss', mode='min')
                ]
                
                if str(lr).startswith('lr_callback'):
                    optimizer=tf.keras.optimizers.Adam()
                    callbacks.append(get_lr_callback(batch_size=CFG.batch_size, mode='cos', epochs=CFG.epochs))
                else:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr))
                    
                model.compile(optimizer=optimizer,
                              loss=tf.keras.losses.KLDivergence(),
                              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

                if os.path.exists(f'weights/{bb}.{fb}.{aug}.{lr}.keras'):
                    print(f'Loading Model weights from: weights/{bb}.{fb}.{aug}.{lr}.keras')
                    #model = load_model(f'weights/{bb}.{fb}.{aug}.{lr}.keras')
                    model.load_weights(f'weights/{bb}.{fb}.{aug}.{lr}.keras')

                history = model.fit(
                    train_ds,
                    validation_data=valid_ds,
                    batch_size=CFG.batch_size,
                    epochs=CFG.epochs,
                    callbacks=callbacks,
                    shuffle=True,
                    steps_per_epoch=train_steps, 
                    validation_steps=valid_steps
                )

                model.load_weights(f'weights/{bb}.{fb}.{aug}.{lr}.keras')

                train_loss, train_accuracy, train_precision, train_recall = model.evaluate(train_ds, steps=train_steps)
                val_loss, val_accuracy, val_precision, val_recall = model.evaluate(valid_ds, steps=valid_steps)
                test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_ds, steps=test_steps)

                del model
                del train_ds
                del history
                gc.collect()

                results_list.append({
                    'BackBone': bb,
                    'FreezeBase': fb,
                    'Augment': aug,
                    'LearningRate': lr,
                    'Epochs': CFG.epochs,
                    'BatchSize' : CFG.batch_size,
                    'Patience': patience,
                    'TrainLoss': train_loss,
                    'TrainAccuracy': train_accuracy,
                    'TrainPrecision': train_precision,
                    'TrainRecall': train_recall,
                    'ValidLoss': val_loss,
                    'ValidAccuracy': val_accuracy,
                    'ValidPrecision': val_precision,
                    'ValidRecall': val_recall,
                    'TestLoss': test_loss,
                    'TestAccuracy': test_accuracy,
                    'TestPrecision': test_precision,
                    'TestRecall': test_recall,
                })

                if not os.path.exists("results"): os.makedirs("results")
                
                # Save results each iteration so we can check progress
                results_df = pd.DataFrame(results_list)
                results_df.to_csv('results/Results.csv', index=False)
                results_df.to_csv(f'results/{bb}.{fb}.{aug}.{lr}.csv', index=False)
                print(results_df)

results_df = pd.DataFrame(results_list)
results_df.to_csv('results/Results.csv', index=False)
print(results_df)
