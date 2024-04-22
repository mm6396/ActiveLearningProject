# Merhnoush
# Jeremy Lim
# jlim@wpi.edu
# Quick testbench framework, to help guide us in implementation.

# Basically, once this testbench is up and running, we can tweak/re-tweak DRLA using the training set to optimize performance.
# NOTE: NOT a good idea to tweak the classifier, only the AL method!!!



import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import h5py

from ActiveLearningMethods import EntropyStrategy

def convert_to_one_hot(labels, num_classes):
    
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    return one_hot_labels.squeeze()  

def resize_images(images):
    ###Resize images to the standard for ResNet50.
    return tf.image.resize(images, [224, 224])

class ResNet50Classifier:
    def __init__(self, input_shape, num_class, lr=0.001):
        self.input_shape = input_shape
        self.num_class = num_class
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = GlobalAveragePooling2D()(base_model.output)
        predictions = Dense(self.num_class, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def load_data(path):
    with h5py.File(path + 'train250_train_x.h5', 'r') as file:
        x_train = file['x'][:]
    with h5py.File(path + 'train250_train_y.h5', 'r') as file:
        y_train = file['y'][:]
        
        
    with h5py.File(path + 'train250_val_x.h5', 'r') as file:
        x_val = file['x'][:]
    with h5py.File(path + 'train250_val_y.h5', 'r') as file:
        y_val = file['y'][:]
    return x_train, y_train , x_val , y_val
def main():
    print("Starting the main function.")
    path = '/Users/mehrnoushalizade/Desktop/TA-solutions/ActiveLearningProject/PatchCamelyon/output/'

    x_train, y_train , x_val , y_val = load_data(path)
    x_train = resize_images(x_train.astype('float32'))
    y_train = convert_to_one_hot(y_train, num_classes=2)
    
    x_val = resize_images(x_val.astype('float32'))
    y_val = convert_to_one_hot(y_val, num_classes=2)
    
    
    strategies = [EntropyStrategy(),
                #    RandomStrategy(),
                #    LeastConfidenceStrategy(),
                #    DRLA()
                   ]
    
    for strategy in strategies:                           # Loop over strategies 
        print(f" ------------------- << {strategy} >> --------------------------")
        classifier = ResNet50Classifier(input_shape=(224, 224, 3), num_class=2)
        model = classifier.model

        num_samples = len(y_train)
        print("num samples are : {num_samples}")
        initial_samples = 130
        batch_size = 10

        initial_indices = np.random.choice(num_samples, initial_samples, replace=False)
        labeled_mask = np.zeros(num_samples, dtype=bool)
        labeled_mask[initial_indices] = True

    
        x_selected = tf.gather(x_train, initial_indices)
        y_selected = tf.gather(y_train, initial_indices)
        # print(f"x_selected shape: {x_selected.shape}, y_selected shape: {y_selected.shape}")

        model.fit(x_selected, y_selected, epochs=5 , batch_size=32)
        current_performance = model.evaluate(x_val, y_val)
        print(f"Initial performance for {strategy}: {current_performance}")

        remaining_indices = np.where(labeled_mask == False)[0]

        while len(remaining_indices) > 0:
            predictions = model.predict(tf.gather(x_train, remaining_indices))  # Predict those datasets that have not been labeled yet (unlabeled)
            entropy_strategy = EntropyStrategy()
            selected_indices = entropy_strategy.choose_n_samples(batch_size, predictions, labeled_mask[remaining_indices]) # select the indices that have highest entropy (most informative samples)
        
            labeled_mask[remaining_indices[selected_indices]] = True  # update the labeled_mask after marking the selected samples as labeled

            train_indices = np.where(labeled_mask==True)[0] # All data that those label is True 
            x_train_truelabeled = tf.gather(x_train, train_indices)
            y_train_truelabeled = tf.gather(y_train, train_indices)
       

            model.fit(x_train_truelabeled, y_train_truelabeled, epochs=1, batch_size=10)
            new_performance = model.evaluate(x_val, y_val)
            print(f"----------------------------------*3 >> Updated performance is for {strategy} : ------------------*3>> {new_performance}")

            remaining_indices = np.where(labeled_mask == False)[0]

    print("Active learning with entropy method completed.")

if __name__ == "__main__":
    main()
