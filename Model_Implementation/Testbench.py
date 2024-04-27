# Merhnoush
# Jeremy Lim
# jlim@wpi.edu
# Quick testbench framework, to help guide us in implementation.

# Basically, once this testbench is up and running, we can tweak/re-tweak DRLA using the training set to optimize performance.
# NOTE: NOT a good idea to tweak the classifier, only the AL method!!!

# INITIAL_SAMPLES = 130
BATCH_SIZE = 32
N_S =1 # one sample each time from unlabeled data
# EPOCHS = 5
LEARNING_RATE = 0.0001
# NUM_CLASSES = 2
# INPUT_SHAPE = (224, 224, 3)
#regarding the Jeremy's Comment. I will modify it 


import numpy as np
import copy
from MetricPlotter import MetricPlotter
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import h5py
import SplitSkinCancerMnist
from ActiveLearningMethods import EntropyStrategy, RandomSamplingStrategy , LeastConfidenceStrategy , DRLA


def convert_to_one_hot(labels, num_classes):
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    return one_hot_labels.squeeze()  

def resize_images(images):
    ###Resize images to the standard for ResNet50.
    return tf.image.resize(images, [224, 224])


#These metrics will be implemented
# def F1_Micro(y_actual, y_pred):
    
#    return F1_Micro

# def F1_Macro(y_actual, y_pred):
#     return F1_Macro

class ResNet50Classifier:
    def __init__(self, input_shape, num_class, lr=LEARNING_RATE):
        self.input_shape = input_shape
        self.num_class = num_class
        self.lr = lr
        self.model = self._build_model()

    def _build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Freezing all layers
        for l in base_model.layers:
            l.trainable = False

        avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)

        output = Dense(2, activation='softmax')(avg_pool)
        
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=SGD(learning_rate=self.lr, momentum=0.9, nesterov=True),loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    
    def fit(self, x, y, epochs, batch_size, validation_data):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    
    def predict(self, x):
        return self.model.predict(x)
    
    
    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

def load_data(path):
    with h5py.File(path + 'train250_train_x.h5', 'r') as file:
        x_train = np.array(file['x'][:])
    with h5py.File(path + 'train250_train_y.h5', 'r') as file:
        y_train = np.array(file['y'][:])
        
        
    with h5py.File(path + 'train250_val_x.h5', 'r') as file:
        x_val = np.array(file['x'][:])
    with h5py.File(path + 'train250_val_y.h5', 'r') as file:
        y_val = np.array(file['y'][:])
    return x_train, y_train , x_val , y_val


def get_skin_mnist_x_y(dataFrame):
    return dataFrame.path.values, dataFrame.label.values

def main():
    print("Starting the main function.\n")
    
    path = '/Users/mehrnoushalizade/Desktop/TA-solutions/ActiveLearningProject/PatchCamelyon/output/'

    skin_train_train_x, skin_train_train_y = get_skin_mnist_x_y(SplitSkinCancerMnist.scMnist_train)
    skin_train_val_x, skin_train_val_y = get_skin_mnist_x_y(SplitSkinCancerMnist.scMnist_val)
    skin_test_train_x, skin_test_train_y = get_skin_mnist_x_y(SplitSkinCancerMnist.scMnist_test)
    skin_test_val_x, skin_test_val_y = get_skin_mnist_x_y(SplitSkinCancerMnist.scMnist_testVal)

    x_train, y_train, x_val, y_val = load_data(path)
    x_train = resize_images(x_train)  
    y_train = convert_to_one_hot(y_train, num_classes=2)

    x_val = resize_images(x_val)
    y_val = convert_to_one_hot(y_val, num_classes=2)

    strategies = [ 
                  # EntropyStrategy(),
                #   RandomSamplingStrategy(), LeastConfidenceStrategy() ,
                    DRLA(250 , 2 , y_train)  # n_samples, k_classes, n_truth_labels
                    ]
    
    metric_plotter = MetricPlotter()

    for strategy in strategies:
        print(f"------------------- << {strategy} >> --------------------------")
        classifier =  ResNet50Classifier(input_shape=(224, 224, 3), num_class=2) 
        model = classifier

        num_samples = len(x_train)
        initial_samples = 200  

        
        labeled_mask = np.zeros(num_samples, dtype=bool)
        initial_indices = np.random.choice(num_samples, initial_samples, replace=False)
        labeled_mask[initial_indices] = True

        # Initial labeled data
        x_labeled = np.array(x_train)[initial_indices]
        y_labeled = np.array(y_train)[initial_indices]

        # Unlabeled data indices
        remaining_indices = np.where(labeled_mask==False)[0]

        while len(remaining_indices) > 0:
            print(f"The Strategy is: {strategy} \n")
            # Choose one new samples from unlabeled pool to label it
            predictions = model.predict(np.array(x_train))  # Predict on all data
            print(f"dimension labeled_mask is : {labeled_mask.shape}")
            old_mask = copy.deepcopy(labeled_mask)
            
            
            selected_indices = strategy.choose_n_samples(N_S, predictions, labeled_mask)
            labeled_mask[selected_indices] = True

            # Update label mask
            labeled_indices = np.where(labeled_mask==True)[0]
            x_labeled = np.array(x_train)[labeled_indices]
            y_labeled = np.array(y_train)[labeled_indices]

            # Train the classifier with all labeled data
            model.fit(x_labeled, y_labeled, epochs=1, validation_data=(x_val, y_val), batch_size=BATCH_SIZE)

            # Evaluate the classifier (new_perfomance)
            new_performance = model.evaluate(x_val, y_val)
            new_state = model.predict(np.array(x_train))
            print(f"New performance: Loss = {new_performance[0]}, Accuracy = {new_performance[1]}")
            
            
            metric_plotter.save_epoch_metrics(None, None, loss=new_performance[0], accuracy=new_performance[1])
            strategy.update_on_new_state(new_state, labeled_mask, predictions, old_mask)

            # Update the remaining indices
            remaining_indices = np.where(labeled_mask==False)[0]

        print(f"Active learning with {strategy} completed.")
        metric_plotter.display_all_plots()

if __name__ == "__main__":
    main()
