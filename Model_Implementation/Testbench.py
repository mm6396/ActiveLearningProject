# Merhnoush
# Jeremy Lim
# jlim@wpi.edu
# Quick testbench framework, to help guide us in implementation.
import sys, os

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


CLASS_FRACTION = 0.1

import pickle

import numpy as np
import copy
from MetricPlotter import MetricPlotter
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import h5py

import PIL
from PIL import Image

from ActiveLearningMethods import EntropyStrategy, RandomSamplingStrategy , LeastConfidenceStrategy, DRLA

import sklearn
from sklearn import metrics
from sklearn import preprocessing

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
        self.base_model, self.model = self._build_models()

        self.precompute_batch_size = 32  # Just to manage this prediction step, so it doesn't consume too much RAM.
        self.model_batch_size = 32  # Manage most of the other sets, don't try all images simultaneously!

    def _build_models(self):

        # Separating the model parts to run prediction in batched parts!

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Freezing all layers
        for l in base_model.layers:
            l.trainable = False

        avg_pool = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)

        construct_base_model = Model(inputs=base_model.input, outputs=avg_pool)

        trainable_input = tf.keras.layers.Input((avg_pool.shape[-1]))
        trainable_output = Dense(self.num_class, activation='softmax')(trainable_input)

        model = Model(inputs=trainable_input, outputs=trainable_output)
        model.compile(optimizer=SGD(learning_rate=self.lr, momentum=0.9, nesterov=True),loss='categorical_crossentropy', metrics=['accuracy'])

        return construct_base_model, model

    def precompute_input(self, x):
        # Use the frozen model trunk to precompute features for all x
        # Better than re-running the model every time...
        return self.base_model.predict(x, batch_size=self.precompute_batch_size)

    def fit(self, x, y, epochs, batch_size, validation_data):
        # NOTE: only fit precomputed x!
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    
    def predict(self, x):
        # NOTE: only fit precomputed x!
        return self.model.predict(x, batch_size=self.model_batch_size)
    
    
    def evaluate(self, x, y):
        # NOTE: only fit precomputed x!
        return self.model.evaluate(x, y, batch_size=self.model_batch_size)

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


def test_on_dataset(x_train, y_train_numerical, x_val, y_val_numerical, run_name, num_classes):


    x_train = resize_images(x_train)  
    y_train = convert_to_one_hot(y_train_numerical, num_classes=num_classes)

    x_val = resize_images(x_val)
    y_val = convert_to_one_hot(y_val_numerical, num_classes=num_classes)

    # JL - moving model outside of loop.
    classifier = ResNet50Classifier(input_shape=(224, 224, 3), num_class=num_classes)
    model = classifier

    # This step might take a little bit, but it only runs once!
    print("Precomputing feature maps for frozen resnet layers...")
    x_train = model.precompute_input(x_train)
    x_val = model.precompute_input(x_val)
    print("Done Precomputing feature maps!")

    strategies = [ 
                  # EntropyStrategy(),
                #   RandomSamplingStrategy(), LeastConfidenceStrategy() ,
                    DRLA(x_train.shape[0] , num_classes , y_train)  # n_samples, k_classes, n_truth_labels
                    ]
    
    metric_plotter = MetricPlotter()
    # val_metric_plotter = MetricPlotter()

    for strategy in strategies:
        print(f"------------------- << {strategy} >> --------------------------")

        num_samples = len(x_train)

        # 10% of each class, for binary case.
        negative_class_idx = np.where(np.squeeze(y_train_numerical) == 0)[0]
        negative_class_count = negative_class_idx.shape[0]
        positive_class_idx = np.where(np.squeeze(y_train_numerical) == 1)[0]
        positive_class_count = positive_class_idx.shape[0]

        negative_class_idx = np.random.choice(negative_class_idx, int(negative_class_count * CLASS_FRACTION), replace=False)
        positive_class_idx = np.random.choice(positive_class_idx, int(positive_class_count * CLASS_FRACTION), replace=False)

        initial_indices = np.concatenate([negative_class_idx, positive_class_idx], axis=0)

        labeled_mask = np.zeros(num_samples, dtype=bool)

        labeled_mask[initial_indices] = True

        # Initial labeled data
        # x_labeled = np.array(x_train)[initial_indices]
        # y_labeled = np.array(y_train)[initial_indices]

        # Unlabeled data indices
        remaining_indices = np.where(labeled_mask==False)[0]

        while len(remaining_indices) > 0:
            print("Remaining samples: " + str(len(remaining_indices)))
            # print(f"The Strategy is: {strategy} \n")
            # Choose one new samples from unlabeled pool to label it
            predictions = model.predict(np.array(x_train))  # Predict on all data
            # print(f"dimension labeled_mask is : {labeled_mask.shape}")
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

            # Metrics
            val_predictions = model.predict(x_val)
            micro_f1_val = sklearn.metrics.f1_score(np.argmax(y_val, axis=1), np.argmax(val_predictions, axis=1), average="micro")
            macro_f1_val = sklearn.metrics.f1_score(np.argmax(y_val, axis=1), np.argmax(val_predictions, axis=1), average="macro")

            micro_f1_train = sklearn.metrics.f1_score(np.argmax(y_train, axis=1), np.argmax(new_state, axis=1), average="micro")
            macro_f1_train = sklearn.metrics.f1_score(np.argmax(y_train, axis=1), np.argmax(new_state, axis=1), average="macro")

            train_performance = model.evaluate(x_train, y_train)

            # metric_plotter.save_epoch_metrics(None, None, loss=new_performance[0], accuracy=new_performance[1], )
            metric_plotter.save_epoch_metrics(None, None,
                                              train_loss=train_performance[0],
                                              train_accuracy=train_performance[1],
                                              train_micro_f1=micro_f1_train,
                                              train_macro_f1=macro_f1_train,
                                              val_loss=new_performance[0],
                                              val_accuracy=new_performance[1],
                                              val_micro_f1=micro_f1_val,
                                              val_macro_f1=macro_f1_val)

            strategy.update_on_new_state(new_state, labeled_mask, predictions, old_mask)

            # Update the remaining indices
            remaining_indices = np.where(labeled_mask==False)[0]

        # print(f"Active learning with {strategy} completed.")
        metric_plotter.display_all_plots()

        os.makedirs(run_name, exist_ok=True)
        # Save all plots separately
        metric_plotter.save_plots(metric_plotter.get_metric_names(), save_dir=run_name)

        # train/val loss for judging fit/overfit issues
        metric_plotter.display_plot_simultaneous(["train_loss", "val_loss"], "Train vs Val loss")

        with open(run_name + ".pickle", 'ab') as f:
            pickle.dump(metric_plotter, f)

        print("Run : " + run_name + " done.")


def load_image_paths(pathslist):
    arr_output = []
    for p in pathslist:
        img = np.asarray(Image.open(p))

        img = np.array(resize_images(img))

        # Resizing when I load, keep memory use down
        arr_output.append(img)

    return np.array(arr_output)


# def labels_to_int(labels_arr):
#     # Make numerical categories to

def main():
    # print("Test patch camelyon")
    #
    # histo_path = '/home/jeremy/Documents/WPI_Spring_24/CS_541/Group_Project/repository/ActiveLearningProject/PatchCamelyon/output/'
    # x_train, y_train_numerical, x_val, y_val_numerical = load_data(histo_path)
    #
    # test_on_dataset(x_train, y_train_numerical, x_val, y_val_numerical, run_name="Camelyon_DRLA", num_classes=2)

    print("Test Skin mnist")

    # Fixing path issues
    # Change to the Model_Implementation directory, wherever it is on your system
    old_path = os.getcwd()
    os.chdir("Model_Implementation")
    import SplitSkinCancerMnist


    skin_train_train_x, skin_train_train_y = get_skin_mnist_x_y(SplitSkinCancerMnist.scMnist_train)
    skin_train_val_x, skin_train_val_y = get_skin_mnist_x_y(SplitSkinCancerMnist.scMnist_val)
    # skin_test_train_x, skin_test_train_y = get_skin_mnist_x_y(SplitSkinCancerMnist.scMnist_test)
    # skin_test_val_x, skin_test_val_y = get_skin_mnist_x_y(SplitSkinCancerMnist.scMnist_testVal)

    # Load imagery
    skin_train_train_x = load_image_paths(skin_train_train_x)
    skin_train_val_x = load_image_paths(skin_train_val_x)

    # move back, to keep from messing other code up
    os.chdir(old_path)

    test_on_dataset(skin_train_train_x, skin_train_train_y, skin_train_val_x, skin_train_val_y, run_name="Skin_MNIST_DRLA", num_classes=7)

    # print("Test diabetic retinopathy")
    #
    # # Add parent directory to path, so we can import Diabetic_Retinopathy properly.
    # sys.path.append("/home/jeremy/Documents/WPI_Spring_24/CS_541/Group_Project/repository/ActiveLearningProject")
    # import Diabetic_Retinopathy
    #
    # # flip the mapping!
    # mapping_dict = {}
    # for key in Diabetic_Retinopathy.diagnosis_dict:
    #     mapping_dict[Diabetic_Retinopathy.diagnosis_dict[key]] = key
    #
    # train1_lbls = [mapping_dict[x] for x in Diabetic_Retinopathy.train1_labels]
    # test1_lbls = [mapping_dict[x] for x in Diabetic_Retinopathy.test1_labels]
    #
    # test_on_dataset(Diabetic_Retinopathy.train1_images,
    #                 train1_lbls,
    #                 Diabetic_Retinopathy.test1_images,
    #                 test1_lbls, run_name="Db_R_DRLA", num_classes=5)

    print("Done")


if __name__ == "__main__":
    main()
