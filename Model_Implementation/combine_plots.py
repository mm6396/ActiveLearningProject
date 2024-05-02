# jeremy lim
# quick script to combine an plot MetricPlotter runs

import sys, os
import pickle

import numpy
from matplotlib import pyplot as plt

import MetricPlotter

def main():

    # mark_interval = 4
    #
    # plot_interval = 8

    # dataset_prefix = "Camelyon"
    #
    # rand_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\Random_Camelyon\\Random_Camelyon.pickle"
    # entropy_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\Entropy_Camelyon\\Entropy_Camelyon.pickle"
    # lc_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\LC_Camelyon\\LC_Camelyon.pickle"
    # drla_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Camelyon_DRLA\\Camelyon_DRLA.pickle"

    # dataset_prefix = "Skin_Mnist"
    #
    # mark_interval = 16
    #
    # plot_interval = 32
    #
    # rand_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\Random_Skin_MNIST\\Random_Skin_MNIST.pickle"
    # entropy_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\Entropy_Skin_MNIST\\Entropy_Skin_MNIST.pickle"
    # lc_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\LC_Skin_MNIST\\LC_Skin_MNIST.pickle"
    # drla_path = "C:\\Users\\jerem\\Documents\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Skin_MNIST_DRLA\\Skin_MNIST_DRLA.pickle"

    dataset_prefix = "Diabetic_Retinopathy"

    mark_interval = 16

    plot_interval = 16

    rand_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\Random_Diabetic_Retinopathy\\Random_Diabetic_Retinopathy.pickle"
    entropy_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\Entropy_Diabetic_Retinopathy\\Entropy_Diabetic_Retinopathy.pickle"
    lc_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Model_Implementation\\LC_Diabetic_Retinopathy\\LC_Diabetic_Retinopathy.pickle"
    drla_path = "C:\\Users\\jerem\\Documents\\WPI_Stuff\\CS_541_Group_Project\\ActiveLearningProject\\Db_R_DRLA\Db_R_DRLA.pickle"

    with open(rand_path, "rb") as f:
        rand_plotter = pickle.load(f)

    with open(entropy_path, "rb") as f:
        entropy_plotter = pickle.load(f)

    with open(lc_path, "rb") as f:
        lc_plotter = pickle.load(f)

    with open(drla_path, "rb") as f:
        DRLA_plotter = pickle.load(f)

    # combine plots.
    # https://www.geeksforgeeks.org/linestyles-in-matplotlib-python/
    # https://stackoverflow.com/questions/8409095/set-markers-for-individual-points-on-a-line
    # https://stackoverflow.com/questions/2040306/plot-with-fewer-markers-than-data-points-or-a-better-way-to-plot-cdfs-matplo

    print(entropy_plotter.get_metric_dict_copy()["val_accuracy"])
    print(entropy_plotter.get_metric_dict_copy()["val_macro_f1"])
    print(entropy_plotter.get_metric_dict_copy()["val_micro_f1"])
    # print(entropy_plotter.get_metric_dict_copy()["val_accuracy"])
    # print(lc_plotter.get_metric_dict_copy()["val_accuracy"])
    # print(DRLA_plotter.get_metric_dict_copy()["val_accuracy"])

    epoch_nums = entropy_plotter.get_epochs_list()
    epoch_nums = epoch_nums[0:-1:plot_interval]

    # Accuracy val_accuracy
    plt.title("Validation Accuracy")
    plt.plot(epoch_nums, rand_plotter.get_metric_dict_copy()["val_accuracy"][0:-1:plot_interval], label="Random", linestyle="dashdot", marker="o", markevery=mark_interval)
    plt.plot(epoch_nums, entropy_plotter.get_metric_dict_copy()["val_accuracy"][0:-1:plot_interval], label="Entropy", linestyle="dashdot", marker="^", markevery=mark_interval)
    plt.plot(epoch_nums, lc_plotter.get_metric_dict_copy()["val_accuracy"][0:-1:plot_interval], label="Least Confidence", linestyle="dashdot", marker="v", markevery=mark_interval)
    plt.plot(epoch_nums, DRLA_plotter.get_metric_dict_copy()["val_accuracy"][0:-1:plot_interval], label="DRLA", linestyle="dashdot", marker="s", markevery=mark_interval)
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(dataset_prefix + "_Combined_Accuracy.png")
    plt.show()

    # Macro F1
    plt.title("Validation Macro F1")
    plt.plot(epoch_nums, rand_plotter.get_metric_dict_copy()["val_macro_f1"][0:-1:plot_interval], label="Random", linestyle="dashdot", marker="o", markevery=mark_interval)
    plt.plot(epoch_nums, entropy_plotter.get_metric_dict_copy()["val_macro_f1"][0:-1:plot_interval], label="Entropy", linestyle="dashdot", marker="^", markevery=mark_interval)
    plt.plot(epoch_nums, lc_plotter.get_metric_dict_copy()["val_macro_f1"][0:-1:plot_interval], label="Least Confidence", linestyle="dashdot", marker="v", markevery=mark_interval)
    plt.plot(epoch_nums, DRLA_plotter.get_metric_dict_copy()["val_macro_f1"][0:-1:plot_interval], label="DRLA", linestyle="dashdot", marker="s", markevery=mark_interval)
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(dataset_prefix + "_Combined_Macro_F1.png")
    plt.show()

    # Micro F1
    plt.title("Validation Micro F1")
    plt.plot(epoch_nums, rand_plotter.get_metric_dict_copy()["val_micro_f1"][0:-1:plot_interval], label="Random", linestyle="dashdot", marker="o", markevery=mark_interval)
    plt.plot(epoch_nums, entropy_plotter.get_metric_dict_copy()["val_micro_f1"][0:-1:plot_interval], label="Entropy", linestyle="dashdot", marker="^", markevery=mark_interval)
    plt.plot(epoch_nums, lc_plotter.get_metric_dict_copy()["val_micro_f1"][0:-1:plot_interval], label="Least Confidence", linestyle="dashdot", marker="v", markevery=mark_interval)
    plt.plot(epoch_nums, DRLA_plotter.get_metric_dict_copy()["val_micro_f1"][0:-1:plot_interval], label="DRLA", linestyle="dashdot", marker="s", markevery=mark_interval)
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(dataset_prefix + "_Combined_Micro_F1.png")
    plt.show()



if __name__ == "__main__":
    main()