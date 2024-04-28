# Jeremy Lim
# jlim@wpi.edu
# A small class to help with plotting metrics for various projects/assignments.
# This is my own code, so I'll use it across different assignments.

import sys, os
import copy
import datetime # For epoch timestamps

# Below: just for testing.
import math
import random

from matplotlib import pyplot as plt


class MetricPlotter:

    # If you want to clear/restart, delete an existing instance, and create a new one!
    def __init__(self):
        self.epoch_counter = 1

        self.metrics = {}

    # Extend this function to calculate metrics specific to use case!
    def calculate_metrics(self, y_true, y_pred):
        pass

    def get_metric_names(self):
        return list(self.metrics.keys())

    # Return a copy of the raw metrics dictionary.
    # It's a copy, so the original cannot get messed up!
    def get_metric_dict_copy(self):
        return copy.deepcopy(self.metrics)

    # Get a list of integers, denoting plot epochs.
    # make it easy to plot stuff if you call get_metric_dict_copy
    def get_epochs_list(self):
        return list(range(1, self.epoch_counter+1))

    # How many epochs we have metrics recorded for.
    def get_num_epochs(self):
        return self.epoch_counter

    # Return a dictionary containing the last element from each metric.
    # Good for building tables.
    def get_last_all_metrics(self):
        ret_dict = {}
        for key in self.metrics.keys():
            ret_dict[key] = self.metrics[key][-1]

        return ret_dict

    # Calculate and save some metrics. Also advances the epoch. Subclass to provide a useful behavior for this function!
    # DO NOT OVERWRITE IN SUBCLASS! Do that with calculate_metrics instead!
    def save_epoch_metrics(self, y_true, y_pred, **kwargs):
        if kwargs is not None:
            for key in kwargs.keys():
                self._handle_metric(key, kwargs[key])

        self.calculate_metrics(y_true, y_pred)

        self.epoch_counter += 1

    # Handle adding to a metric associated with a particular name.
    def _handle_metric(self, name, value):
        # assumes tuple is in the format (string_name, value)
        if not (name in self.metrics):
            assert self.epoch_counter == 1, "Adding a new metric that was not recorded before! Check your code in case you forget to record a metric sometimes!"
            self.metrics[name] = []

        self.metrics[name].append(value)

    # Display plots of metrics.
    def display_all_plots(self):
        self.display_plots(list(self.metrics.keys()))

    # Display plots of metrics.
    # Pass a list of strings of the plots you want to see. This does it sequentially.
    def display_plots(self, display_list):
        epochs_set = list(range(1, self.epoch_counter))
        for metric_name in display_list:

            plt.plot(epochs_set, self.metrics[metric_name])
            plt.title(metric_name)
            plt.xlabel("Epochs")

            plt.show()

    # Variation on display_plots; does the provided list of metrics simultaneously on the same plot
    # Useful for comparisons.
    # Also provides a legend.
    # This only produces one plot!
    def display_plot_simultaneous(self, display_list, plot_title):

        epochs_set = list(range(1, self.epoch_counter))
        for metric_name in display_list:
            # Colors will loop automatically
            plt.plot(epochs_set, self.metrics[metric_name], label=metric_name)

        plt.title(plot_title)
        plt.xlabel("Epochs")
        plt.legend()  # Add a legend. loc="upper right"
        plt.show()

    # Save plots to image files. Like display_plots
    def save_plots(self, display_list, save_dir=None):
        if save_dir is None:
            save_dir = os.getcwd()

        epochs_set = list(range(1, self.epoch_counter))
        for metric_name in display_list:
            plt.plot(epochs_set, self.metrics[metric_name])
            plt.title(metric_name)
            plt.xlabel("Epochs")

            # add in epoch time real quick!
            plt.savefig(os.path.join(save_dir, metric_name + "_" + datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".png"))
            plt.close()

    # Save plots to image files. Like display_plot_simultaneous.
    def save_plot_simultaneous(self, display_list, plot_title, save_dir=None):
        if save_dir is None:
            save_dir = os.getcwd()

        epochs_set = list(range(1, self.epoch_counter))
        for metric_name in display_list:
            # Colors will loop automatically
            plt.plot(epochs_set, self.metrics[metric_name], label=metric_name)

        plt.title(plot_title)
        plt.xlabel("Epochs")
        plt.legend()  # Add a legend
        # loc="upper right"

        plt.savefig(os.path.join(save_dir, plot_title + "_" + datetime.datetime.utcnow().strftime("%s") + ".png"))
        plt.close()


if __name__ == "__main__":
    test_num = 100

    # test the metric plotter
    sine_trend = [math.sin(x*math.pi*2.0/(float(test_num))) for x in range(test_num)]
    rand_trend = [random.random() for x in range(test_num)]

    test_plotter = MetricPlotter()

    for idx in range(test_num):
        # Doesn't usually work with None, but this is a basic test...
        test_plotter.save_epoch_metrics(None, None, sine=sine_trend[idx], random=rand_trend[idx])

    test_plotter.display_all_plots()

    test_plotter.display_plots(["sine"])

    test_plotter.display_plot_simultaneous(["sine", "random"], "Simultaneous")

    test_plotter.save_plots(["sine", "random"])

    test_plotter.save_plot_simultaneous(["sine", "random"], "Simultaneous")

    print("Done")
