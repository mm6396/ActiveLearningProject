# Jeremy Lim
# jlim@wpi.edu
# Trying to determine how many hidden units are needed for our actor network to approximate Shannon Entropy
# Use this to establish a sane baseline on how many hidden units to use for our actor.

# Taking some hints from HW5 code to help implement this
import random

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


TORCH_DEVICE = "cpu"  # Don't get too fancy!

class MiniActorNetwork(nn.Module):
    # Implment a network similar to our actor, but will be adapted to approximating Entropy instead.

    def __init__(self, k_classes_input, n_hidden_units):
        super(MiniActorNetwork, self).__init__()

        # Expects softmaxed class output as input.
        self.linear1 = torch.nn.Linear(k_classes_input, n_hidden_units, bias=True)

        self.linear2 = torch.nn.Linear(n_hidden_units, n_hidden_units, bias=True)
        self.linear3 = torch.nn.Linear(n_hidden_units, n_hidden_units, bias=True)

        # For sigmoidal activation.
        self.linear4 = torch.nn.Linear(n_hidden_units, 1, bias=True)

        self.out_activation = torch.nn.Sigmoid()

        self.layer_activation = torch.nn.ReLU()

    def forward(self, x):

        # z_out = self.layer_activation(self.linear1(x))
        # z_out = self.layer_activation(self.linear2(z_out))
        # z_out = self.layer_activation(self.linear3(z_out))

        # z_out1 = torch.nn.LeakyReLU()(self.linear1(x))
        # z_out2 = torch.nn.LeakyReLU()(self.linear2(z_out1))
        # z_out3 = torch.nn.LeakyReLU()(self.linear3(z_out2))

        z_out1 = torch.nn.ReLU()(self.linear1(x))
        z_out2 = torch.nn.ReLU()(self.linear2(z_out1))
        z_out3 = torch.nn.ReLU()(self.linear3(z_out2))

        # z_out1 = self.linear1(x)
        # z_out2 = self.linear2(z_out1)
        # z_out3 = self.linear3(z_out2)

        # print(x)
        # print(z_out3)

        out_presigmoid = self.linear4(z_out3)

        # print(out_presigmoid)

        return self.out_activation(out_presigmoid)
        # return out_presigmoid

        # return self.out_activation(self.linear4(z_out2))


def entropy(in_arr):
    """
    Compute shannon entropy on a batch of softmax class "probabilities"
    :param in_arr: nxk matrix, n being the number of samples, k being the number of classes.
    :return: an array of length n, being the entropy computed for each sample.
    """
    # Entropy notes:
    # The minimum entropy is always zero, which corresponds to a 1-hot encoding with all
    # zeros except for the selected class being 1.
    # The maximum entropy is when the output is equally balanced between all classes.
    # So the max is

    # I'm using base 2 in this case.

    arr_log = np.log2(in_arr)
    # This may generate runtime warnings, but the following line will handle it and prevent -np.inf from propagating...
    arr_log[np.isinf(arr_log)] = 0  # handle -infinity.

    return -np.sum(np.log2(in_arr)*in_arr, axis=1)


def entropy_scaled(in_arr):
    """
    Wrapper for the above function that scales the max entropy to the range 0-1.0, so it's compatible
    with the actor model.
    :param in_arr: nxk matrix, n being the number of samples, k being the number of classes.
    :return: an array of length n, being the entropy computed for each sample, normalized to 0-1.0
    """
    max_entropy = -np.log2(1.0/in_arr.shape[1])

    return entropy(in_arr)/max_entropy


def softmax(in_arr):
    """
    Computes softmax.
    :param in_arr: nxk matrix, n being the number of samples, k being the number of classes.
    :return:
    """

    scale_arr = np.sum(np.exp(in_arr), axis=1, keepdims=True)
    return np.exp(in_arr) / scale_arr


def build_uniform_synthetic_softmax(n_samples, k_classes):
    """
    Sampling routine that enforces that every sample has roughly equal probability appearing.
    :param n_samples:
    :param k_classes:
    :return:
    """

    synthetic_softmaxes = np.zeros((n_samples, k_classes))

    for k in range(k_classes):
        if k <= 0:
            # Assign first 0-9 sampling.
            synthetic_softmaxes[:, k] = np.random.rand(n_samples)
        else:
            # Look at the previous rows, determine how much "probability" you have left.
            prop_remain = 1.0 - np.sum(synthetic_softmaxes[:,0:k], axis=1)
            synthetic_softmaxes[:, k] = prop_remain*np.random.rand(n_samples)

    assert np.max(np.sum(synthetic_softmaxes, axis=1)) <= 1.0, "Softmax broken!"

    return synthetic_softmaxes


def synthetic_softmax_reject_sampling(n_samples, k_classes):
    """
    Softmax outputs are surprisingly difficult to sample well from. Doing rejection sampling to get ACTUALLY uniform distribution.
    :param n_samples:
    :param k_classes:
    :return:
    """

    rand_batch_size = 2000

    rand_center_offset = -0.5
    rand_multiplier = 3.0

    close_enough_epsilon = 0.05  # higher values sample faster, but introduce bias/issues into dataset quality.

    samples_filled = 0
    synthetic_softmaxes = np.zeros((n_samples, k_classes))

    while samples_filled < n_samples:
        # get a batch of random candidates
        rand_batch = (np.random.rand(rand_batch_size, k_classes) + rand_center_offset) * rand_multiplier

        # how close it is to an actual softmax output. Remember this array
        softmax_dist1 = 1.0 - np.sum(rand_batch, axis=1)

        # compute offsets.
        rand_batch = rand_batch + np.expand_dims((softmax_dist1 / k_classes), axis=1)

        softmax_dist2 = 1.0 - np.sum(rand_batch, axis=1)

        good_indices = np.logical_and(
            np.logical_and(np.abs(softmax_dist1) <= close_enough_epsilon, np.all(rand_batch >= 0, axis=1)),
            np.all(rand_batch <= 1, axis=1))

        # Remove some samples that introduce fp noise...
        good_indices = np.logical_and(good_indices, np.abs(softmax_dist2) == 0.0)

        good_samples = rand_batch[good_indices]

        if n_samples - samples_filled < good_samples.shape[0]:
            remaining = n_samples - samples_filled
            synthetic_softmaxes[samples_filled:samples_filled+remaining, :] = good_samples[0:remaining,:]
        else:
            synthetic_softmaxes[samples_filled:samples_filled+good_samples.shape[0],:] = good_samples

        samples_filled += good_samples.shape[0]

        print("Total samples filled: " + str(samples_filled))

    # It's very close to correct, but FP noise is pushing some values above/below this ever so slightly.
    # Will keep for now, no easy way to rework this

    assert np.max(np.sum(synthetic_softmaxes, axis=1)) == 1.0 and np.min(np.sum(synthetic_softmaxes, axis=1)) == 1.0, "Softmax broken!"

    return synthetic_softmaxes


def test():
    num_classes = 2
    hidden_units = 128

    # For 2 classes, the best I got was:

    # Other settings, a bit less important.
    rand_sample_count = 500  # Should definitely be enough!
    # epoch_count = 600  # Should be enough!
    # lr = 1.0 for sgd
    lr = 0.000001
    batch_size = 64

    test_model = MiniActorNetwork(num_classes, hidden_units)

    print("Testing with " + str(hidden_units) + " hidden units and " + str(num_classes) + " classes.")

    # build the synthetic dataset
    # 1-hot encoded softmax.
    # NOTE: Sampling randomly is not getting excellent results. Instead, some sort of grid approach?
    # synthetic_x = (np.random.rand(rand_sample_count, num_classes) - 0.5) * 1.0  # Scaling factor to allow good chance of exploring the whole range.
    # synthetic_x = softmax(synthetic_x)

    # synthetic_x = build_uniform_synthetic_softmax(rand_sample_count, num_classes)

    # TODO: Need to understand this sampling as k increases better. Staying with this sampling approach for now.
    print("Creating synthetic dataset...")
    synthetic_x = synthetic_softmax_reject_sampling(rand_sample_count, num_classes)

    print("Synthetic dataset creation done.")

    # TODO: Need to understand this distribution better for k > 2...
    # plt.hist(synthetic_x[:,0])
    # plt.show()

    # Scaled entropy, 0-1.0, roughly.
    synthetic_y = entropy_scaled(synthetic_x)
    # synthetic_y = entropy(synthetic_x)

    # Convert to torch tensors
    synthetic_x = torch.tensor(synthetic_x, dtype=torch.float32)
    synthetic_y = torch.tensor(synthetic_y, dtype=torch.float32)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(test_model.parameters(), lr=lr)  # will do adam for now.
    # optimizer = optim.SGD(test_model.parameters(), lr=lr)

    last_loss = 1
    max_loss = 0.00001

    epoch_int = 0

    print("Train...")
    # for epoch in range(epoch_count):
    while last_loss > max_loss:
        # Shuffle
        idxs = [i for i in range(len(synthetic_x))]
        random.shuffle(idxs)

        train_x = synthetic_x[idxs]
        train_y = synthetic_y[idxs]

        # Batch is chosen as a multiple of the whole dataset size.
        for i in range(0, len(synthetic_x), batch_size):
            inputs = train_x[i:i + batch_size]
            targets = torch.unsqueeze(train_y[i:i + batch_size], 1)

            # inputs = synthetic_x[i:i + batch_size]
            # targets = torch.unsqueeze(synthetic_y[i:i + batch_size], 1)

            optimizer.zero_grad()
            outputs = test_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # print(loss)
        # Do it on the whole training set
        # Save memory! No gradients on this forward pass
        with torch.no_grad():
            train_loss = criterion(test_model(synthetic_x), synthetic_y).detach().item()

        # print("Epoch: " + str(epoch + 1) + "/" + str(epoch_count) + "; Last batch Train Loss: " + str(loss.item()))

        last_loss = train_loss
        print("Epoch: " + str(epoch_int) + "; Last whole Train Loss: " + str(train_loss))
        epoch_int += 1


    print("Compare on 10 random samples.")

    test_idxs = [i for i in range(len(synthetic_x))]
    # random.shuffle(test_idxs)

    x_samples = synthetic_x[test_idxs][:10,:]
    x_arr = x_samples.detach().numpy()

    print(x_arr)

    print("Scaled entropy output: ")
    ent = np.expand_dims(entropy_scaled(x_arr), axis=1)
    # ent = np.expand_dims(entropy(x_arr), axis=1)
    print(ent)

    print("Model output: ")

    with torch.no_grad():
        out_tensor = test_model(x_samples)
    print(out_tensor.detach().numpy())

    print("Done")


if __name__ == "__main__":
    test()
