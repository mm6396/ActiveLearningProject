jlim@wpi.edu

The scripts in this folder assume that you've downloaded all of the files for PatchCamelyon from here:(https://github.com/basveeling/pcam?tab=readme-ov-file) into a local folder called "raw"

Run build_test_train to build 2 datasets: a large test set and a large training set. This will also build a set of balanced 313/313 datasets to match with the original paper.

(This is 250 train + 20% validation, so 63 validation in a set of 313 images)

For this, we use a 20% fraction for validation in datasets.