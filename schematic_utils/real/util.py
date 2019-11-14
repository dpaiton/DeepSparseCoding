import numpy as np

from data.dataset import Dataset
import utils.data_processing as dp

from .. import shared

selected_image_index = 5
completed_attack_index = 50

orig_bf_idx = 62
attack_bf_idx = 718


def retrieve_attack(npz_filepath,
                    image_index=selected_image_index,
                    completed_attack_index=completed_attack_index):

    images = np.squeeze(np.load(npz_filepath, allow_pickle=True)
                        ["data"].item()["adversarial_images"])

    attack = images[:completed_attack_index+1, image_index, :]

    return attack


def retrieve_basis_functions(analyzer,
                             orig_class_bf_idx=orig_bf_idx,
                             attack_class_bf_idx=attack_bf_idx):
    """
    Original and attack class basis function indices selected as
    most-changed indices between the original and the attack.
    """
    dummy = np.zeros(shape=(1, shared.mnist_dim))
    weights_var_name = 'lca/weights/w:0'
    weights = analyzer.evaluate_model(
        dummy, var_names=[weights_var_name])[weights_var_name]

    orig_class_bf = weights[:, orig_class_bf_idx]
    attack_class_bf = weights[:, attack_class_bf_idx]

    return orig_class_bf, attack_class_bf


def get_contour_inputs(analyzer,
                       target_vector, comparison_vector,
                       x_range, y_range, num_images):
    """
    modified version of dylan's code, streamlined for a single comparison
    """
    x_pts = np.linspace(x_range[0], x_range[1], int(np.sqrt(num_images)))
    y_pts = np.linspace(y_range[0], y_range[1], int(np.sqrt(num_images)))
    X_mesh, Y_mesh = np.meshgrid(x_pts, y_pts)
    proj_datapoints = np.stack(
        [X_mesh.reshape(num_images), Y_mesh.reshape(num_images)], axis=1)

    out_dict = {
        "proj_target_neuron": np.array([[1, 0]]),
        "proj_comparison_neuron": [],
        "proj_orth_vect": np.array([[0, 1]]),
        "orth_vect": [],
        "proj_datapoints": proj_datapoints,
        "x_pts": x_pts,
        "y_pts": y_pts}

    proj_matrix, orth_vect = dp.bf_projections(
        target_vector, comparison_vector)

    out_dict["orth_vect"] = orth_vect
    out_dict["proj_comparison_neuron"] = np.dot(
        proj_matrix, comparison_vector).T

    datapoints = np.stack(
        [np.dot(proj_matrix.T, proj_datapoints[data_id, :])
         for data_id in range(num_images)], axis=0)  # inject

    datapoints = dp.reshape_data(datapoints, flatten=False)[0]
    datapoints = {
        "test": Dataset(datapoints, lbls=None, ignore_lbls=None,
                        rand_state=analyzer.rand_state)}
    datapoints = analyzer.model.reshape_dataset(
        datapoints, analyzer.model_params)

    return out_dict, datapoints


def vec_to_mnist_image(vec):
    return np.reshape(vec, shared.mnist_dims)


def project_onto_plane(vec, vec1, vec2):
    """Unit vectors only"""
    return [np.dot(vec, vec1), np.dot(vec, vec2)]
