from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np

from . import util
from .. import shared

color_vals = dict(
    zip(["lt_green", "md_green", "dk_green", "lt_blue",
         "md_blue", "dk_blue", "lt_red", "md_red", "dk_red"],
        ["#A9DFBF", "#196F3D", "#27AE60", "#AED6F1",
         "#3498DB", "#21618C", "#F5B7B1", "#E74C3C", "#943126"]))


def make_panel(analyzer,
               attack,
               orig_class_bf, attack_class_bf,
               attack_class_bf_index=util.attack_bf_idx,
               f=None, ax=None):

    ax_lims = [-0.1, 1.25]

    # Contour plotting parameters
    x_pts = [0.3, 1]  # Ranges for computing contour values
    y_pts = [0, 1]
    n_contour_images = 900
    n_contour_side = int(np.sqrt(n_contour_images))

    if ax is None:
        figsize = (12, 12)
        f, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)

    # get images and basis functions
    original_image_vec, advers_image_vec = attack[0], attack[-1]

    # compute the plane in image coordinates
    orth_orig_class_bf = shared.orthogonalize(orig_class_bf, attack_class_bf)
    plane = [attack_class_bf, orth_orig_class_bf]

    # project the images during the attack,
    #  including original and final, onto plane
    projected_attacks = [util.project_onto_plane(
        attack_vec / np.linalg.norm(attack_vec), *plane)
        for attack_vec in attack]
    orig_coords, advers_coords = projected_attacks[0], projected_attacks[-1]

    # project the bfs onto that plane
    orig_class_bf_coord, attack_class_bf_coord = [
            util.project_onto_plane(vec, *plane)
            for vec in [orig_class_bf, attack_class_bf]]

    # compute the values for the contours
    out_dict, datapoints = util.get_contour_inputs(
        analyzer, attack_class_bf, orig_class_bf,
        x_pts, y_pts, n_contour_images)
    proj_activations = analyzer.compute_activations(
        datapoints["test"].images)[:, attack_class_bf_index]

    # add the original and attack images
    [add_image(util.vec_to_mnist_image(vec), coords, ax=ax)
     for vec, coords in zip([original_image_vec, advers_image_vec],
                            [orig_coords, advers_coords])]

    # plot arrows for the basis functions
    [add_arrow(coords, ax)
     for coords in [orig_class_bf_coord, attack_class_bf_coord]]

    # plot the contours
    add_contours(out_dict, proj_activations, n_contour_side, ax=ax)

    # add the weights as images
    [add_image(util.vec_to_mnist_image(vec), np.array(coords) * 1.2, ax=ax)
     for vec, coords in zip([orig_class_bf, attack_class_bf],
                            [orig_class_bf_coord, attack_class_bf_coord])]

    # plot the attack as a line plus arrowhead
    add_attack(projected_attacks, ax=ax)

    ax.set_xlim(*ax_lims)
    ax.set_ylim(*ax_lims)
    ax.axis("off")

    return f, ax


def make_annotation_bbox(image, loc, offset_image_kwargs={"cmap": "Greys"}):
    imagebox = OffsetImage(image, zoom=1.5, **offset_image_kwargs)
    ab = AnnotationBbox(imagebox, loc)
    return ab


def add_image(image, coords, ax):
    ax.add_artist(make_annotation_bbox(image, coords))


def add_arrow(coords, ax):
    ax.arrow(0, 0, *coords, lw=12, head_width=0.05, color="k")


def add_contours(out_dict, proj_activations, n_contour_side, ax):
    reshaped_activations = np.reshape(proj_activations,
                                      (n_contour_side, n_contour_side))

    lws = [0] + [6] * 7  # zeroes out lowest contour, which looked bad
    ax.contour(out_dict["x_pts"], out_dict["y_pts"],
               reshaped_activations,
               zorder=0,
               linewidths=lws,
               cmap=shared.contour_cmap
               )


def add_attack(projected_attacks, ax, end_attack_index=38):

    ax.plot(*np.array(projected_attacks[:end_attack_index]).T,
            color=color_vals["md_red"], lw=6, zorder=0)

    ax.plot(*np.array(projected_attacks[:end_attack_index][12:-5:5]).T,
            color=[0, 0, 0, 0], lw=6, zorder=0,
            marker=".",
            markerfacecolor="k", markeredgecolor=color_vals["md_red"],
            markersize=24, markeredgewidth=3)

    ax.scatter(*np.array(projected_attacks[end_attack_index-1]).T,
               marker=(3, 0, 50),  # rotated triangle to point correctly
               facecolor="k", linewidths=3,
               edgecolor=color_vals["md_red"],
               s=512)
