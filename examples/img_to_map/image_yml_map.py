import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from matplotlib.patches import Rectangle

from mobrob.envs.maps.builder import MapBuilder
from mobrob.envs.wrapper import get_env

# Tunable
# offset = [55,65] # this may be tuned
offset = [30, 45]
size = 402  # this is the size of the map in pixels (after crop)
field_threshold = 100000  # field is >= this size
tape_threshold = 85  # tape is <= this size
max_len = 70  # longest possible side length

pixel_size = 0.02  # maybe change


# Fixed
field_size = 2.42  # meters
## Object sizes
cylinder_radius = 0.0375  # meters
smallbox_height = 0.091
smallbox_length = 0.15


# Calculated
meter_per_pixel = field_size / size


# Visualize segmentation
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def segment(model, device, image, display=False):
    crop_size = [size, size]
    image = image[
        offset[0] : offset[0] + crop_size[0], offset[1] : offset[1] + crop_size[1]
    ]

    if display:
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    everything_results = model(
        image,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )[0]

    segmentation = everything_results.masks.data.cpu().numpy().astype(bool)
    boxes = everything_results.boxes.data.cpu().numpy()

    # convert to segment anything style format
    masks = []
    for i in range(segmentation.shape[0]):
        mask = {}
        mask["segmentation"] = segmentation[i]
        mask["bbox"] = [
            boxes[i][0],
            boxes[i][1],
            boxes[i][2] - boxes[i][0],
            boxes[i][3] - boxes[i][1],
        ]
        mask["area"] = mask["bbox"][2] * mask["bbox"][3]
        masks.append(mask)

    filtered_masks = []
    robot_color = np.zeros(3)  # black
    robot_color_dists = []
    areas = []

    # Filter masks by area and height/width
    for mask in masks:
        if (
            mask["area"] <= field_threshold
            and mask["area"] >= tape_threshold
            and mask["bbox"][2] <= max_len
            and mask["bbox"][3] <= max_len
        ):
            # compute color
            color = np.mean(image[mask["segmentation"], :], axis=0)
            color_dist = np.linalg.norm(color - robot_color)
            robot_color_dists.append(color_dist)
            areas.append(mask["area"])
            filtered_masks.append(mask)

    # find robot
    k = 3
    sorted_index = np.argsort(robot_color_dists)
    robot_index = sorted_index[:k]
    # largest area out of top k
    robot_index = [robot_index[np.array(areas)[robot_index].argmax()]]

    robot_masks = []
    for index in sorted(robot_index, reverse=True):
        robot_masks.append(filtered_masks[index])
        del filtered_masks[index]

    # remove additional masks which are inside a robot mask
    external_masks = []
    for mask in filtered_masks:
        inside = False
        x_1 = mask["bbox"][0]
        y_1 = mask["bbox"][1]
        x_2 = mask["bbox"][0] + mask["bbox"][2]
        y_2 = mask["bbox"][1] + mask["bbox"][3]
        for robot_mask in robot_masks:
            r_x_1 = robot_mask["bbox"][0]
            r_y_1 = robot_mask["bbox"][1]
            r_x_2 = robot_mask["bbox"][0] + robot_mask["bbox"][2]
            r_y_2 = robot_mask["bbox"][1] + robot_mask["bbox"][3]
            # check using x/y coordinates
            if (
                r_x_1 <= x_1 <= r_x_2
                and r_x_1 <= x_2 <= r_x_2
                and r_y_1 <= y_1 <= r_y_2
                and r_y_1 <= y_2 <= r_y_2
            ):
                inside = True
                break

        if not inside:
            external_masks.append(mask)
    filtered_masks = external_masks

    if display:
        # show filtered bboxs
        boxes = []
        for mask in filtered_masks:
            boxes.append(mask["bbox"])

        # show boxes w/ matplotlib
        plt.imshow(image)
        for box in boxes:
            # add rectangle to plot
            rect = Rectangle(box[:2], box[2], box[3])
            plt.gca().add_patch(rect)
        # display plot
        plt.show()

    if display:
        plt.imshow(image)
        show_anns(filtered_masks)
        plt.axis("off")
        plt.show()

    return filtered_masks, robot_masks


def generate_objects(filtered_masks, strategy="center"):
    objects = []
    for mask in filtered_masks:
        object_dict = generate_object(mask, strategy)
        objects.append(object_dict)
    return objects


def generate_object(mask, strategy="center"):
    if strategy == "center":
        box = np.array(mask["bbox"]) / size * field_size  # convert to meters
        # shift origin
        box[0] -= field_size / 2
        box[1] -= field_size / 2
        box[1] = -box[1]  # flip y axis
        width = float(box[2])
        height = float(box[3])
        center_x = float(box[0] + width / 2)
        center_y = float(box[1] + height / 2)
        object_dict = {}
        object_dict["center"] = [center_x, center_y]
        object_dict["size"] = [width / 2, height / 2]
        object_dict["type"] = "rectangle"
    else:
        raise Exception("Invalid strategy")
    return object_dict


def main(args):
    # load segmentation model
    model = FastSAM("./FastSAM-s.pt")
    model.to(args.device)

    # load image
    img_path = os.path.join(args.img_dir, args.img_name)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # segment image
    filtered_masks, robot_masks = segment(model, args.device, image, display=args.debug)

    # generate objects
    field_objects = generate_objects(filtered_masks, strategy=args.strategy)
    map_config = {}
    map_config["obstacles"] = field_objects
    map_config["map_size"] = [field_size, field_size]
    map_config["pixel_size"] = pixel_size

    yml_str = yaml.dump(map_config, default_style=None)

    # save file
    with open(args.yaml_name, "w") as f:
        f.write(yml_str)

    # build map from yml
    map_builder = MapBuilder.from_predefined_map(config_dict=map_config)
    map_builder.plot_map()

    # use map for env
    env = get_env("point", enable_gui=True, map_config=map_config)

    env.reset()
    env.toggle_render_mode()

    image = env.render()
    plt.imshow(image)
    plt.show()
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image to parse
    parser.add_argument("--img_name", type=str, default="37901.png")
    parser.add_argument("--img_dir", type=str, default="./example_imgs_2")
    parser.add_argument(
        "--yaml_name",
        type=str,
        default="map.yaml",
        help="name of yaml file to save map to",
    )

    # strategy for generating objects
    parser.add_argument("--strategy", type=str, default="center", choices=["center"])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--debug", action="store_true", default=False
    )  # display intermediate results

    args = parser.parse_args()
    main(args)
