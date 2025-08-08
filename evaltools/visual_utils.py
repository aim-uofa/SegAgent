import os

import cv2
import numpy as np
import torch


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    overlay = image.copy()
    binary_mask = mask
    color_mask = np.zeros_like(image)
    color_mask[binary_mask > 0] = color
    for c in range(0, 3):
        overlay[:, :, c] = np.where(
            binary_mask > 0,
            overlay[:, :, c] * (1 - alpha) + color_mask[:, :, c] * alpha,
            overlay[:, :, c],
        )
    return overlay


def overlay_points(image, points, color=(255, 0, 0)):
    if points is None:
        return image
    overlay = image.copy()
    cv2.circle(overlay, (int(points[1]), int(points[0])), 5, color, -1)
    return overlay


def overlay_boxes(image, box, color=(255, 0, 0), thickness=2):
    # box: [x1, y1, x2, y2]
    overlay = image.copy()
    overlay = cv2.rectangle(
        overlay,
        (int(box[0]), int(box[1])),
        (
            int(box[2]),
            int(box[3]),
        ),
        color,
        thickness,
    )
    return overlay


def visualize_mask_and_point(
    image, masks, points, path, is_positive, text=None, box=None
):
    # check image type
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
        print(image.min())
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
        if masks.ndim == 3:
            masks = masks[0]
    if points is not None:
        if points[0] < 1:
            points = [points[i] * image.shape[i] for i in range(2)]
        print(points)
    # overlay mask on image
    overlay = overlay_mask(image, masks)
    if is_positive:
        overlay = overlay_points(overlay, points)
    else:
        overlay = overlay_points(overlay, points, color=(0, 0, 255))

    if box is not None:
        overlay = overlay_boxes(overlay, box)

    if text is not None:
        # Add text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Smaller font scale
        font_color = (255, 255, 255)
        font_thickness = 1  # Thinner font thickness
        max_width = overlay.shape[1] - 20  # Maximum width for text

        # Split text into lines if necessary
        words = text.split(" ")
        lines = []
        current_line = words[0]
        for word in words[1:]:
            if (
                cv2.getTextSize(
                    current_line + " " + word, font, font_scale, font_thickness
                )[0][0]
                <= max_width
            ):
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        # Calculate text height
        text_height = sum(
            [
                cv2.getTextSize(line, font, font_scale, font_thickness)[0][1]
                for line in lines
            ]
        ) + 10 * len(lines)

        # Create a new image with extra space for the text
        new_image = np.zeros(
            (overlay.shape[0] + text_height + 20, overlay.shape[1], 3), dtype=np.uint8
        )
        new_image[text_height + 20 :, :] = overlay

        # Put the text on the new image
        y = 10
        for line in lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            text_x = (overlay.shape[1] - text_size[0]) // 2
            cv2.putText(
                new_image,
                line,
                (text_x, y + text_size[1]),
                font,
                font_scale,
                font_color,
                font_thickness,
            )
            y += text_size[1] + 10

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, new_image)
    return new_image


def visualize_mask_and_pointlist(
    image, masks, point_list, path, is_positive_list, text=None, box=None
):
    # check image type
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
        print(image.min())
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
        if masks.ndim == 3:
            masks = masks[0]
    # if points is not None:
    #     if points[0] < 1:
    #         points = [points[i] * image.shape[i] for i in range(2)]
    #     print(points)
    # overlay mask on image
    overlay = overlay_mask(image, masks)
    for points, is_positive in zip(point_list, is_positive_list):
        if is_positive:
            overlay = overlay_points(overlay, points)
        else:
            overlay = overlay_points(overlay, points, color=(0, 0, 255))

    if box is not None:
        overlay = overlay_boxes(overlay, box)

    if text is not None:
        # Add text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Smaller font scale
        font_color = (255, 255, 255)
        font_thickness = 1  # Thinner font thickness
        max_width = overlay.shape[1] - 20  # Maximum width for text

        # Split text into lines if necessary
        words = text.split(" ")
        lines = []
        current_line = words[0]
        for word in words[1:]:
            if (
                cv2.getTextSize(
                    current_line + " " + word, font, font_scale, font_thickness
                )[0][0]
                <= max_width
            ):
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        # Calculate text height
        text_height = sum(
            [
                cv2.getTextSize(line, font, font_scale, font_thickness)[0][1]
                for line in lines
            ]
        ) + 10 * len(lines)

        # Create a new image with extra space for the text
        new_image = np.zeros(
            (overlay.shape[0] + text_height + 20, overlay.shape[1], 3), dtype=np.uint8
        )
        new_image[text_height + 20 :, :] = overlay

        # Put the text on the new image
        y = 10
        for line in lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            text_x = (overlay.shape[1] - text_size[0]) // 2
            cv2.putText(
                new_image,
                line,
                (text_x, y + text_size[1]),
                font,
                font_scale,
                font_color,
                font_thickness,
            )
            y += text_size[1] + 10

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, new_image)
    return new_image
