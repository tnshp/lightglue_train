import matplotlib.pyplot as plt
import numpy as np

def visualize_matches(image1, image2, keypoints1, keypoints2, matches):
    """
    Visualizes matching points between two images based on keypoints and match assignments.
    
    Parameters:
        image1 (torch.Tensor): The first image tensor of shape (C, H, W).
        image2 (torch.Tensor): The second image tensor of shape (C, H, W).
        keypoints1 (list of lists): List of [x, y] coordinates in the first image.
        keypoints2 (list of lists): List of [x, y] coordinates in the second image.
        matches (list or array): List of indices indicating matches, where matches[i] gives
                                 the index of the corresponding keypoint in keypoints2 for keypoints1[i].
                                 A value of -1 means no match for that keypoint.
    """
    # Convert images to numpy arrays for concatenation
    image1_np = image1.permute(1, 2, 0).numpy() if image1.shape[0] == 3 else image1.squeeze().numpy()
    image2_np = image2.permute(1, 2, 0).numpy() if image2.shape[0] == 3 else image2.squeeze().numpy()

    # Determine the new height for concatenation
    max_height = max(image1_np.shape[0], image2_np.shape[0])
    padded_image1 = np.pad(image1_np, ((0, max_height - image1_np.shape[0]), (0, 0), (0, 0)), mode='constant')
    padded_image2 = np.pad(image2_np, ((0, max_height - image2_np.shape[0]), (0, 0), (0, 0)), mode='constant')

    # Concatenate images horizontally
    combined_image = np.concatenate((padded_image1, padded_image2), axis=1)

    # Plot the concatenated image
    plt.figure(figsize=(10, 5))
    plt.imshow(combined_image, cmap='gray' if combined_image.ndim == 2 else None)
    # plt.axis('off')
    # plt.gca().invert_yaxis()
    # Draw lines with different colors for each match
    colors = plt.cm.get_cmap('hsv', len(matches))  # Use a color map for distinct colors

    for i, (y1, x1) in enumerate(keypoints1):
        match_idx = matches[i]
        if match_idx != -1:  # Check if there's a valid match
            y2, x2 = keypoints2[match_idx]

            # Offset x2 by the width of the first image for correct alignment
            x2_offset = x2 + image1_np.shape[1]

            # Draw the line connecting the matching keypoints
            plt.plot([x1, x2_offset], [y1, y2], color=colors(i), linewidth=0.8)
            # plt.plot(x1, y1, 'o', color=colors(i))   # Plot keypoint in image1
            # plt.plot(x2_offset, y2, 'o', color=colors(i))  # Plot keypoint in image2 with x-offset

    plt.show()

