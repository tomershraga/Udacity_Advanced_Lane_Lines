import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_interest_mask(combined_binary_image):
    mask = np.zeros_like(combined_binary_image)
    region_of_interest = np.array([[0, combined_binary_image.shape[0] - 1], [combined_binary_image.shape[1] / 2, int(0.5 * combined_binary_image.shape[0])],
                                   [combined_binary_image.shape[1] - 1, combined_binary_image.shape[0] - 1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest], 1)
    ret_image = cv2.bitwise_and(combined_binary_image, mask)
    return ret_image

def transform_perspective(threshold_image):
    # Define source points
    src = np.array([[200, 720], [1150, 720], [750, 480], [550, 480]], np.float32)
    # Define destination points
    dst = np.array([[300, 720], [900, 720], [900, 0], [300, 0]], np.float32)
    # Get the transformation matrix by performing perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    # Get the inverse transformation matrix
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Get the image size
    image_size = (threshold_image.shape[1], threshold_image.shape[0])
    # Warp the image
    warped_image = cv2.warpPerspective(threshold_image, M, image_size)
    return warped_image, M, Minv

def show_warped_image(combined_binary_image, warped_image):
    # Plot the warped image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    ax1.imshow(combined_binary_image, cmap='gray')
    ax1.set_title('Combined binary image', fontsize=30)
    ax2.imshow(warped_image, cmap='gray')
    ax2.set_title('Warped Image', fontsize=30)
    plt.show()
    #cv2.imwrite('output_images/warped_image.png', warped_image)

def histogram(img):
    bottom_half = img[img.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    plt.plot(histogram)
    plt.show()
