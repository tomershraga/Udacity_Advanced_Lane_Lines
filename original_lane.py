import numpy as np
import cv2
import matplotlib.pyplot as plt

def original_lane_lines(warp_img, undistorted_line_image, x_line_values, MatrInv, left_curverad, right_curverad, offset):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Generate x and y values for plotting
    image_y_values = np.linspace(0, warp_img.shape[0] - 1, warp_img.shape[0])
    left_x_values = x_line_values[0]
    right_x_values = x_line_values[1]
    # Recast the x and y points into usable format for cv2.fillPoly()
    left_points = np.array([np.transpose(np.vstack([left_x_values, image_y_values]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_x_values, image_y_values])))])
    points = np.hstack((left_points, right_points))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([points]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warped_image = cv2.warpPerspective(color_warp, MatrInv, (warp_img.shape[1], warp_img.shape[0]))
    # Combine the result with the original image
    original_lane_image = cv2.addWeighted(undistorted_line_image, 1, new_warped_image, 0.3, 0)
    cv2.putText(original_lane_image, "radius of curvature: (L): " + str(round(left_curverad,3)) + " m (R): " + str(round(right_curverad,3)) + " m",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(original_lane_image, "offset from center: " + str(round(offset, 3)) + " m",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return original_lane_image

def show_original_lane_image(original_lane_image):
    plt.imshow(cv2.cvtColor(original_lane_image, cv2.COLOR_BGR2RGB))
    plt.show()