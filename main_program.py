import project_code as pc
import glob
import cv2
import matplotlib.pyplot as plt
if __name__ == "__main__":
    #calibrate camera:
#1. read chess images for calibration and apply camera calibration function
    images = glob.glob('camera_cal/calibration*.jpg')
    mtx, dist = pc.camera_calib(images)
#2. read an example frame and apply undistort function
    road_image = cv2.imread('test_images/test3.jpg')
    undist_img = pc.undistort_image(road_image, mtx, dist)
    #pc.show_undistorted_and_distorted_images(road_image, undist_img)
#3. Use color transforms, gradients, etc., to create a thresholded binary image
    color_binary, combined_binary = pc.create_thresholded_binary_image(undist_img, 140, 230, 160, 255)
    #pc.show_color_binary_and_combined_binary(color_binary, combined_binary)
#4. Apply interest mask and then apply a perspective transform to rectify binary image ("birds-eye view")
    after_mask = pc.create_interest_mask(combined_binary)
    transformed_perspective_image, M, Minv = pc.transform_perspective(after_mask)
    #pc.show_warped_image(after_mask, transformed_perspective_image)
#5. Plot histogram of lane in the transformed perspective image
    #pc.histogram(transformed_perspective_image)
#6. Detect lane pixels and fit to find the lane boundary
    sliding_window_img, ploty, left_fit, right_fit, left_fitx, right_fitx = pc.fit_polynomial(transformed_perspective_image)
    #pc.show_sliding_window_image(sliding_window_img)
#7. Use the previous polynomial to skip the sliding window
    result = pc.search_around_poly(transformed_perspective_image, left_fit, right_fit)
    plt.imshow(result)
    plt.show()
#8.
    print(pc.measure_curvature_pixels(ploty, left_fit, right_fit))
    print(pc.offset((left_fit, right_fit), road_image))
#9.
    plt.imshow(transformed_perspective_image)
    plt.show()
    original_lane_image = pc.original_lane_lines(transformed_perspective_image, undist_img, (left_fitx, right_fitx), Minv)
    pc.show_original_lane_image(original_lane_image)
    print('FINISHED')
