import project_code as pc
import camera_calibration
import undistort
import thresholded_binary
import perspective_transform
import find_lane
import curvature_and_offset
import original_lane
import glob
import cv2
import matplotlib.pyplot as plt
if __name__ == "__main__":
# 1. read chess images for calibration and apply camera calibration function
    images = glob.glob('camera_cal/calibration*.jpg')
    mtx, dist = camera_calibration.camera_calib(images)
# 2. read an example frame and apply undistort function
    road_image = cv2.imread('test_images/test5.jpg')
    undist_img = undistort.undistort_image(road_image, mtx, dist)
    #undistort.show_undistorted_and_distorted_images(road_image, undist_img)
# 3. Use color transforms, gradients, etc., to create a thresholded binary image
    color_binary, combined_binary = thresholded_binary.create_thresholded_binary_image(undist_img, 140, 230, 160, 255)
    # thresholded_binary.show_color_binary_and_combined_binary(color_binary, combined_binary)
# 4. Apply interest mask and then apply a perspective transform to rectify binary image ("birds-eye view")
    after_mask = perspective_transform.create_interest_mask(combined_binary)
    transformed_perspective_image, M, Minv = perspective_transform.transform_perspective(after_mask)
    # perspective_transform.show_warped_image(after_mask, transformed_perspective_image)
# 5. Plot histogram of lane in the transformed perspective image
    # perspective_transform.histogram(transformed_perspective_image)
# 6. Detect lane pixels and fit to find the lane boundary
    sliding_window_img, ploty, left_fit, right_fit, left_fitx, right_fitx = find_lane.fit_polynomial(transformed_perspective_image)
    # find_lane.show_sliding_window_image(sliding_window_img)
# 7. Use the previous polynomial to skip the sliding window
    result = find_lane.search_around_poly(transformed_perspective_image, left_fit, right_fit)
    # plt.imshow(result)
    # plt.show()
# 8. Calculates the curvature of polynomial functions in meters
    left_curverad, right_curverad = curvature_and_offset.measure_curvature_pixels(ploty, left_fit, right_fit)
    print(left_curverad)
    print(right_curverad)
# 9. Calculate position of the car from the centre
    offset = curvature_and_offset.offset((left_fit, right_fit), road_image)
    print(offset)
    plt.imshow(cv2.cvtColor(transformed_perspective_image, cv2.COLOR_BGR2RGB))
    plt.show()
# 10. Combine original image with area of lane
    original_lane_image = original_lane.original_lane_lines(transformed_perspective_image, undist_img, (left_fitx, right_fitx), Minv,
                                                            left_curverad, right_curverad, offset)
    original_lane.show_original_lane_image(original_lane_image)
    print('FINISHED')
