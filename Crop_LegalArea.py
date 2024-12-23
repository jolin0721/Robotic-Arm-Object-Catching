import cv2
import numpy as np

# 预处理模板并保存
def preprocess_template(template_path, save_path):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Error: Could not load template at {template_path}")
    # 对模板进行二值化
    binary_template = cv2.adaptiveThreshold(template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    # 保存预处理后的模板
    np.save(save_path, binary_template)
    print(f"Preprocessed template saved at {save_path}")

# 用左上角和右下角的两个码来做区域裁剪
def detect_qr_corners(image_path, template1_features_path, template2_features_path):
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load the image")
        return None

    # 加载预处理后的模板特征
    template1 = np.load(template1_features_path)
    template2 = np.load(template2_features_path)

    # 图像二值化
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 模板匹配
    res1 = cv2.matchTemplate(binary, template1, cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(binary, template2, cv2.TM_CCOEFF_NORMED)

    # 获取匹配位置
    _, max_val1, _, max_loc1 = cv2.minMaxLoc(res1)
    _, max_val2, _, max_loc2 = cv2.minMaxLoc(res2)

    print("Template 1 match confidence:", max_val1)
    print("Template 2 match confidence:", max_val2)

    # 模板尺寸
    h1, w1 = template1.shape
    h2, w2 = template2.shape

    # 计算裁剪区域
    top_left1 = max_loc1
    bottom_right1 = (top_left1[0] + w1, top_left1[1] + h1)
    top_left2 = max_loc2
    bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)

    img_color = cv2.imread(image_path)
    cropped_region = img_color[top_left1[1]:bottom_right2[1], bottom_right2[0] - w2:top_left1[0] + w1]

    # Save the cropped region
    cv2.imwrite('cropped_region.jpg', cropped_region)
    
    print(f"Cropped region saved as cropped_region.jpg")

    # 返回裁剪结果
    return (top_left1[0] + w1, top_left1[1]), (bottom_right2[0] - w2, bottom_right2[1]), cropped_region


# # 用左上角和右下角的两个码来做区域裁剪
# def detect_qr_corners(image_path, template1_path, template2_path):
#     # Read images
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     template1 = cv2.imread(template1_path, cv2.IMREAD_GRAYSCALE)
#     template2 = cv2.imread(template2_path, cv2.IMREAD_GRAYSCALE)
    
#     if img is None or template1 is None or template2 is None:
#         print("Error: Could not load one or more images")
#         return None
    
#     # Binarize the image with adaptive thresholding for better QR code detection
#     # _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#     binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY, 11, 2)
    
#     # Template matching with normalized correlation coefficient
#     res1 = cv2.matchTemplate(binary, template1, cv2.TM_CCOEFF_NORMED)
#     res2 = cv2.matchTemplate(binary, template2, cv2.TM_CCOEFF_NORMED)
    
#     # Get the best match locations
#     min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
#     min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
    
#     print("Template 1 match confidence:", max_val1)
#     print("Template 2 match confidence:", max_val2)
    
#     # Get template dimensions after cropping
#     h1, w1 = template1.shape
#     h2, w2 = template2.shape
    
#     # Define corners with actual QR code dimensions
#     top_left1 = max_loc1
#     bottom_right1 = (top_left1[0] + w1, top_left1[1] + h1)
    
#     top_left2 = max_loc2
#     bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
    
#     # Calculate midpoints
#     midpoint1 = (int((top_left1[0] + bottom_right1[0])/2), int((top_left1[1] + bottom_right1[1])/2))
#     midpoint2 = (int((top_left2[0] + bottom_right2[0])/2), int((top_left2[1] + bottom_right2[1])/2))

#     # Create color image for visualization
#     # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     img_color = cv2.imread(image_path)

#     # Crop the region using numpy slicing = array[y1:y2, x1:x2]
#     cropped_region = img_color[top_left1[1]:bottom_right2[1], bottom_right2[0]-w2:top_left1[0]+w1]
#     # cropped_region_rotated = cv2.rotate(cropped_region, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     cropped_region_rotated = cropped_region
#     # Save the cropped region
#     cv2.imwrite('cropped_region.jpg', cropped_region_rotated)
    
#     print(f"Cropped region saved as cropped_region.jpg")

#     # Draw rectangles and circles
#     cv2.rectangle(img_color, top_left1, bottom_right1, (0, 255, 0), 2)  # Green for top-left
#     cv2.rectangle(img_color, top_left2, bottom_right2, (255, 0, 0), 2)  # Blue for bottom-right
#     cv2.rectangle(img_color, (top_left1[0]+w1,top_left1[1]), (bottom_right2[0]-w2,bottom_right2[1]), (0, 0, 255), 2)  # Red for full area
#     cv2.circle(img_color, midpoint1, 5, (0, 0, 255), -1)
#     cv2.circle(img_color, midpoint2, 5, (0, 0, 255), -1)
    
#     print("Top-left corner:", top_left1[0]+w1, ",",top_left1[1])
#     print("Bottom-right corner:", bottom_right2[0]-w2,",",bottom_right2[1])
#     print(f"Image dimensions: {img.shape[1]}x{img.shape[0]} pixels (Width x Height)")
#     print(f"Template1 dimensions: {template1.shape[1]}x{template1.shape[0]} pixels (Width x Height)")
#     print(f"Template2 dimensions: {template2.shape[1]}x{template2.shape[0]} pixels (Width x Height)")
    
#     cv2.imwrite('photo_for_check.jpg', img_color)
#     print("识别有效区域已保存为 photo_for_check.jpg")
#     # cv2.imshow("Result", img_color)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     top_left_corner = (top_left1[0]+w1, top_left1[1])
#     bottom_right_corner = (bottom_right2[0]-w2, bottom_right2[1])
    
#     return top_left_corner, bottom_right_corner, cropped_region_rotated