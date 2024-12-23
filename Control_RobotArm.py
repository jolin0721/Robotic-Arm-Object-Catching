from pymycobot.mycobot import MyCobot
from PixelToCoords import pixel_to_coords
import time

# 机械臂控制
def control_robot(top_left_pixel, bottom_right_pixel, target_pixel, bin_coord, bin_color, corner_coords):
    # 设置机械臂
    mc = MyCobot("COM10", 115200)  # 修改为你的端口号

    # 原点机械臂坐标
    origin_pos = [-70, -30, 300, -90, -45, -90]

    # 四个筒的坐标
    green = [268.9, 64.8, 200, -165.59, -4.23, -14.65]
    blue = [161.2, 115.5, 200, -175.22, -3.52, 12.65] 
    gray = [64.6, -182.6, 200, -176.85, -8.71, -91.81]
    red = [181.9, -208.3, 200, -176.24, -10.3, -76.54]

    # 物块可摆放区域的四个顶点的机械臂坐标
    left_upper = [168, -120]
    right_upper = [81, -95]
    left_lower = [215, -22]
    right_lower = [122, 18]
    corner_coords = left_upper + left_lower + right_upper + right_lower

    center_pos = [145, -60, 100, -180, 0, -45] # 区域中心点坐标，测试用

    # 摄像头固有参数
    full_size = [640, 480] # 摄像头分辨率
    block_size = [70, 70] # 物块顶面面积的像素大小

    # 此处像素坐标参数为测试图像中获得的一组常量
    # 在最后的项目实现中，需要结合test_cam以及yolo的功能实时获取
    # top_left_pixel = [450, 140] # 区域左上角像素坐标
    # bottom_right_pixel = [240, 350] # 区域右下角像素坐标
    target_pixel = [target_pixel[0], target_pixel[1]] # 目标中心点像素坐标

    # 考虑物块大小，对区域边界像素坐标进行修正
    fixed_top_left_pixel = [top_left_pixel[0] - block_size[0] / 2, top_left_pixel[1] + block_size[1] / 2]
    fixed_bottom_right_pixel = [bottom_right_pixel[0] + block_size[0] / 2, bottom_right_pixel[1] - block_size[1] / 2]
    corner_pixel = fixed_top_left_pixel + fixed_bottom_right_pixel

    print(target_pixel)
    # 计算目标的机械臂坐标
    target_coords = pixel_to_coords(corner_pixel, target_pixel, corner_coords)
    print(target_coords)
    target_pos = target_coords + [100, -180, 0, -45]
    print(target_pos)
    # target_pos = [127, -110, 100, -180, 0, -45]

    # 取到物块后，需要先将物块移动到一个合适的位置，再移动到筒的位置
    # mid_pos即为这个“中间位置”的坐标
    # mid_pos = [100, -70, 280, 180, 0, -45]
    # mid_pos = [148.7, -38.3, 279.0, -173.32, -5.86, 20.68] # 按上面的设定值，物块行进间会与筒壁产生碰撞，故暂时用原点位置代替
    mid_pos_red = [123.7, -148.3, 270.4, 178.47, -6.88, -13.93]
    mid_pos_green = [194.3, 21.2, 263.3, -174.08, -4.27, 23.64]

    if bin_color == "red" or bin_color == "gray":
        mid_pos = mid_pos_red
    elif bin_color == "green" or bin_color == "blue":
        mid_pos = mid_pos_green

    # 复位
    mc.send_coords(origin_pos, 40, 0)
    time.sleep(3)
    print(mc.get_coords())

    # 移动到指定物块的位置
    mc.send_coords(target_pos, 40, 0)
    time.sleep(3)
    current_coords = mc.get_coords()
    print(current_coords)

    # 判断是否成功到达了指定位置
    error = 0
    for i in range(3):
        error += abs(current_coords[i] - target_pos[i])

    if error < 10:
        print("Succeed in moving to the point")
    else:
        print("Failed in moving to the point")

    # 夹取物块
    mc.set_basic_output(2, 0)  
    mc.set_basic_output(5, 0)
    time.sleep(3)

    # 移动到中间位置
    mc.send_coords(mid_pos, 40, 1)
    time.sleep(3)

    current_coords = mc.get_coords()
    print(current_coords)

    # 判断是否成功到达了指定位置
    error = 0
    for i in range(3):
        error += abs(current_coords[i] - mid_pos[i])

    if error < 10:
        print("Succeed in moving to the point")
    else:
        print("Failed in moving to the point")

    mc.send_coords(bin_coord, 60, 1)
    time.sleep(3)

    # 放置物块
    mc.set_basic_output(2, 1)  
    mc.set_basic_output(5, 1)
    time.sleep(8.5) # 这个值似乎还有优化空间

    # 复位
    mc.send_coords(origin_pos, 60, 1)
    time.sleep(3)

# mc = MyCobot("COM10", 115200)
# origin_pos = [-70, -30, 300, -90, -45, -90]
# mc.send_coords(origin_pos, 60, 0)
# mc.set_basic_output(2, 1)  
# mc.set_basic_output(5, 1)
# mc.release_all_servos()

# import cv2
# import gradio as gr
# import tempfile


# # 打开摄像头
# camera_index = 0  # 默认摄像头索引为 0。如果有多个摄像头，可以尝试 1、2 等。
# cap = cv2.VideoCapture(camera_index)

# if not cap.isOpened():
#     print("无法打开摄像头，请检查设备连接")
#     exit()

# print("按 'q' 退出")


# while True:
#     # 读取帧
#     ret, frame = cap.read()
#     if not ret:
#         print("无法读取帧")



#     # cv2.imwrite('photo.jpg', frame) 
#     # print("照片已保存为 photo.jpg")

#     # 显示摄像头画面
#     cv2.imshow('Camera', frame)

#     # 按下 'q' 键退出
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # 释放资源
# cap.release()
# cv2.destroyAllWindows()