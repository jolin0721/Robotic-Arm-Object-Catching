import cv2
import gradio as gr
import time
import numpy as np
import speech_recognition as sr
import keyboard

from sklearn.cluster import DBSCAN

from Control_RobotArm import control_robot
from Crop_LegalArea import detect_qr_corners
from ExtractTarget_LLM import extract_target_from_command
from Crop_LegalArea import preprocess_template
from rclip import find_best_match_center
from sort2 import find_best_match_centers


# 摄像头不清晰，所以做上采样，但是因为上采样的qr code也要上采样，所以就不用了
def upscale_image_from_path(image_path, output_path, scale_factor=2):
    """
    从路径读取图像并进行上采样，保存结果为 JPG 文件。
    
    参数:
        image_path (str): 输入图像的路径。
        output_path (str): 上采样后保存图像的路径。
        scale_factor (int): 上采样比例，默认值为 2。
    """
    # 从路径读取图像
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"无法加载图像，请检查路径：{image_path}")
    
    # 获取图像尺寸并计算新的尺寸
    height, width = image.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    
    # 使用OpenCV进行上采样
    upscaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    
    # 将上采样后的图像保存为 JPG
    cv2.imwrite(output_path, upscaled_image)
    
    print(f"上采样后的图像已保存至 {output_path}")

def list_available_cameras(max_tested=10):
    available_cameras = []
    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"摄像头 {index} 可用")
            available_cameras.append(index)
            cap.release()  # 释放摄像头资源
        else:
            print(f"摄像头 {index} 不可用")
    return available_cameras

def speech_to_text():
    recognizer = sr.Recognizer()
    
    try:
        print("Press 'space' to start listening...")  # 提示用户按键
        keyboard.wait('space')  # 等待按下空格键
        
        with sr.Microphone(device_index=1) as source:
            print("Microphone initialized")
            print("Say something...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)  # 开始捕获音频
            print("Audio captured")
        
        print("Recognizing...")
        command = recognizer.recognize_google(audio)  # 使用Google API进行识别
        print(f"You said: {command}")
        return command
    
    except sr.WaitTimeoutError:
        print("No speech detected within the timeout period")
    except sr.UnknownValueError:
        print("Speech was detected, but it couldn't be recognized")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def parse_commands(command):
    """
    将主命令解析成多个子命令。
    """
    return [cmd.strip() for cmd in command.split(";") if cmd.strip()]

# 主函数
def main():

    # # 尝试检测最多10个摄像头（可以根据实际情况增加max_tested的值）
    # available_cameras = list_available_cameras()
    # print(f"可用的摄像头索引: {available_cameras}")

    camera_index = 0  # 默认摄像头索引为 0。如果有多个摄像头，可以尝试 1、2 等。
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7) # 降低曝光
    
    while(1):
        # 这部分
        flag_sort = 0
        print("语音输入请输1，文字输入请输2，退出请输入3，分类请输入4")
        input_of_command = input("> ")
        # print(input_of_command)
        if input_of_command == "3":
            print("closed")
            break
        elif input_of_command == "1":
            time.sleep(1)
            command = speech_to_text()
            print(command)
        elif input_of_command == "2":
            command = input("请输入文字> ")
        elif input_of_command == "4":
            print("classification start!")
            flag_sort = 1
        
        ret, frame = cap.read()
        cv2.imwrite('original_photo.jpg', frame) 
        print("照片已保存为 original_photo.jpg")

        # image_qr = r'C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\original_photo.jpg'
        image_qr = r'C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\original_photo.jpg'
        # image_up = r'C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\project\pymycobot\image_up.jpg'
        # image_down = r'C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\project\pymycobot\image_down.jpg'
        template1_features_path = r'C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\template1_features.npy'
        template2_features_path = r'C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\template2_features.npy'
        # # preprocess_template(image_up, template1_features_path)
        # # preprocess_template(image_down, template2_features_path)   
        top_left_corner, bottom_right_corner, cropped_image = detect_qr_corners(image_qr,template1_features_path,template2_features_path)

        cropped_region_path = r"C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\cropped_region.jpg"  # 这里是裁剪区域的图片路径
        
        if flag_sort:
            results = find_best_match_centers(cropped_region_path)
            print("Results:", results)
            bottom_left_corner = (bottom_right_corner[0],top_left_corner[1])

            # 控制机械臂
            left_upper = [168, -120]
            right_upper = [81, -95]
            left_lower = [215, -22]
            right_lower = [122, 18]
            corner_coords = left_upper + left_lower + right_upper + right_lower

            if results['food']:
                for coord in results['food']:
                    center_x = coord[0]
                    center_y = coord[1]
                    print(center_x,center_y,"food")
                    target_pixel = (center_x+bottom_left_corner[0],center_y+bottom_left_corner[1])
                    bin_color = "green"
                    bin_coordinates = [161.2, 115.5, 200, -175.22, -3.52, 12.65] #green
                    #调用函数
                    target_pixel = (center_x+bottom_left_corner[0],center_y+bottom_left_corner[1])
                    print("target pixel(x, y):", target_pixel)
                    control_robot(top_left_corner, bottom_right_corner, target_pixel, bin_coordinates, bin_color, corner_coords)

            else:
                print("No food found.")

            if results['animal']:
                for coord in results['animal']:
                    center_x = coord[0]
                    center_y = coord[1]
                    print(center_x,center_y,"animal")
                    target_pixel = (center_x+bottom_left_corner[0],center_y+bottom_left_corner[1])
                    bin_color = "red"
                    bin_coordinates = [181.9, -208.3, 200, -176.24, -10.3, -76.54] #red
                    #调用函数
                    target_pixel = (center_x+bottom_left_corner[0],center_y+bottom_left_corner[1])
                    print("target pixel(x, y):", target_pixel)
                    control_robot(top_left_corner, bottom_right_corner, target_pixel, bin_coordinates, bin_color, corner_coords)

            else:
                print("No animal found.")

            if results['non_living']:
                for coord in results['non_living']:
                    center_x = coord[0]
                    center_y = coord[1]
                    print(center_x,center_y,"others")
                    target_pixel = (center_x+bottom_left_corner[0],center_y+bottom_left_corner[1])
                    bin_color = "gray"
                    bin_coordinates = [64.6, -182.6, 200, -176.85, -8.71, -91.81] #gray
                    #调用函数
                    target_pixel = (center_x+bottom_left_corner[0],center_y+bottom_left_corner[1])
                    print("target pixel(x, y):", target_pixel)
                    control_robot(top_left_corner, bottom_right_corner, target_pixel, bin_coordinates, bin_color, corner_coords)

            else:
                print("No other non_living thing found.")
        else:
            if ";" in command:
                commands = parse_commands(command)
                for sub_command in commands:
                    center_x, center_y = find_best_match_center(cropped_region_path, sub_command)

                    cv2.circle(cropped_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                    # 显示图片
                    # cv2.imshow('Best Match Image', cropped_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    bottom_left_corner = (bottom_right_corner[0],top_left_corner[1])

                    bin_color = extract_target_from_command(sub_command)
                    print(f"Bin color: {bin_color}")

                    # 定义颜色到坐标的映射a
                    color_to_coordinates = {
                        'blue': [268.9, 64.8, 200, -165.59, -4.23, -14.65],
                        'green': [161.2, 115.5, 200, -175.22, -3.52, 12.65],
                        'gray': [64.6, -182.6, 200, -176.85, -8.71, -91.81],
                        'red': [181.9, -208.3, 200, -176.24, -10.3, -76.54]
                    }

                    # 根据bin_color获取对应的坐标
                    bin_coordinates = color_to_coordinates.get(bin_color, None)

                    target_pixel = (center_x+bottom_left_corner[0],center_y+bottom_left_corner[1])
                    print("target pixel(x, y):", target_pixel)
                    # 控制机械臂
                    left_upper = [168, -120]
                    right_upper = [81, -95]
                    left_lower = [215, -22]
                    right_lower = [122, 18]
                    corner_coords = left_upper + left_lower + right_upper + right_lower
                    # print(top_left_corner)
                    # print(bottom_right_corner)
                    control_robot(top_left_corner, bottom_right_corner, target_pixel, bin_coordinates, bin_color, corner_coords)
        
            else:
                center_x, center_y = find_best_match_center(cropped_region_path, command)

                cv2.circle(cropped_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                # 显示图片
                # cv2.imshow('Best Match Image', cropped_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                bottom_left_corner = (bottom_right_corner[0],top_left_corner[1])

                bin_color = extract_target_from_command(command)
                print(f"Bin color: {bin_color}")

                # 定义颜色到坐标的映射a
                color_to_coordinates = {
                    'blue': [268.9, 64.8, 200, -165.59, -4.23, -14.65],
                    'green': [161.2, 115.5, 200, -175.22, -3.52, 12.65],
                    'gray': [64.6, -182.6, 200, -176.85, -8.71, -91.81],
                    'red': [181.9, -208.3, 200, -176.24, -10.3, -76.54]
                }

                # 根据bin_color获取对应的坐标
                bin_coordinates = color_to_coordinates.get(bin_color, None)

                target_pixel = (center_x+bottom_left_corner[0],center_y+bottom_left_corner[1])
                print("target pixel(x, y):", target_pixel)
                # 控制机械臂
                left_upper = [168, -120]
                right_upper = [81, -95]
                left_lower = [215, -22]
                right_lower = [122, 18]
                corner_coords = left_upper + left_lower + right_upper + right_lower
                # print(top_left_corner)
                # print(bottom_right_corner)
                control_robot(top_left_corner, bottom_right_corner, target_pixel, bin_coordinates, bin_color, corner_coords)

# 示例运行

# for index, name in enumerate(sr.Microphone.list_microphone_names()):
#     print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))
main()
