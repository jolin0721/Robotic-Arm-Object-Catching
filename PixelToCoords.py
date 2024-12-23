import math

def pixel_to_coords(corner_pixel, target_pixel, corner_coords):
    # 输入：
    # corner_pixel：长度为4的一维数组，分别代表左上角点的像素x坐标、像素y坐标，右下角点的像素x坐标、像素y坐标
    # target_pixel：长度为2的一维数组，分别代表目标中心点像素x坐标、像素y坐标
    # corner_coords：长度为8的一维数组，分别代表左上角点的机械臂x坐标、机械臂y坐标，左下角点的机械臂x坐标、机械臂y坐标
    #                                        右上角点的机械臂x坐标、机械臂y坐标，右下角点的机械臂x坐标、机械臂y坐标
    # 输出：
    # target_coords：长度为2的一维数组，分别代表目标中心点机械臂x坐标、机械臂y坐标
    
    # 计算区域像素大小
    region_pixel = [corner_pixel[0] - corner_pixel[2], corner_pixel[3] - corner_pixel[1]]
    
    # 计算加权平均比例系数
    avg_weight_1 = [(target_pixel[1] - corner_pixel[1]) / region_pixel[1], (corner_pixel[3] - target_pixel[1]) / region_pixel[1]]
    avg_weight_2 = [(target_pixel[0] - corner_pixel[2]) / region_pixel[0], (corner_pixel[0] - target_pixel[0]) / region_pixel[0]]
    
    # 计算坐标中间量
    mid_coords_1 = [avg_weight_1[0]*corner_coords[6] + avg_weight_1[1]*corner_coords[2], avg_weight_1[0]*corner_coords[7] + avg_weight_1[1]*corner_coords[3]]
    mid_coords_2 = [avg_weight_1[0]*corner_coords[4] + avg_weight_1[1]*corner_coords[0], avg_weight_1[0]*corner_coords[5] + avg_weight_1[1]*corner_coords[1]]
    
    # 计算目标中心点机械臂坐标
    target_coords = [avg_weight_2[0]*mid_coords_2[0] + avg_weight_2[1]*mid_coords_1[0], avg_weight_2[0]*mid_coords_2[1] + avg_weight_2[1]*mid_coords_1[1]]
    
    # 将输出坐标改为两位小数
    target_coords = [round(coord, 2) for coord in target_coords]
    
    return target_coords
