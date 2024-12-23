import re

# 添加自然语言处理模块
# def extract_target_from_command(command):
    
#     # Extract the bin color (assumes color is just before "bin")
#     color_match = re.search(r"(\w+)\s+bin", command)
#     bin_color = color_match.group(1) if color_match else "red"

#     return bin_color.lower()

def extract_target_from_command(command):
    # 获取命令最后三个单词
    
    # 匹配颜色词（可以根据需要扩展颜色列表）
    color_pattern = r"\b(red|blue|green|gray)\b"
    color_match = re.search(color_pattern, command, re.IGNORECASE)
    
    # 提取颜色
    color = color_match.group(0) if color_match else "red"
    
    return color.lower()

# command = "There is a red bin, put apple into it"
# print(extract_target_from_command(command))