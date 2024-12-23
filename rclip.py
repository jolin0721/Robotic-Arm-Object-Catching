import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
from clip import clip

def find_best_match_center(image_path, text_prompt, device="cpu"):
    # 加载预训练的Faster R-CNN模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # 设置为评估模式

    # 加载图像
    image = Image.open(image_path)
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 将图像转换为模型所需的格式
    image_tensor = F.to_tensor(image).unsqueeze(0)  # 增加一个批次维度

    # 使用模型的RPN部分生成候选框
    with torch.no_grad():  # 确保不会计算梯度
        prediction = model(image_tensor)

    # 提取RPN生成的候选框
    proposal_boxes = prediction[0]['boxes'].cpu().numpy()

    # 过滤掉小于40x40像素的候选框
    filtered_boxes = []
    for box in proposal_boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if width * height >= 1600 and width * height <= 6400:  # 只保留大于40x40像素的框
            filtered_boxes.append(box)

    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 处理文本提示，只保留'put'之前的内容
    if "into" in text_prompt:
        truncated_text = text_prompt.split("into")[0].strip()  # 截断到'put'之前，并去除多余空格
    else:
        truncated_text = text_prompt  # 如果没有'put'，使用整个文本
    print(f"Using truncated text: {truncated_text}")  # 输出调试信息

    # 文本提示
    text = clip.tokenize([truncated_text]).to(device)

    # 裁剪并预处理每个检测到的物块
    preprocessed_images = []
    cropped_boxes = []
    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        crop_image = image.crop((x1, y1, x2, y2))
        preprocessed_images.append(preprocess(crop_image).unsqueeze(0))
        cropped_boxes.append((x1, y1, x2, y2))

    # 将所有预处理后的物块堆叠成一个批次
    image_input = torch.cat(preprocessed_images, dim=0).to(device)

    # 使用CLIP模型对图片和文本进行编码
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text)

    similarity = image_features @ text_features.T
    similarity_scores = similarity.squeeze(1).cpu().numpy()

    # 找出相似度分数最高的图片
    max_index = similarity_scores.argmax()
    max_score = similarity_scores[max_index]

    # 计算中心点坐标
    x1, y1, x2, y2 = cropped_boxes[max_index]
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    output_path = r"C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\output_img.jpg"

    # 可视化：在原图上绘制所有框，并高亮最佳匹配框
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色框

    # 高亮最佳框
    x1, y1, x2, y2 = map(int, cropped_boxes[max_index])
    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色框
    cv2.putText(cv_image, f"Best Match: {max_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 保存结果图片
    cv2.imwrite(output_path, cv_image)
    print(f"Output image saved to {output_path}")

    return center_x, center_y
