import torch
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import numpy as np
from clip import clip
from torchvision.ops import nms

def find_best_match_centers(image_path, device="cpu",  nms_threshold=0.3, coord_threshold=6):
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

    # 提取RPN生成的候选框和得分
    proposal_boxes = prediction[0]['boxes'].cpu().numpy()
    proposal_scores = prediction[0]['scores'].cpu().numpy()

    # 过滤掉小于40x40像素的候选框
    filtered_boxes = []
    filtered_scores = []
    for i, box in enumerate(proposal_boxes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        if width * height >= 1600 and width * height <= 10000:  # 只保留中等大小的框
            filtered_boxes.append(box)
            filtered_scores.append(proposal_scores[i])

    # 转换为Tensor格式，方便执行NMS
    filtered_boxes_tensor = torch.tensor(filtered_boxes, dtype=torch.float32)
    filtered_scores_tensor = torch.tensor(filtered_scores, dtype=torch.float32)

    # 检查是否是二维张量，形状应为 (N, 4)
    if len(filtered_boxes_tensor.shape) == 1:
        # 将1D张量重塑为二维张量 (N, 4)
        filtered_boxes_tensor = filtered_boxes_tensor.view(-1, 4)
    # 使用NMS去除重叠的框
    keep_idx = nms(filtered_boxes_tensor, filtered_scores_tensor, nms_threshold)

    # 根据NMS选择保留的框
    final_boxes = filtered_boxes_tensor[keep_idx].cpu().numpy()

    # 进一步过滤：删除相邻位置坐标差小于6的框
    def filter_close_boxes(boxes, threshold=6):
        filtered = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            if not filtered:
                filtered.append(box)
            else:
                # 计算已有框的中心点与当前框中心点的距离
                keep = True
                for prev_box in filtered:
                    prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
                    prev_center_x, prev_center_y = (prev_x1 + prev_x2) / 2, (prev_y1 + prev_y2) / 2
                    distance = np.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)
                    if distance < threshold:
                        keep = False
                        break
                if keep:
                    filtered.append(box)
        return filtered

    # 过滤掉相邻坐标差小于6的框
    final_boxes_filtered = filter_close_boxes(final_boxes, threshold=coord_threshold)

    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # 文本提示列表
    prompts = ["a photo of food", "a photo of animal", "a photo of shoe", "a photo of flower", "a photo of gun","a photo of clock","a photo of camera"]

    # 存储最终结果（每个类别的中心点坐标）
    results = {"food": [], "animal": [], "non_living": []}

    # 对于每个目标框，计算与所有提示的相似度并选择最匹配的类别
    for box in final_boxes_filtered:
        x1, y1, x2, y2 = box
        crop_image = image.crop((x1, y1, x2, y2))
        preprocessed_image = preprocess(crop_image).unsqueeze(0).to(device)

        # 计算与所有文本提示的相似度
        best_prompt = None
        highest_score = -1  # 初始最小相似度

        for prompt in prompts:
            # 文本提示
            text = clip.tokenize([prompt]).to(device)

            # 使用CLIP模型对图像和文本进行编码
            with torch.no_grad():
                image_features = clip_model.encode_image(preprocessed_image)
                text_features = clip_model.encode_text(text)

            # 计算相似度
            similarity = image_features @ text_features.T
            similarity_score = similarity.item()  # 取出单一的相似度值

            # 更新最匹配的提示
            if similarity_score > highest_score:
                highest_score = similarity_score
                best_prompt = prompt

        # 对比相似度，选择最匹配的类别
        if best_prompt in ["a photo of food", "a photo of flower"]:
            results["food"].append(((x1 + x2) / 2, (y1 + y2) / 2))
        elif best_prompt in ["a photo of animal", "a photo of cat", "a photo of dog"]:
            results["animal"].append(((x1 + x2) / 2, (y1 + y2) / 2))
        else:
            results["non_living"].append(((x1 + x2) / 2, (y1 + y2) / 2))

    # 可视化：只显示符合阈值的匹配框的中心点
    for category, coords in results.items():
        for (center_x, center_y) in coords:            
            # 绘制红点
            cv2.circle(cv_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # 红点
            cv2.putText(cv_image, f"{category}", (int(center_x), int(center_y)- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    output_path = r"C:\Users\Jolin\Documents\THU_life\year3_autumn\recognition\out_4_img.jpg"
    # 保存结果图片
    cv2.imwrite(output_path, cv_image)
    print(f"Output image saved to {output_path}")

    # # 显示图片
    # cv2.imshow("Result Image", cv_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 返回每个类别的中心点坐标
    return results


