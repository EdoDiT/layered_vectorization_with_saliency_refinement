import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from PIL import Image
import os
import glob
from torchvision import transforms
import torch
import pydiffvg
from collections import Counter
from scipy.spatial.distance import cdist
from typing import Union,List
from tqdm import tqdm
import time
import random
from scipy.spatial import cKDTree

def svg_to_img(width: int, height: int, shapes, shape_groups, device):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        width, height, shapes, shape_groups
        )
    _render = pydiffvg.RenderFunction.apply
    img = _render(width,  # width
                height,  # height
                2,  # num_samples_x
                2,  # num_samples_y
                0,  # seed
                None,
                *scene_args)
    img = img.to(device)
    return img

def rgba_to_rgb(img: torch.Tensor, device, para_bg=None):
    if para_bg is None:
        para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
    img = img.permute(2, 0, 1)
    return img

def sam_img_seq(device, simp_img_seq: list, masks_save_path: str = "-1", sam_conf: dict = None) -> list:
    def mask_preprocessing(masks):
        prepro_masks = []
        for mask in masks:
            filled_mask = filling_mask_holes(mask)
            num_labels, labels = cv2.connectedComponents(filled_mask)
            for i in range(1, num_labels):
                single_region = np.zeros_like(mask)
                single_region[labels == i] = 255
                prepro_masks.append(single_region)
        return prepro_masks
    
    def sorted_masks(masks):
        area_list=[]
        for mask in masks:
            area = cv2.countNonZero(mask)
            area_list.append(area)
        sorted_indices = sorted(range(len(area_list)), key=lambda k: area_list[k])
        sorted_indices.reverse()
        sorted_masks = []
        for i in sorted_indices:
            sorted_masks.append(masks[i])
        return sorted_masks
    
    all_masks = []
    with tqdm(total=len(simp_img_seq), desc="Processing image", unit="value") as pbar:
        for i,simp_img in enumerate(simp_img_seq):
            if masks_save_path != "-1":
                masks_save_path_i = f"{masks_save_path}/{i+1}"
                os.makedirs(masks_save_path_i,exist_ok=True)
                masks = sam_image(device, simp_img, masks_save_path_i, sam_conf)
            else:
                masks = sam_image(device, simp_img, "-1", sam_conf)
            masks = mask_preprocessing(masks)
            all_masks.append(sorted_masks(masks))
            pbar.update(1)
    all_masks.reverse()
    all_masks = [mask for sublist in all_masks for mask in sublist]
    return all_masks

def sam_image(device, image: Union[str, np.ndarray], masks_save_path: str = "-1", sam_conf: dict = None) -> list:
    sam_checkpoint = sam_conf["sam_checkpoint"]
    model_type = sam_conf["model_type"]
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=sam_conf["points_per_side"],
        pred_iou_thresh=sam_conf["pred_iou_thresh"],
        stability_score_thresh=sam_conf["stability_score_thresh"],
        crop_n_layers=sam_conf["crop_n_layers"],
        crop_n_points_downscale_factor=sam_conf["crop_n_points_downscale_factor"],
        min_mask_region_area=sam_conf["min_mask_region_area"],  # Requires open-cv to run post-processing
        box_nms_thresh=sam_conf["box_nms_thresh"],
    )
    masks2 = mask_generator_2.generate(image)
    masks = [np.full(image.shape[:2], 255, dtype=np.uint8)]
    for i,mask in enumerate(masks2):
        image = np.where(mask['segmentation'], 255, 0).astype(np.uint8)
        masks.append(image)
        if masks_save_path != "-1":
            image = Image.fromarray(image)
            image.save(f'{masks_save_path}/{i}.png')
    return masks

def filling_mask_holes(image):
    if isinstance(image,str):
        image = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    flood_fill_image = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood_fill_image, mask, (0, 0), 255)
    flood_fill_inv = cv2.bitwise_not(flood_fill_image)
    filled_image = cv2.bitwise_or(image, flood_fill_inv)
    return filled_image

def layer_segmented_masks(layered_masks: list, unlayered_masks: list) -> list:
    while len(unlayered_masks)>0:
        found = False
        unlayered_mask = unlayered_masks.pop(0)
        for layer_i,layer in enumerate(layered_masks):
            if len(layer) > 0:
                for mask_i,mask in enumerate(layer):
                    area0 = cv2.countNonZero(mask)
                    area1 = cv2.countNonZero(unlayered_mask)
                    intersection_area = np.sum(mask.astype(np.float32)+unlayered_mask.astype(np.float32) == 510)
                    total_area = cv2.countNonZero(mask+unlayered_mask)
                    if intersection_area/total_area>0.70:  # 0.65
                        found = True
                        break 
                    if intersection_area/area0>=0.5 and area1>area0:
                        found = True
                        break 
                    if intersection_area/area1>=0.5 and area1<area0:
                        if layer_i == len(layered_masks)-1:
                            new_layer = [unlayered_mask]
                            layered_masks.append(new_layer)
                        break
                    if mask_i == len(layer)-1:
                        layered_masks[layer_i].append(unlayered_mask)
                        found = True
                        break

                if found:
                    break
    return layered_masks

def get_struct_masks_by_area(layerd_struct_masks: list, N: int):
    def get_top_n_indices(matrix,n):
        elements = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elements.append((matrix[i][j], i, j))
        sorted_elements = sorted(elements, key=lambda x: (-x[0], x[1], x[2]))
        indices = [(i, j) for val, i, j in sorted_elements[:n]]
        return indices
    
    if sum([len(x) for x in layerd_struct_masks])<= N:
        return layerd_struct_masks

    layer_area_list = []
    for struct_masks in layerd_struct_masks:
        area_list = []
        for struct_mask in struct_masks:
            area = np.sum(struct_mask.astype(np.float32))
            area_list.append(area)
        layer_area_list.append(area_list)
    indices = get_top_n_indices(layer_area_list,N)

    new_layerd_struct_masks = []
    for i,struct_masks in enumerate(layerd_struct_masks):
        new_struct_masks = []
        for j,struct_mask in enumerate(struct_masks):
            if (i,j) in indices:
                new_struct_masks.append(struct_mask)
        if len(new_struct_masks)>0:
            new_layerd_struct_masks.append(new_struct_masks)
    return new_layerd_struct_masks


def init_struct_target_imgs(layerd_struct_masks: list):
    struct_target_imgs = []
    struct_colors_list = []
    for i,monolayer_masks in enumerate(layerd_struct_masks):
        seg_image = 0
        mask_colors = []
        for mask in monolayer_masks:
            tensor0 = torch.zeros((3, 512, 512))
            color = []
            for channel in range(3):
                channel_value = 0.2 + (1 - 0.2) * torch.rand(1)
                tensor0[channel, :, :] = channel_value
                color.append(channel_value)
            mask_colors.append(color)

            # mask_image1 = Image.open(mask)
            mask_image1 = transforms.ToTensor()(Image.fromarray(mask))
            seg_image += tensor0*mask_image1
            seg_image = torch.clamp(seg_image, max=1.0)

        struct_target_imgs.append(seg_image)
        struct_colors_list.append(mask_colors)
    return struct_target_imgs, struct_colors_list


def find_closest_contours(contours):
    min_distance = float('inf')
    contours = [x[:, 0, :] for x in contours]
    for i, contour1 in enumerate(contours):
        for j, contour2 in enumerate(contours):
            if i >= j:
                continue
            distances = cdist(contour1, contour2)
            distance = np.min(distances)
            if distance < min_distance:
                min_distance = distance 
                row_idx, col_idx = np.where(distances == np.min(min_distance))
                closest_point_1 = contour1[row_idx[0]]
                closest_point_2 = contour2[col_idx[0]]
    return closest_point_1, closest_point_2



def save_contours_as_single_png(contour1, contour2, filename):
    # 找出所有轮廓点的最小和最大坐标
    all_points = np.vstack((contour1, contour2))
    min_x = np.min(all_points[:, 0, 0])
    max_x = np.max(all_points[:, 0, 0])
    min_y = np.min(all_points[:, 0, 1])
    max_y = np.max(all_points[:, 0, 1])

    # 创建空白图像
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 调整轮廓坐标以适应图像
    contour1_adj = contour1 - [min_x, min_y]
    contour2_adj = contour2 - [min_x, min_y]

    # 绘制轮廓
    cv2.drawContours(image, [contour1_adj], -1, (255, 255, 255), 1)
    cv2.drawContours(image, [contour2_adj], -1, (255, 255, 255), 1)

    # 保存图像
    cv2.imwrite(filename, image)


def init_path_by_mask(mask: Union[np.ndarray,torch.Tensor], epsilon: int = 5):
    def insert_points_in_segments(points, num_interpolations=2):
        def interpolate_points(point1, point2, num_interpolations):
            x_values = np.linspace(point1[0], point2[0], num=num_interpolations + 2)[1:-1]
            y_values = np.linspace(point1[1], point2[1], num=num_interpolations + 2)[1:-1]
            interpolated_points = np.column_stack((x_values, y_values))
            return interpolated_points
        new_points = []
        for i in range(len(points) - 1):
            point1 = points[i]
            point2 = points[i + 1]
            new_points.append(point1)
            interpolated_points = interpolate_points(point1, point2, num_interpolations)
            new_points.extend(interpolated_points)
        new_points.append(points[-1])
        return np.array(new_points)

    if isinstance(mask,torch.Tensor):
        mask = (mask).detach().cpu().numpy()
        mask = (mask*255).astype(np.uint8)

    mask = np.where(mask > 1, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    while len(contours)>1:
        point_1, point_2 = find_closest_contours(contours)
        cv2.line(mask, point_1, point_2, (255), 3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour = contours[0]

    simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
    if len(simplified_contour)==2:
        new_point = np.array([simplified_contour[0,0,0]+random.uniform(2, 5),simplified_contour[0,0,1]+random.uniform(2, 5)])
        simplified_contour = np.insert(simplified_contour, 1, new_point, axis=0)
    if len(simplified_contour)==1:
        new_point = np.array([simplified_contour[0,0,0]+3,simplified_contour[0,0,1]+3])
        simplified_contour = np.insert(simplified_contour, 1, new_point, axis=0)
        new_point = np.array([simplified_contour[0,0,0]-3,simplified_contour[0,0,1]+3])
        simplified_contour = np.insert(simplified_contour, 1, new_point, axis=0)
    simplified_contour = simplified_contour[:,0,:]
    points = np.vstack((simplified_contour, simplified_contour[0]))
    # Insert control points for cubic Bezier curves
    points = insert_points_in_segments(points, num_interpolations=2)
    points = torch.FloatTensor(points[:-1])
    num_control_points = [2] * len(simplified_contour)
    path = pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=points,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True
                    )
    return path

def get_mean_color(image: np.ndarray, mask: np.ndarray):
        masked_pixels = image[mask > 0]
        if len(masked_pixels) > 0:
            average_color = tuple(np.mean(masked_pixels, axis=0, dtype=int))
        else:
            average_color = (0, 0, 0)
        return average_color

def init_svg_by_mask(layerd_masks: list,target_img: np.ndarray, epsilon:int = 5):
    shapes =[]
    shape_groups=[]
    for masks in layerd_masks:
        for mask in masks:
            path = init_path_by_mask(mask,epsilon=epsilon)
            mean_color = get_mean_color(target_img,mask)
            path_group = pydiffvg.ShapeGroup(
                                    shape_ids=torch.LongTensor([len(shapes)]),
                                    fill_color=torch.tensor(list(mean_color)+[255])/255,
                                    stroke_color=torch.FloatTensor([0,0,0,1])
                                )
            shapes.append(path)
            shape_groups.append(path_group)
    return shapes,shape_groups


def color_fitting(shape_groups, target_img: np.ndarray, layerd_struct_masks: list, is_cluster: bool, k: int = 30):
    def combine_masks(masks):
        combine_masks = masks[0].copy()
        if len(masks)>1:
            for mask in masks[1:]:
                combine_masks+=mask
        return combine_masks
    if is_cluster:
        data = target_img.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        target_img = segmented_data.reshape((target_img.shape))

    most_common_color_list =[]
    for i,masks in enumerate(layerd_struct_masks):
        if i<=len(layerd_struct_masks)-2:
            masks_combine = combine_masks(layerd_struct_masks[i+1])
        else:
            masks_combine = np.zeros_like(masks[0])
        for j,mask in enumerate(masks):
            mask = mask - masks_combine
            mask_indices = np.where(mask == 255)
            colors = target_img[mask_indices]
            color_tuples = [tuple(color) for color in colors]
            color_counts = Counter(color_tuples)
            if color_counts:
                most_common_color = color_counts.most_common(1)[0][0]
            else:
                most_common_color = (0,0,0)
            most_common_color_list.append(list(most_common_color))
    for i,shape_group in enumerate(shape_groups):
        shape_group.fill_color=torch.FloatTensor(most_common_color_list[i]+[255])/255
    return shape_groups,target_img

def select_mask_by_conn_area(
        pred: np.ndarray,
        gt: np.ndarray,
        saliency_map: np.ndarray,
        N: int = -1,
        area_weight: float = 2.0,
        saliency_log_strength: float = 10.0,
        min_area: int = 4
    ):
    """
    Select masks using a unified leaderboard strategy.
    Ranking is based on a composite score of Focal Intensity (saliency * magnitude)
    and Spatial Presence (weighted area).

    Args:
        pred: Predicted image
        gt: Ground truth image
        saliency_map: Saliency map with values in [0, 1]
        N: Maximum number of masks to return (-1 for all)
        area_weight: Coefficient to boost importance of large areas.
                     Higher values (e.g., 2.0-5.0) allow large background blobs 
                     to outrank small, mediocre focal details.
        min_area: Hard filter for minimum pixel size to ignore pure noise.
    """
    # 1. Compute Difference Map (unchanged logic for consistency)
    map_diff = ((pred - gt)**2).sum(0)
    
    quantile_interval = 139
    nodiff_thres = 0.025
    map_diff[map_diff < nodiff_thres] = 0

    q_vals = np.linspace(0., 1., quantile_interval)
    quantized_interval = np.quantile(map_diff, q_vals)
    quantized_interval = np.unique(quantized_interval)
    quantized_interval = sorted(quantized_interval[1:-1])
    
    # map_bins contains values 0..255 indicating error magnitude
    map_bins = np.digitize(map_diff, quantized_interval, right=False)
    map_bins = np.clip(map_bins, 0, 255).astype(np.uint8)

    # 2. Extract Connected Components
    # We flatten the process to collect all candidates into one list immediately
    candidates = []
    image_total_area = map_bins.shape[0] * map_bins.shape[1]
    
    # Iterate over non-zero bin levels
    unique_levels = np.unique(map_bins)
    if unique_levels[0] == 0:
        unique_levels = unique_levels[1:]

    for level in unique_levels:
        # Create binary mask for this level
        level_mask = (map_bins == level).astype(np.uint8)
        
        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(level_mask, connectivity=4)
        
        # Iterate components (skip label 0 which is background of the mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < min_area:
                continue

            # Extract mask for this specific component
            comp_mask = (labels == i)
            
            # --- SCORING METRICS ---
            
            # 1. Focal Intensity: Saliency * Magnitude
            # (Magnitude is normalized 0-1 from the bin level)
            norm_magnitude = level / 255.0
            
            # Mean saliency under the mask
            # Optimization: Use mask slicing/indexing rather than full image multiplication if possible
            # but boolean indexing is fast enough here.
            # 1. Saliency with Log Smoothing
            raw_saliency = np.mean(saliency_map[comp_mask])
            norm_saliency = raw_saliency / 255.0
            
            if saliency_log_strength > 0:
                # Log normalization formula: ln(1 + a*x) / ln(1 + a)
                # This maps 0->0 and 1->1, but boosts the middle.
                log_saliency = np.log(1 + saliency_log_strength * norm_saliency) / \
                               np.log(1 + saliency_log_strength)
            else:
                log_saliency = norm_saliency
            
            focal_score = log_saliency * norm_magnitude
            
            # 2. Area Term (Geometric Fix)
            # We use SQRT to make Area linearly comparable to Intensity
            # Example: 1% area (0.01) -> sqrt -> 0.1 (Now it matches magnitude scale)
            norm_area_linear = area / image_total_area
            norm_area = np.sqrt(norm_area_linear)
            
            # --- FINAL SCORE ---
            # Score = (Intensity) + (Weighted Area)
            # or 'area_weight' to make background pop more.
            final_score = focal_score + (norm_area * area_weight)
            
            candidates.append({
                'mask': comp_mask,  # Store boolean mask directly
                'score': final_score,
                'metrics': {
                    'saliency': log_saliency,
                    'magnitude': norm_magnitude,
                    'area': norm_area
                }
            })

    # 3. Leaderboard Sorting
    # Sort descending by score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # 4. Selection
    if N > 0:
        selected_candidates = candidates[:N]
    else:
        selected_candidates = candidates
    
    # 5. Convert to Output Format (uint8 0-255 masks)
    final_masks = []
    for c in selected_candidates:
        # Convert boolean mask to uint8 255
        m = c['mask'].astype(np.uint8) * 255
        final_masks.append(m)
        
    return final_masks

def insert_in_struct_layer(mask: Union[np.ndarray,torch.Tensor],struct_masks: List[np.ndarray]):
    if isinstance(mask,torch.Tensor):
        mask = (mask).detach().cpu().numpy()
        mask = (mask*255).astype(np.uint8)
    is_pseudo_struct_mask = False
    for index,pseudo_struct_mask in enumerate(struct_masks):
        mask_area = np.sum(mask==255)
        pseudo_struct_area = np.sum(pseudo_struct_mask==255)
        intersection_area = np.sum((mask.astype(np.uint16)+pseudo_struct_mask.astype(np.uint16))==510)
        if intersection_area/pseudo_struct_area>=0.7 and mask_area > 1.1*pseudo_struct_area:
            struct_masks.insert(index,np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.uint8))
            is_pseudo_struct_mask = True
            break
    return is_pseudo_struct_mask,struct_masks,index


def add_visual_paths(shapes,shape_groups,device,
                     struct_path_num: int,
                     target_img: np.ndarray, 
                     pseudo_struct_masks: List[np.ndarray], 
                     is_opt_list: List[int] = [],
                     epsilon: int=5, 
                     N: int = 50,
                     saliency_map: np.ndarray = None):
    if len(is_opt_list)==0:
        is_opt_list = [0 for i in range(len(shapes))]

    img_height, img_width = target_img.shape[:2]
    raster_img = svg_to_img(img_height,img_width,shapes,shape_groups,device)
    raster_img = rgba_to_rgb(raster_img,device=device)
    raster_img = raster_img.detach().cpu().numpy()
    target_img1 = np.transpose((target_img/255).astype(np.float16), (2, 0, 1))

    masks = select_mask_by_conn_area(
        pred=raster_img, 
        gt=target_img1, 
        saliency_map=saliency_map, 
        N=N
    )

    if len(masks) == 0:
        return shapes,shape_groups,pseudo_struct_masks,is_opt_list,-1

    for mask in masks:
        color = get_mean_color(target_img,mask)
        path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([0]),
                                fill_color=torch.FloatTensor(list(color)+[255])/255,
                                stroke_color=torch.FloatTensor([0,0,0,1])
                            )

        mask = connect_mask_interior_exterior(mask)
        # print(np.max(mask))
        path = init_path_by_mask(mask,epsilon)
        is_pseudo_struct_mask,pseudo_struct_masks,insert_index=insert_in_struct_layer(mask,pseudo_struct_masks)
        if is_pseudo_struct_mask:
            shapes.insert(insert_index,path)
            shape_groups.insert(insert_index,path_group)
            is_opt_list.insert(insert_index,1)
            struct_path_num += 1
        else:
            shapes.append(path)
            shape_groups.append(path_group)
            is_opt_list.append(1)

    for index,shape_group1 in enumerate(shape_groups):
        shape_group1.shape_ids=torch.LongTensor([index])
    return shapes,shape_groups,pseudo_struct_masks,is_opt_list,struct_path_num

# def merge_path(shapes,shape_groups,device,
#                img_width: int, 
#                img_height: int,
#                struct_path_num: int,
#                pseudo_struct_masks: List[np.ndarray], 
#                is_opt_list: List[int],
#                color_threshold: float = 0.05,
#                overlapping_area_threshold: int = 3):
    
#     def is_merge(shape_img,shape_img1,color,color1):
#         if torch.sum((shape_img+shape_img1) > 1.9).item() >= overlapping_area_threshold and torch.sum(torch.abs(color-color1))<=color_threshold:
#             return True
#         else:
#             return False

#     def merge_obj(shape_img,shape_img1,color,color1,record1,record2):
#         new_shape_img = shape_img+shape_img1
#         new_shape_img = torch.clamp(new_shape_img,max=1)
#         new_color = ((color+color1)/2).detach()
#         new_record=record1+record2
#         return new_shape_img,new_color,new_record
    
#     def merge_objs_based_on_threshold(path_raster_imgs,path_color_list):
#         record_list = [[i] for i in range(len(path_raster_imgs))]
#         merged = False
#         i = 0
#         while i < len(path_raster_imgs):
#             j = i + 1
#             while j < len(path_raster_imgs):
#                 if is_merge(path_raster_imgs[i],path_raster_imgs[j],path_color_list[i],path_color_list[j]):
#                     new_shape_img,new_color,new_record = merge_obj(path_raster_imgs[i],path_raster_imgs[j],path_color_list[i],path_color_list[j],record_list[i],record_list[j])
#                     path_raster_imgs[i] = new_shape_img
#                     path_raster_imgs.pop(j)
#                     path_color_list[i] = new_color
#                     path_color_list.pop(j)
#                     record_list[i] = new_record
#                     record_list.pop(j)
#                     merged = True
#                 else:
#                     j += 1
#             if not merged:
#                 i += 1
#             else:
#                 merged = False
#         path_raster_imgs = [x for i,x in enumerate(path_raster_imgs) if len(record_list[i])>1]
#         path_color_list = [x for i,x in enumerate(path_color_list) if len(record_list[i])>1]
#         record_list = [x for i,x in enumerate(record_list) if len(record_list[i])>1]
#         return record_list,path_raster_imgs,path_color_list
    
#     struct_shapes,struct_shape_groups = shapes[:struct_path_num],shape_groups[:struct_path_num]
#     visual_shapes,visual_shape_groups = shapes[struct_path_num:],shape_groups[struct_path_num:]

#     black_pg = torch.tensor([0., 0., 0.], requires_grad=False, device=device)
#     white_path_group = pydiffvg.ShapeGroup(
#                                 shape_ids=torch.LongTensor([0]),
#                                 fill_color=torch.FloatTensor([1,1,1,1]),
#                                 stroke_color=torch.FloatTensor([1,1,1,1])
#                             )
#     path_raster_imgs=[]
#     for shape in visual_shapes:
#         img = svg_to_img(img_width,img_height,[shape],[white_path_group],device)
#         img = rgba_to_rgb(img,device=device,para_bg=black_pg)
#         img = img[0]
#         path_raster_imgs.append(img)
#     path_color_list=[]
#     for shape_group in visual_shape_groups:
#         path_color_list.append(shape_group.fill_color)

#     record_list,merge_imgs,merge_color_list = merge_objs_based_on_threshold(path_raster_imgs,path_color_list)

#     flattened_record_list = [item for sublist in record_list for item in sublist]       
#     visual_shapes = [x for i,x in enumerate(visual_shapes) if i not in flattened_record_list]
#     visual_shape_groups = [x for i,x in enumerate(visual_shape_groups) if i not in flattened_record_list]
#     is_opt_list = is_opt_list[:len(shapes)-len(flattened_record_list)]

#     for i,record in enumerate(record_list):
#         if len(record)>1:
#             new_mask = connect_mask_interior_exterior(merge_imgs[i])
#             merge_path = init_path_by_mask(new_mask)
#             merge_path_group = pydiffvg.ShapeGroup(
#                             shape_ids=torch.LongTensor([0]),
#                             fill_color=merge_color_list[i],
#                             stroke_color=torch.FloatTensor([0,0,0,1])
#                         )
#             is_pseudo_struct_mask,pseudo_struct_masks,insert_index = insert_in_struct_layer(merge_imgs[i],pseudo_struct_masks)
#             if is_pseudo_struct_mask:
#                 struct_shapes.insert(insert_index,merge_path)
#                 struct_shape_groups.insert(insert_index,merge_path_group)
#                 is_opt_list.insert(insert_index,1)
#                 struct_path_num += 1
#             else:
#                 visual_shapes.append(merge_path)
#                 visual_shape_groups.append(merge_path_group)
#                 is_opt_list.append(1)
#     shapes = struct_shapes+visual_shapes
#     shape_groups = struct_shape_groups+visual_shape_groups

#     for index,shape_group in enumerate(shape_groups):
#         shape_group.shape_ids=torch.LongTensor([index])   

#     return shapes,shape_groups,pseudo_struct_masks,is_opt_list,struct_path_num

def merge_path(shapes,shape_groups,device,
               img_width: int, 
               img_height: int,
               struct_path_num: int,
               pseudo_struct_masks: List[np.ndarray], 
               is_opt_list: List[int],
               color_threshold: float = 0.05,
               overlapping_area_threshold: int = 3):
    
    def is_merge(shape_img,shape_img1,color,color1):
        if torch.sum((shape_img+shape_img1) > 1.9).item() >= overlapping_area_threshold and torch.sum(torch.abs(color-color1))<=color_threshold:
            return True
        else:
            return False

    def merge_obj(shape_img,shape_img1,color,color1,record1,record2):
        new_shape_img = shape_img+shape_img1
        new_shape_img = torch.clamp(new_shape_img,max=1)
        new_color = ((color+color1)/2).detach()
        new_record=record1+record2
        return new_shape_img,new_color,new_record
    
    def merge_objs_based_on_threshold(path_raster_imgs,path_color_list):
        record_list = [[i] for i in range(len(path_raster_imgs))]
        merged = False
        i = 0
        while i < len(path_raster_imgs):
            j = i + 1
            while j < len(path_raster_imgs):
                if is_merge(path_raster_imgs[i],path_raster_imgs[j],path_color_list[i],path_color_list[j]):
                    new_shape_img,new_color,new_record = merge_obj(path_raster_imgs[i],path_raster_imgs[j],path_color_list[i],path_color_list[j],record_list[i],record_list[j])
                    path_raster_imgs[i] = new_shape_img
                    path_raster_imgs.pop(j)
                    path_color_list[i] = new_color
                    path_color_list.pop(j)
                    record_list[i] = new_record
                    record_list.pop(j)
                    merged = True
                else:
                    j += 1
            if not merged:
                i += 1
            else:
                merged = False
        path_raster_imgs = [x for i,x in enumerate(path_raster_imgs) if len(record_list[i])>1]
        path_color_list = [x for i,x in enumerate(path_color_list) if len(record_list[i])>1]
        record_list = [x for i,x in enumerate(record_list) if len(record_list[i])>1]
        return record_list,path_raster_imgs,path_color_list
    
    struct_shapes,struct_shape_groups = shapes[:struct_path_num],shape_groups[:struct_path_num]
    visual_shapes,visual_shape_groups = shapes[struct_path_num:],shape_groups[struct_path_num:]

    black_pg = torch.tensor([0., 0., 0.], requires_grad=False, device=device)
    white_path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([0]),
                                fill_color=torch.FloatTensor([1,1,1,1]),
                                stroke_color=torch.FloatTensor([1,1,1,1])
                            )
    path_raster_imgs=[]
    for shape in visual_shapes:
        img = svg_to_img(img_width,img_height,[shape],[white_path_group],device)
        img = rgba_to_rgb(img,device=device,para_bg=black_pg)
        img = img[0]
        path_raster_imgs.append(img)
    path_color_list=[]
    for shape_group in visual_shape_groups:
        path_color_list.append(shape_group.fill_color)

    record_list,merge_imgs,merge_color_list = merge_objs_based_on_threshold(path_raster_imgs,path_color_list)

    flattened_record_list = [item for sublist in record_list for item in sublist]       
    visual_shapes = [x for i,x in enumerate(visual_shapes) if i not in flattened_record_list]
    visual_shape_groups = [x for i,x in enumerate(visual_shape_groups) if i not in flattened_record_list]
    is_opt_list = is_opt_list[:len(shapes)-len(flattened_record_list)]

    for i,record in enumerate(record_list):
        if len(record)>1:
            new_mask = connect_mask_interior_exterior(merge_imgs[i])
            merge_path = init_path_by_mask(new_mask)
            merge_path_group = pydiffvg.ShapeGroup(
                            shape_ids=torch.LongTensor([0]),
                            fill_color=merge_color_list[i],
                            stroke_color=torch.FloatTensor([0,0,0,1])
                        )
            
            visual_shapes.append(merge_path)
            visual_shape_groups.append(merge_path_group)
            is_opt_list.append(1)
    shapes = struct_shapes+visual_shapes
    shape_groups = struct_shape_groups+visual_shape_groups

    for index,shape_group in enumerate(shape_groups):
        shape_group.shape_ids=torch.LongTensor([index])   

    return shapes,shape_groups,pseudo_struct_masks,is_opt_list,struct_path_num

def find_closest_pair(contour1,contour2):
        closest_pair = (None, None)
        min_distance = float('inf')
        for outer_point in contour1:
            for inner_point in contour2:
                dist = np.linalg.norm(outer_point[0] - inner_point[0])  # 计算距离
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (tuple(outer_point[0]), tuple(inner_point[0]))
        return closest_pair

def connect_mask_interior_exterior(mask: Union[np.ndarray,torch.Tensor]):
    if isinstance(mask,torch.Tensor):
        mask = (mask).detach().cpu().numpy()
        mask = (mask*255).astype(np.uint8)
    mask = np.where(mask > 1, 255, 0).astype(np.uint8)
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_areas = [cv2.contourArea(contour) for contour in contours]
    sorted_contours = sorted(zip(contour_areas, contours), key=lambda x: x[0], reverse=True)
    if len(contours)>1:
        for contou in sorted_contours[1:]:
            closest_pair=find_closest_pair(sorted_contours[0][1],contou[1])  
            cv2.line(mask, closest_pair[0], closest_pair[1], (0), 3)  # 连接线 
    return mask

def remove_lowquality_paths(shapes, shape_groups,device,
                img_width: int, 
                img_height: int,
                visual_difference_threshold: float = 8.0,
                struct_path_num: int = 0,
                ):
    img = svg_to_img(img_width,img_height,shapes,shape_groups,device)
    img = rgba_to_rgb(img,device)
    remove_index_list = []
    for i in range(len(shapes)):
        shapes_copy=[x for j,x in enumerate(shapes) if j != i]
        shape_groups_copy=[x for j,x in enumerate(shape_groups) if j != i]
        for j in range(len(shape_groups_copy)):
            shape_groups_copy[j].shape_ids=torch.tensor([j])
        img_j = svg_to_img(img_width,img_height,shapes_copy,shape_groups_copy,device)
        img_j = rgba_to_rgb(img_j,device)
        pixel_difference = torch.sum(torch.abs(img_j-img)).item()
        if pixel_difference<=visual_difference_threshold:
            remove_index_list.append(i)

    if struct_path_num>0:
        remove_index_list = [x for x in remove_index_list if x >= struct_path_num]

    shapes = [x for i,x in enumerate(shapes) if i not in remove_index_list]
    shape_groups = [x for i,x in enumerate(shape_groups) if i not in remove_index_list]
    for i in range(len(shape_groups)):
        shape_groups[i].shape_ids=torch.tensor([i])
    return shapes,shape_groups


def _get_path_orientation(points: torch.Tensor) -> float:
    """Return +1.0 or -1.0 for the closed cubic Bezier path's winding, or 0.0 if degenerate."""
    num_segs = points.shape[0] // 3
    if num_segs < 3:
        return 0.0
    t_vals = torch.tensor([0.0, 0.25, 0.5, 0.75], device=points.device, dtype=points.dtype)
    sampled = []
    for k in range(num_segs):
        p0 = points[3 * k]
        c1 = points[3 * k + 1]
        c2 = points[3 * k + 2]
        p1 = points[(3 * ((k + 1) % num_segs))]
        for t in t_vals:
            omt = 1.0 - t
            pt = (omt ** 3) * p0 + 3.0 * (omt ** 2) * t * c1 + 3.0 * omt * (t ** 2) * c2 + (t ** 3) * p1
            sampled.append(pt)
    poly = torch.stack(sampled, dim=0)
    poly_next = torch.roll(poly, shifts=-1, dims=0)
    signed_area = 0.5 * torch.sum(poly[:, 0] * poly_next[:, 1] - poly_next[:, 0] * poly[:, 1]).item()
    if abs(signed_area) < 1e-6:
        return 0.0
    return 1.0 if signed_area > 0 else -1.0


def _find_concave_vertices(points: torch.Tensor, orient: float, eps: float = 1e-3) -> List[int]:
    """Return indices of anchor vertices that form a concave angle based on anchor chords."""
    num_segs = points.shape[0] // 3
    concave = []
    for k in range(num_segs):
        prev_k = (k - 1) % num_segs
        next_k = (k + 1) % num_segs
        pk = points[3 * k]
        v_in = pk - points[3 * prev_k]
        v_out = points[3 * next_k] - pk
        n_in = torch.norm(v_in).item()
        n_out = torch.norm(v_out).item()
        if n_in < 1e-6 or n_out < 1e-6:
            continue
        v_in = v_in / n_in
        v_out = v_out / n_out
        cross_val = orient * (v_in[0] * v_out[1] - v_in[1] * v_out[0]).item()
        if cross_val < -eps:
            concave.append(k)
    return concave


def _is_segment_straight(
    p0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    p1: torch.Tensor,
    flat_tol: float = 0.5,
) -> bool:
    """Return True if both control points lie within flat_tol of the chord p0 -> p1."""
    chord = p1 - p0
    chord_len = torch.norm(chord).item()
    if chord_len < 1e-4:
        return True
    chord_unit = chord / chord_len
    d1 = abs((c1[0] - p0[0]) * chord_unit[1] - (c1[1] - p0[1]) * chord_unit[0]).item()
    d2 = abs((c2[0] - p0[0]) * chord_unit[1] - (c2[1] - p0[1]) * chord_unit[0]).item()
    return max(d1, d2) < flat_tol


def _find_concave_segments(points: torch.Tensor, orient: float, flat_tol: float = 0.5, curv_eps: float = 1e-2) -> List[int]:
    """Return indices of cubic Bezier segments that bend inward (concave)."""
    num_segs = points.shape[0] // 3
    concave = []
    t_vals = [0.2, 0.4, 0.6, 0.8]
    for k in range(num_segs):
        p0 = points[3 * k]
        c1 = points[3 * k + 1]
        c2 = points[3 * k + 2]
        p1 = points[3 * ((k + 1) % num_segs)]
        if _is_segment_straight(p0, c1, c2, p1, flat_tol):
            continue
        min_curv = float("inf")
        for t in t_vals:
            omt = 1.0 - t
            dp = 3.0 * (omt ** 2) * (c1 - p0) + 6.0 * omt * t * (c2 - c1) + 3.0 * (t ** 2) * (p1 - c2)
            ddp = 6.0 * omt * (c2 - 2.0 * c1 + p0) + 6.0 * t * (p1 - 2.0 * c2 + c1)
            curv_num = orient * (dp[0] * ddp[1] - dp[1] * ddp[0]).item()
            if curv_num < min_curv:
                min_curv = curv_num
        if min_curv < -curv_eps:
            concave.append(k)
    return concave


def _remove_vertex_harmonic(
    path: pydiffvg.Path, k: int, flat_tol: float = 0.5, harmonic: bool = True
) -> pydiffvg.Path:
    """Remove anchor vertex k and bridge P_{k-1} -> P_{k+1} with harmonic or straight-chord Bezier handles."""
    pts = path.points.detach().clone()
    num_segs = pts.shape[0] // 3
    prev_k = (k - 1) % num_segs
    next_k = (k + 1) % num_segs
    prev2_k = (k - 2) % num_segs
    next2_k = (k + 2) % num_segs

    p_prev = pts[3 * prev_k]
    p_next = pts[3 * next_k]
    chord = p_next - p_prev
    chord_len = torch.norm(chord).item()

    c2_other = pts[3 * prev2_k + 2]
    prev_seg_curved = harmonic and not _is_segment_straight(
        pts[3 * prev2_k], pts[3 * prev2_k + 1], c2_other, p_prev, flat_tol
    )
    u = p_prev - c2_other
    u_len = torch.norm(u).item()
    if prev_seg_curved and u_len > 1e-4 and chord_len > 1e-4:
        c1_new = p_prev + (u / u_len) * min(u_len, chord_len / 3.0)
    else:
        c1_new = p_prev + chord / 3.0

    c1_other = pts[3 * next_k + 1]
    next_seg_curved = harmonic and not _is_segment_straight(
        p_next, c1_other, pts[3 * next_k + 2], pts[3 * next2_k], flat_tol
    )
    v = p_next - c1_other
    v_len = torch.norm(v).item()
    if next_seg_curved and v_len > 1e-4 and chord_len > 1e-4:
        c2_new = p_next + (v / v_len) * min(v_len, chord_len / 3.0)
    else:
        c2_new = p_next - chord / 3.0

    pts[3 * prev_k + 1] = c1_new
    pts[3 * k + 2] = c2_new

    drop_indices = {3 * prev_k + 2, 3 * k, 3 * k + 1}
    keep_mask = [idx for idx in range(pts.shape[0]) if idx not in drop_indices]
    new_pts = pts[keep_mask]
    if k == 0:
        new_pts = torch.roll(new_pts, shifts=-1, dims=0)

    new_num_ctrl = torch.full((num_segs - 1,), 2, dtype=path.num_control_points.dtype, device=path.num_control_points.device)
    return pydiffvg.Path(
        num_control_points=new_num_ctrl,
        points=new_pts,
        stroke_width=path.stroke_width.detach().clone(),
        is_closed=path.is_closed,
    )


def _straighten_segment(path: pydiffvg.Path, k: int) -> pydiffvg.Path:
    """Straighten cubic Bezier segment k into a line between its endpoints."""
    pts = path.points.detach().clone()
    num_segs = pts.shape[0] // 3
    p0 = pts[3 * k]
    p1 = pts[3 * ((k + 1) % num_segs)]
    pts[3 * k + 1] = p0 + (p1 - p0) / 3.0
    pts[3 * k + 2] = p0 + 2.0 * (p1 - p0) / 3.0
    return pydiffvg.Path(
        num_control_points=path.num_control_points.detach().clone(),
        points=pts,
        stroke_width=path.stroke_width.detach().clone(),
        is_closed=path.is_closed,
    )


def _render_alpha_mask(width: int, height: int, shapes_subset: list, device) -> torch.Tensor:
    """Render a subset of shapes into a boolean coverage mask (H, W)."""
    if len(shapes_subset) == 0:
        return torch.zeros((height, width), dtype=torch.bool, device=device)
    white_groups = [
        pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([idx]),
            fill_color=torch.FloatTensor([1.0, 1.0, 1.0, 1.0]),
            stroke_color=torch.FloatTensor([0.0, 0.0, 0.0, 0.0]),
        )
        for idx in range(len(shapes_subset))
    ]
    rendered = svg_to_img(width, height, shapes_subset, white_groups, device)
    return rendered[:, :, 3] > 0.5


def _reverse_cubic_path_points(points: torch.Tensor) -> torch.Tensor:
    """Reverse the winding direction of a closed cubic Bezier path's points tensor (3*N, 2)."""
    num_segs = points.shape[0] // 3
    rev_pts = torch.empty_like(points)
    for m in range(num_segs):
        k = (num_segs - 1 - m) % num_segs
        next_k = (k + 1) % num_segs
        rev_pts[3 * m] = points[3 * next_k]
        rev_pts[3 * m + 1] = points[3 * k + 2]
        rev_pts[3 * m + 2] = points[3 * k + 1]
    return rev_pts


def _sample_bezier_points(points: torch.Tensor) -> np.ndarray:
    """Sample 4 points per cubic Bezier segment for fast spatial distance checks."""
    pts_np = points.detach().cpu().numpy()
    num_segs = pts_np.shape[0] // 3
    if num_segs == 0:
        return np.zeros((0, 2), dtype=np.float32)
    t_vals = (0.0, 0.25, 0.5, 0.75)
    sampled = []
    for k in range(num_segs):
        p0 = pts_np[3 * k]
        c1 = pts_np[3 * k + 1]
        c2 = pts_np[3 * k + 2]
        p1 = pts_np[3 * ((k + 1) % num_segs)]
        for t in t_vals:
            omt = 1.0 - t
            sampled.append((omt ** 3) * p0 + 3.0 * (omt ** 2) * t * c1 + 3.0 * omt * (t ** 2) * c2 + (t ** 3) * p1)
    return np.asarray(sampled, dtype=np.float32)


def _convex_hull_indices(pts: np.ndarray, orient: float) -> List[int]:
    """Return indices of 2D convex hull vertices ordered in the given winding orientation (+1 or -1)."""
    n = len(pts)
    if n <= 2:
        return list(range(n))
    order = np.lexsort((pts[:, 1], pts[:, 0]))

    def _cross(o_idx: int, a_idx: int, b_idx: int) -> float:
        return float(
            (pts[a_idx, 0] - pts[o_idx, 0]) * (pts[b_idx, 1] - pts[o_idx, 1])
            - (pts[a_idx, 1] - pts[o_idx, 1]) * (pts[b_idx, 0] - pts[o_idx, 0])
        )

    lower = []
    for idx in order:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], int(idx)) <= 1e-5:
            lower.pop()
        lower.append(int(idx))

    upper = []
    for idx in reversed(order):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], int(idx)) <= 1e-5:
            upper.pop()
        upper.append(int(idx))

    hull = lower[:-1] + upper[:-1]
    if orient < 0:
        hull.reverse()
    return hull


def _extract_cyclic_hull_runs(hull_tags: list) -> list:
    """Group cyclic [(shape_id, anchor_idx), ...] into maximal contiguous runs [(shape_id, start_idx, end_idx), ...]."""
    if not hull_tags:
        return []
    m = len(hull_tags)
    first_diff = 0
    for k in range(m):
        if hull_tags[k][0] != hull_tags[(k - 1) % m][0]:
            first_diff = k
            break
    rot = hull_tags[first_diff:] + hull_tags[:first_diff]
    runs = []
    curr_s, start_idx = rot[0]
    end_idx = start_idx
    for s, idx in rot[1:]:
        if s == curr_s:
            end_idx = idx
        else:
            runs.append((curr_s, start_idx, end_idx))
            curr_s = s
            start_idx = idx
            end_idx = idx
    runs.append((curr_s, start_idx, end_idx))
    return runs


def _build_spliced_bezier_path_from_runs(
    pts1: torch.Tensor,
    pts2: torch.Tensor,
    runs: list,
    stroke_width: torch.Tensor,
    is_closed: bool,
) -> pydiffvg.Path:
    """
    Splice two closed cubic Bezier paths along cyclic hull runs [(shape_id, start_idx, end_idx), ...],
    keeping each shape's original Bezier segments within each run and bridging transitions with straight chords.
    """
    n1 = pts1.shape[0] // 3
    n2 = pts2.shape[0] // 3
    seg_triples = []
    num_runs = len(runs)

    for r_idx in range(num_runs):
        s_id, start_idx, end_idx = runs[r_idx]
        next_s_id, next_start_idx, _ = runs[(r_idx + 1) % num_runs]

        curr_pts = pts1 if s_id == 1 else pts2
        curr_n = n1 if s_id == 1 else n2
        next_pts = pts1 if next_s_id == 1 else pts2
        next_n = n1 if next_s_id == 1 else n2

        u = start_idx % curr_n
        steps = (end_idx - start_idx) % curr_n
        for _ in range(steps):
            seg_triples.append(curr_pts[3 * u : 3 * u + 3])
            u = (u + 1) % curr_n

        p_start = curr_pts[3 * (end_idx % curr_n)]
        p_end = next_pts[3 * (next_start_idx % next_n)]
        chord = p_end - p_start
        if torch.norm(chord).item() > 1e-4:
            bridge = torch.stack([p_start, p_start + chord / 3.0, p_start + 2.0 * chord / 3.0], dim=0)
            seg_triples.append(bridge)

    if len(seg_triples) < 3:
        return None

    new_pts = torch.cat(seg_triples, dim=0)
    num_new_segs = new_pts.shape[0] // 3
    new_num_ctrl = torch.full((num_new_segs,), 2, dtype=torch.long, device=pts1.device)
    return pydiffvg.Path(
        num_control_points=new_num_ctrl,
        points=new_pts,
        stroke_width=stroke_width.detach().clone(),
        is_closed=is_closed,
    )


def _generate_convex_bridge_candidates(path1: pydiffvg.Path, path2: pydiffvg.Path) -> List[pydiffvg.Path]:
    """
    Generate simple merged Bezier path candidates connecting path1 and path2 via their
    outer convex-hull common tangents while preserving existing outer Bezier curves.
    """
    pts1 = path1.points.detach().clone()
    pts2 = path2.points.detach().clone()
    n1 = pts1.shape[0] // 3
    n2 = pts2.shape[0] // 3
    if n1 < 3 or n2 < 3:
        return []

    orient1 = _get_path_orientation(pts1)
    orient2 = _get_path_orientation(pts2)
    if orient1 == 0.0 or orient2 == 0.0:
        return []
    if orient1 * orient2 < 0:
        pts2 = _reverse_cubic_path_points(pts2)

    anchors1 = pts1[0::3].cpu().numpy()
    anchors2 = pts2[0::3].cpu().numpy()

    dists = cdist(anchors1, anchors2)
    u_star, v_star = np.unravel_index(int(np.argmin(dists)), dists.shape)
    u_star, v_star = int(u_star), int(v_star)

    candidates = []
    seen_runs = set()

    window_configs = [
        (max(1, n1 // 4), max(1, n2 // 4)),
        (max(1, n1 // 3), max(1, n2 // 3)),
        (n1, n2),  # global hull
    ]

    for h1, h2 in window_configs:
        if h1 >= n1:
            idx1 = list(range(n1))
        else:
            idx1 = [(u_star + d) % n1 for d in range(-h1, h1 + 1)]
        if h2 >= n2:
            idx2 = list(range(n2))
        else:
            idx2 = [(v_star + d) % n2 for d in range(-h2, h2 + 1)]

        sub_pts = np.vstack([anchors1[idx1], anchors2[idx2]])
        hull = _convex_hull_indices(sub_pts, orient1)
        len1 = len(idx1)
        hull_tags = [(1, idx1[h]) if h < len1 else (2, idx2[h - len1]) for h in hull]
        runs = _extract_cyclic_hull_runs(hull_tags)

        if len(runs) == 1:
            if h1 >= n1 and h2 >= n2:
                candidates.append(path1 if runs[0][0] == 1 else path2)
            continue

        # For local windows only use clean 2-run bridges; for global hull allow multi-run envelope too
        if len(runs) == 2 or (h1 >= n1 and h2 >= n2 and len(runs) <= 6):
            runs_key = tuple(runs)
            if runs_key not in seen_runs:
                seen_runs.add(runs_key)
                cand = _build_spliced_bezier_path_from_runs(
                    pts1, pts2, runs, path1.stroke_width, path1.is_closed
                )
                if cand is not None and (cand.points.shape[0] // 3) >= 3:
                    candidates.append(cand)

    return candidates


def merge_struct_shapes(
    device,
    shapes: list,
    shape_groups: list,
    target_img: np.ndarray,
    layerd_struct_masks: list = None,
    distance_threshold: float = 60.0,
    max_stack_gap_pct: float = 0.20,
    color_range_pct: float = 0.05,
    max_area_change_pct: float = 0.10,
    sse_threshold: float = 25.0,
):
    """
    Merge structural shapes that are spatially close or overlapping, within a stack gap
    proportional to the total number of shapes (max_stack_gap_pct * len(shapes)), and have
    very similar colors.
    Merged shapes are bridged via their outer convex-hull common tangents, placed at the lower
    stack index of the two shapes with an area-weighted average color, and rejected if the fraction
    of area that changes color exceeds max_area_change_pct (or if SSE degrades beyond sse_threshold).
    """
    flat_masks = []
    flat_layer_ids = []
    if layerd_struct_masks is not None:
        for layer_idx, masks in enumerate(layerd_struct_masks):
            for mask in masks:
                flat_masks.append(mask)
                flat_layer_ids.append(layer_idx)

    has_valid_masks = (layerd_struct_masks is not None) and (len(flat_masks) == len(shapes))
    if len(shapes) <= 2:
        return shapes, shape_groups, layerd_struct_masks

    max_stack_gap = max(1, int(round(max_stack_gap_pct * len(shapes))))

    img_height, img_width = target_img.shape[:2]
    target_float = target_img.astype(np.float32) / 255.0
    target_tensor = (torch.tensor(target_img, device=device, dtype=torch.float32) / 255.0).permute(2, 0, 1)

    # Per-channel full dynamic range modulated by image contrast (95th - 5th percentile)
    full_range = target_float.max(axis=(0, 1)) - target_float.min(axis=(0, 1))
    p95 = np.percentile(target_float, 95, axis=(0, 1))
    p5 = np.percentile(target_float, 5, axis=(0, 1))
    contrast_range = p95 - p5
    effective_range = np.maximum(full_range * contrast_range, 0.05)
    channel_thresholds = color_range_pct * effective_range

    with torch.no_grad():
        curr_img = rgba_to_rgb(svg_to_img(img_width, img_height, shapes, shape_groups, device), device)
        curr_sse = torch.nn.functional.mse_loss(curr_img, target_tensor, reduction="sum").item()

        alphas = []
        areas = []
        sampled_curves = []
        for shape in shapes:
            alpha = _render_alpha_mask(img_width, img_height, [shape], device)
            area = max(int(torch.sum(alpha).item()), 1)
            alphas.append(alpha)
            areas.append(area)
            sampled_curves.append(_sample_bezier_points(shape.points))

        # Keep index 0 (full-canvas background) intact; iteratively merge shapes 1..N-1
        changed = True
        while changed:
            changed = False
            i = 1
            while i < len(shapes):
                j = i + 1
                merged_any = False
                while j < len(shapes) and (j - i) <= max_stack_gap:
                    c_i = np.clip(shape_groups[i].fill_color[:3].detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)
                    c_j = np.clip(shape_groups[j].fill_color[:3].detach().cpu().numpy(), 0.0, 1.0).astype(np.float32)

                    # 1. Contrast-aware per-channel color similarity
                    if np.any(np.abs(c_i - c_j) > channel_thresholds):
                        j += 1
                        continue

                    # 2. Spatial closeness check between Bezier curves
                    pts_i = sampled_curves[i]
                    pts_j = sampled_curves[j]
                    if len(pts_i) == 0 or len(pts_j) == 0:
                        j += 1
                        continue
                    min_dist = float(np.min(cdist(pts_i, pts_j)))
                    if min_dist > distance_threshold:
                        j += 1
                        continue

                    # 3. Generate convex-hull tangent-bridged Bezier candidates
                    candidates = _generate_convex_bridge_candidates(shapes[i], shapes[j])
                    if not candidates:
                        j += 1
                        continue

                    # Visible area of shapes i and j before merging
                    upper_j = torch.zeros((img_height, img_width), dtype=torch.bool, device=device)
                    for k in range(j + 1, len(shapes)):
                        upper_j = upper_j | alphas[k]
                    upper_i = upper_j | alphas[j]
                    for k in range(i + 1, j):
                        upper_i = upper_i | alphas[k]
                    vis_i = alphas[i] & (~upper_i)
                    vis_j = alphas[j] & (~upper_j)
                    vis_area = int(torch.sum(vis_i | vis_j).item())
                    if vis_area <= 0:
                        j += 1
                        continue

                    # Area-weighted fill color
                    w_i, w_j = areas[i], areas[j]
                    merged_rgb = (w_i * c_i + w_j * c_j) / float(w_i + w_j)
                    merged_fill = torch.FloatTensor(
                        [float(merged_rgb[0]), float(merged_rgb[1]), float(merged_rgb[2]), 1.0]
                    )
                    own_shift = float(max(np.max(np.abs(c_i - merged_rgb)), np.max(np.abs(c_j - merged_rgb))))

                    # 4. Evaluate candidates placed at the lower stack index i
                    best_cand = None
                    for cand_path in candidates:
                        shapes_cand = shapes[:i] + [cand_path] + shapes[i + 1 : j] + shapes[j + 1 :]
                        groups_prev = shape_groups[:i] + [None] + shape_groups[i + 1 : j] + shape_groups[j + 1 :]
                        groups_cand = []
                        for k_idx, src_g in enumerate(groups_prev):
                            if k_idx == i:
                                fill_c = merged_fill
                                stroke_c = shape_groups[i].stroke_color
                            else:
                                fill_c = src_g.fill_color
                                stroke_c = src_g.stroke_color
                            groups_cand.append(
                                pydiffvg.ShapeGroup(
                                    shape_ids=torch.LongTensor([k_idx]),
                                    fill_color=fill_c,
                                    stroke_color=stroke_c,
                                )
                            )

                        cand_img = rgba_to_rgb(
                            svg_to_img(img_width, img_height, shapes_cand, groups_cand, device), device
                        )
                        px_diff = torch.max(torch.abs(cand_img - curr_img), dim=0)[0]
                        changed_area = int(torch.sum(px_diff > (own_shift + 1e-3)).item())
                        change_frac = changed_area / float(vis_area)
                        if change_frac > max_area_change_pct:
                            continue

                        cand_sse = torch.nn.functional.mse_loss(cand_img, target_tensor, reduction="sum").item()
                        if (cand_sse - curr_sse) > sse_threshold:
                            continue

                        if best_cand is None or change_frac < best_cand[0]:
                            cand_alpha = _render_alpha_mask(img_width, img_height, [cand_path], device)
                            cand_area = max(int(torch.sum(cand_alpha).item()), 1)
                            best_cand = (
                                change_frac,
                                cand_sse,
                                cand_img,
                                cand_path,
                                cand_alpha,
                                cand_area,
                                shapes_cand,
                                groups_cand,
                            )

                    if best_cand is None:
                        j += 1
                        continue

                    # Accept merge at position i and remove j
                    _, curr_sse, curr_img, accepted_path, acc_alpha, acc_area, shapes, shape_groups = best_cand
                    alphas[i] = acc_alpha
                    areas[i] = acc_area
                    sampled_curves[i] = _sample_bezier_points(accepted_path.points)
                    alphas.pop(j)
                    areas.pop(j)
                    sampled_curves.pop(j)

                    if has_valid_masks:
                        flat_masks[i] = np.maximum(flat_masks[i], flat_masks[j])
                        flat_masks.pop(j)
                        flat_layer_ids.pop(j)

                    merged_any = True
                    changed = True
                    break

                if not merged_any:
                    i += 1

    if has_valid_masks:
        new_layer_dict = {}
        for m_u8, lid in zip(flat_masks, flat_layer_ids):
            if lid not in new_layer_dict:
                new_layer_dict[lid] = []
            new_layer_dict[lid].append(m_u8)
        layerd_struct_masks = [
            new_layer_dict[k] for k in sorted(new_layer_dict.keys()) if len(new_layer_dict[k]) > 0
        ]

    return shapes, shape_groups, layerd_struct_masks


def simplify_concave_shapes(
    device,
    shapes: list,
    shape_groups: list,
    target_img: np.ndarray,
    sse_threshold: float = 1.0,
    max_changed_pct: float = 0.0015,
    straighten_segments: bool = False,
):
    """Bottom-to-top pass removing concave angles/curves that do not visibly alter too much of the image or degrade SSE against target_img."""
    img_height, img_width = target_img.shape[:2]
    total_pixels = float(img_height * img_width)
    target_tensor = (torch.tensor(target_img, device=device, dtype=torch.float32) / 255.0).permute(2, 0, 1)

    with torch.no_grad():
        curr_img = rgba_to_rgb(svg_to_img(img_width, img_height, shapes, shape_groups, device), device)
        curr_sse = torch.nn.functional.mse_loss(curr_img, target_tensor, reduction="sum").item()

        for i in tqdm(range(len(shapes)), desc="Concave simplification", unit="shape"):
            def _accept_best(candidates: List[pydiffvg.Path]) -> bool:
                nonlocal curr_sse, curr_img
                best = None
                for cand_path in candidates:
                    shapes_cand = shapes[:i] + [cand_path] + shapes[i + 1 :]
                    cand_img = rgba_to_rgb(svg_to_img(img_width, img_height, shapes_cand, shape_groups, device), device)
                    changed_pct = torch.sum(torch.max(torch.abs(cand_img - curr_img), dim=0)[0] > 0.05).item() / total_pixels
                    if changed_pct > max_changed_pct:
                        continue
                    cand_sse = torch.nn.functional.mse_loss(cand_img, target_tensor, reduction="sum").item()
                    if (cand_sse - curr_sse) > sse_threshold:
                        continue

                    if best is None or cand_sse < best[0]:
                        best = (cand_sse, cand_path, cand_img)

                if best is None:
                    return False

                curr_sse, shapes[i], curr_img = best
                return True

            changed = True
            while changed:
                changed = False
                pts = shapes[i].points.detach()
                orient = _get_path_orientation(pts)
                if orient == 0.0:
                    break

                num_segs = pts.shape[0] // 3
                if num_segs > 3:
                    for k in _find_concave_vertices(pts, orient):
                        cand_harmonic = _remove_vertex_harmonic(shapes[i], k, harmonic=True)
                        cand_straight = _remove_vertex_harmonic(shapes[i], k, harmonic=False)
                        candidates = (
                            [cand_harmonic]
                            if torch.allclose(cand_harmonic.points, cand_straight.points, atol=1e-4)
                            else [cand_harmonic, cand_straight]
                        )
                        if _accept_best(candidates):
                            changed = True
                            break
                    if changed:
                        continue

                if straighten_segments:
                    for k in _find_concave_segments(pts, orient):
                        if _accept_best([_straighten_segment(shapes[i], k)]):
                            changed = True
                            break

    return shapes, shape_groups