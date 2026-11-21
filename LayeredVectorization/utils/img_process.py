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
    """Return indices of anchor vertices that form a concave angle."""
    num_segs = points.shape[0] // 3
    concave = []
    for k in range(num_segs):
        prev_k = (k - 1) % num_segs
        next_k = (k + 1) % num_segs
        pk = points[3 * k]
        v_in = pk - points[3 * prev_k + 2]
        if torch.norm(v_in).item() < 1e-4:
            v_in = pk - points[3 * prev_k]
        v_out = points[3 * k + 1] - pk
        if torch.norm(v_out).item() < 1e-4:
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


def _get_mask_bbox(mask_bool: np.ndarray):
    """Return (ymin, ymax, xmin, xmax) bounding box for a boolean mask."""
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)


def _bboxes_overlap(b1, b2) -> bool:
    """Return True if two (ymin, ymax, xmin, xmax) boxes have non-empty intersection."""
    return (b1[0] < b2[1]) and (b2[0] < b1[1]) and (b1[2] < b2[3]) and (b2[2] < b1[3])


def _fill_small_holes(mask_bool: np.ndarray, max_hole_area: int = 64) -> np.ndarray:
    """Fill small interior pinholes in a boolean mask without altering large interior holes."""
    h, w = mask_bool.shape[:2]
    inv = (~mask_bool).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] <= max_hole_area:
            left = stats[lbl, cv2.CC_STAT_LEFT]
            top = stats[lbl, cv2.CC_STAT_TOP]
            width = stats[lbl, cv2.CC_STAT_WIDTH]
            height = stats[lbl, cv2.CC_STAT_HEIGHT]
            if left > 0 and top > 0 and (left + width) < w and (top + height) < h:
                mask_bool[labels == lbl] = True
    return mask_bool


def merge_struct_shapes(
    device,
    shapes: list,
    shape_groups: list,
    target_img: np.ndarray,
    layerd_struct_masks: list,
    epsilon: int = 5,
    distance_threshold: int = 6,
    color_range_pct: float = 0.13,
    max_intervening_overlap: int = 10,
    min_contact_pixels: int = 4,
):
    """
    Merge structural shapes that are spatially close or partly overlapping,
    contiguous in the layer stack, and have relatively similar colors scaled by
    each channel's contrast-aware dynamic range and local boundary slope.
    """
    flat_masks = []
    flat_layer_ids = []
    for layer_idx, masks in enumerate(layerd_struct_masks):
        for mask in masks:
            flat_masks.append(mask)
            flat_layer_ids.append(layer_idx)

    if len(shapes) <= 2 or len(flat_masks) != len(shapes):
        return shapes, shape_groups, layerd_struct_masks

    img_height, img_width = target_img.shape[:2]
    target_float = target_img.astype(np.float32) / 255.0

    # Per-channel full dynamic range modulated by image contrast (95th - 5th percentile)
    full_range = target_float.max(axis=(0, 1)) - target_float.min(axis=(0, 1))
    p95 = np.percentile(target_float, 95, axis=(0, 1))
    p5 = np.percentile(target_float, 5, axis=(0, 1))
    contrast_range = p95 - p5
    effective_range = np.maximum(full_range * contrast_range, 0.05)
    channel_thresholds = color_range_pct * effective_range  # per-channel [R, G, B] tolerance

    with torch.no_grad():
        raw_masks = []
        for i, shape in enumerate(shapes):
            alpha = _render_alpha_mask(img_width, img_height, [shape], device).detach().cpu().numpy()
            if not np.any(alpha):
                alpha = flat_masks[i] > 127
            raw_masks.append(alpha)

    max_layer_id = max(flat_layer_ids) if flat_layer_ids else 0
    layer_union = [np.zeros((img_height, img_width), dtype=bool) for _ in range(max_layer_id + 1)]
    for m, lid in zip(raw_masks, flat_layer_ids):
        layer_union[lid] |= m

    upper_union_by_layer = [np.zeros((img_height, img_width), dtype=bool) for _ in range(max_layer_id + 1)]
    running_upper = np.zeros((img_height, img_width), dtype=bool)
    for l in range(max_layer_id, -1, -1):
        upper_union_by_layer[l] = running_upper.copy()
        running_upper |= layer_union[l]

    radius = max(1, (distance_threshold + 1) // 2 + 1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    records = []
    for i in range(len(shapes)):
        m_bool = raw_masks[i].copy()
        lid = flat_layer_ids[i]
        vis_bool = m_bool & (~upper_union_by_layer[lid])
        area = max(int(np.sum(m_bool)), 1)
        vis_area = int(np.sum(vis_bool))

        if vis_area >= 16:
            color = target_float[vis_bool].mean(axis=0).astype(np.float32)
        elif int(np.sum(m_bool)) > 0:
            color = target_float[m_bool].mean(axis=0).astype(np.float32)
        else:
            color = shape_groups[i].fill_color[:3].detach().cpu().numpy().astype(np.float32)

        dil_u8 = cv2.dilate(m_bool.astype(np.uint8) * 255, dilate_kernel)
        dil_bool = dil_u8 > 127

        records.append({
            "orig_idx": i,
            "layer_id": lid,
            "mask": m_bool,
            "vis_mask": vis_bool if vis_area >= 16 else m_bool,
            "dil_mask": dil_bool,
            "bbox": _get_mask_bbox(m_bool),
            "dil_bbox": _get_mask_bbox(dil_bool),
            "area": area,
            "color": color,
            "color_min": color.copy(),
            "color_max": color.copy(),
            "merged_count": 1,
        })

    # Keep index 0 (full-canvas background layer) intact; iteratively merge structural shapes 1..N-1
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(records):
            rec_i = records[i]
            j = i + 1
            merged_in_inner = False
            while j < len(records):
                rec_j = records[j]

                # 1. Layer stack contiguity (part A): same or adjacent structural layer
                if rec_j["layer_id"] - rec_i["layer_id"] > 1:
                    break

                # 2. Dynamic-range-aware per-channel color similarity
                if np.any(np.abs(rec_i["color"] - rec_j["color"]) > channel_thresholds):
                    j += 1
                    continue

                new_cmin = np.minimum(rec_i["color_min"], rec_j["color_min"])
                new_cmax = np.maximum(rec_i["color_max"], rec_j["color_max"])
                if np.any((new_cmax - new_cmin) > 1.35 * channel_thresholds):
                    j += 1
                    continue

                # 3. Spatial closeness or partial overlap
                if not _bboxes_overlap(rec_i["dil_bbox"], rec_j["dil_bbox"]):
                    j += 1
                    continue

                y0 = max(rec_i["dil_bbox"][0], rec_j["dil_bbox"][0])
                y1 = min(rec_i["dil_bbox"][1], rec_j["dil_bbox"][1])
                x0 = max(rec_i["dil_bbox"][2], rec_j["dil_bbox"][2])
                x1 = min(rec_i["dil_bbox"][3], rec_j["dil_bbox"][3])

                bridge_crop = rec_i["dil_mask"][y0:y1, x0:x1] & rec_j["dil_mask"][y0:y1, x0:x1]
                contact_px = int(np.sum(bridge_crop))
                min_req_px = min(min_contact_pixels, max(1, int(0.2 * min(rec_i["area"], rec_j["area"]))))
                if contact_px < min_req_px:
                    j += 1
                    continue

                # Local directional color derivative (low slope across interface between shape i and shape j)
                strip_i = rec_i["vis_mask"][y0:y1, x0:x1] & rec_j["dil_mask"][y0:y1, x0:x1]
                strip_j = rec_j["vis_mask"][y0:y1, x0:x1] & rec_i["dil_mask"][y0:y1, x0:x1]
                if int(np.sum(strip_i)) >= 8 and int(np.sum(strip_j)) >= 8:
                    target_crop = target_float[y0:y1, x0:x1]
                    local_ci = target_crop[strip_i].mean(axis=0)
                    local_cj = target_crop[strip_j].mean(axis=0)
                    if np.any(np.abs(local_ci - local_cj) > 1.25 * channel_thresholds):
                        j += 1
                        continue

                # 4. Layer stack contiguity (part B): no intermediate shape k in (i, j) occludes/separates them
                is_stack_contiguous = True
                if j > i + 1:
                    bi = rec_i["bbox"]
                    bj = rec_j["bbox"]
                    ub = (min(bi[0], bj[0]), max(bi[1], bj[1]), min(bi[2], bj[2]), max(bi[3], bj[3]))
                    uy0, uy1, ux0, ux1 = ub
                    combined_crop = rec_i["mask"][uy0:uy1, ux0:ux1] | rec_j["mask"][uy0:uy1, ux0:ux1]
                    for k in range(i + 1, j):
                        rec_k = records[k]
                        if not _bboxes_overlap(rec_k["bbox"], ub):
                            continue
                        inter_px = int(np.sum(rec_k["mask"][uy0:uy1, ux0:ux1] & combined_crop))
                        if inter_px > max_intervening_overlap:
                            is_stack_contiguous = False
                            break

                if not is_stack_contiguous:
                    j += 1
                    continue

                # Merge rec_j into rec_i with local bridging
                merged_mask = rec_i["mask"] | rec_j["mask"]
                merged_mask[y0:y1, x0:x1] |= bridge_crop
                crop_u8 = merged_mask[y0:y1, x0:x1].astype(np.uint8) * 255
                crop_u8 = cv2.morphologyEx(crop_u8, cv2.MORPH_CLOSE, close_kernel)
                merged_mask[y0:y1, x0:x1] = crop_u8 > 127
                merged_mask = _fill_small_holes(merged_mask, max_hole_area=64)

                merged_vis = rec_i["vis_mask"] | rec_j["vis_mask"]
                new_area = max(int(np.sum(merged_mask)), 1)
                if int(np.sum(merged_vis)) >= 16:
                    new_color = target_float[merged_vis].mean(axis=0).astype(np.float32)
                else:
                    total_w = rec_i["area"] + rec_j["area"]
                    new_color = ((rec_i["color"] * rec_i["area"] + rec_j["color"] * rec_j["area"]) / total_w).astype(np.float32)

                dil_u8 = cv2.dilate(merged_mask.astype(np.uint8) * 255, dilate_kernel)
                dil_bool = dil_u8 > 127

                rec_i["mask"] = merged_mask
                rec_i["vis_mask"] = merged_vis
                rec_i["dil_mask"] = dil_bool
                rec_i["bbox"] = _get_mask_bbox(merged_mask)
                rec_i["dil_bbox"] = _get_mask_bbox(dil_bool)
                rec_i["area"] = new_area
                rec_i["color"] = new_color
                rec_i["color_min"] = new_cmin
                rec_i["color_max"] = new_cmax
                rec_i["merged_count"] += rec_j["merged_count"]

                records.pop(j)
                merged_in_inner = True
                changed = True

            if not merged_in_inner:
                i += 1

    # Rebuild shapes, shape_groups, and layerd_struct_masks in exact 1-to-1 order
    new_shapes = []
    new_shape_groups = []
    new_layer_dict = {}

    for idx, rec in enumerate(records):
        lid = rec["layer_id"]
        if lid not in new_layer_dict:
            new_layer_dict[lid] = []

        if rec["merged_count"] == 1:
            orig_idx = rec["orig_idx"]
            path = shapes[orig_idx]
            orig_group = shape_groups[orig_idx]
            group = pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([idx]),
                fill_color=orig_group.fill_color.detach().clone(),
                stroke_color=orig_group.stroke_color.detach().clone(),
            )
            mask_u8 = flat_masks[orig_idx]
        else:
            mask_u8 = np.where(rec["mask"], 255, 0).astype(np.uint8)
            path = init_path_by_mask(mask_u8.copy(), epsilon=epsilon)
            fill_rgb = [float(np.clip(c, 0.0, 1.0)) for c in rec["color"]]
            group = pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([idx]),
                fill_color=torch.FloatTensor(fill_rgb + [1.0]),
                stroke_color=torch.FloatTensor([0.0, 0.0, 0.0, 1.0]),
            )

        new_shapes.append(path)
        new_shape_groups.append(group)
        new_layer_dict[lid].append(mask_u8)

    new_layerd_struct_masks = [new_layer_dict[k] for k in sorted(new_layer_dict.keys()) if len(new_layer_dict[k]) > 0]
    return new_shapes, new_shape_groups, new_layerd_struct_masks


def simplify_concave_shapes(
    device,
    shapes: list,
    shape_groups: list,
    target_img: np.ndarray,
    sse_threshold: float = 3.0,
):
    """Bottom-to-top pass removing concave angles/curves that do not degrade summed squared error against target_img."""
    img_height, img_width = target_img.shape[:2]
    target_tensor = (torch.tensor(target_img, device=device, dtype=torch.float32) / 255.0).permute(2, 0, 1)

    with torch.no_grad():
        curr_img = rgba_to_rgb(svg_to_img(img_width, img_height, shapes, shape_groups, device), device)
        curr_sse = torch.nn.functional.mse_loss(curr_img, target_tensor, reduction="sum").item()

        for i in tqdm(range(len(shapes)), desc="Concave simplification", unit="shape"):
            def _accept_best(candidates: List[pydiffvg.Path]) -> bool:
                nonlocal curr_sse
                best = None
                for cand_path in candidates:
                    shapes_cand = shapes[:i] + [cand_path] + shapes[i + 1 :]
                    cand_img = rgba_to_rgb(svg_to_img(img_width, img_height, shapes_cand, shape_groups, device), device)
                    cand_sse = torch.nn.functional.mse_loss(cand_img, target_tensor, reduction="sum").item()
                    if (cand_sse - curr_sse) > sse_threshold:
                        continue

                    if best is None or cand_sse < best[0]:
                        best = (cand_sse, cand_path)

                if best is None:
                    return False

                curr_sse, shapes[i] = best
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

                for k in _find_concave_segments(pts, orient):
                    if _accept_best([_straighten_segment(shapes[i], k)]):
                        changed = True
                        break

    return shapes, shape_groups