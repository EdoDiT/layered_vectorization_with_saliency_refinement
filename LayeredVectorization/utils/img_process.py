import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from PIL import Image
import os
import glob
from torchvision import transforms
import torch
import torch.nn.functional as F
import pydiffvg
from collections import Counter
from scipy.spatial.distance import cdist
from typing import Union, List, Optional
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

def sam_img_seq(device, simp_img_seq: list, masks_save_path: str = "-1", sam_conf: dict = None, split_connected_components: bool = True) -> list:
    def mask_preprocessing(masks):
        prepro_masks = []
        for mask in masks:
            filled_mask = filling_mask_holes(mask)
            if split_connected_components:
                num_labels, labels = cv2.connectedComponents(filled_mask)
                for i in range(1, num_labels):
                    single_region = np.zeros_like(mask)
                    single_region[labels == i] = 255
                    prepro_masks.append(single_region)
            else:
                if cv2.countNonZero(filled_mask) > 0:
                    prepro_masks.append(filled_mask)
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


def _simplify_contour_to_max_vertices(contour: np.ndarray, epsilon: float, max_vertices: int) -> np.ndarray:
    """Adaptively increase approxPolyDP epsilon until contour has at most max_vertices (>= 3)."""
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
    if len(simplified) <= max_vertices:
        return simplified

    perimeter = cv2.arcLength(contour, True)
    lo = float(epsilon)
    hi = max(perimeter, lo + 1.0)
    best = None

    for _ in range(20):
        mid = 0.5 * (lo + hi)
        cand = cv2.approxPolyDP(contour, mid, closed=True)
        n = len(cand)
        if n > max_vertices:
            lo = mid
        elif n >= 3:
            best = cand
            hi = mid
        else:
            hi = mid

    if best is not None:
        return best

    # Fallback: oriented bounding box (4 clean vertices hugging the silhouette)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.reshape(-1, 1, 2).astype(np.float32)


def init_path_by_mask(
    mask: Union[np.ndarray, torch.Tensor],
    epsilon: int = 5,
    max_init_segments: Optional[int] = None,
    use_convex_hull: bool = False,
):
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

    if use_convex_hull:
        all_pts = np.vstack(contours)
        contour = cv2.convexHull(all_pts)
    else:
        while len(contours)>1:
            point_1, point_2 = find_closest_contours(contours)
            cv2.line(mask, point_1, point_2, (255), 3)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = contours[0]

    if max_init_segments is not None and max_init_segments >= 3:
        simplified_contour = _simplify_contour_to_max_vertices(contour, epsilon, max_init_segments)
    else:
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

def init_svg_by_mask(
    layerd_masks: list,
    target_img: np.ndarray,
    epsilon: int = 5,
    max_init_segments: Optional[int] = None,
    use_convex_hull: bool = False,
    color_masks: Optional[list] = None,
):
    shapes = []
    shape_groups = []
    for layer_i, masks in enumerate(layerd_masks):
        for mask_j, mask in enumerate(masks):
            path = init_path_by_mask(
                mask,
                epsilon=epsilon,
                max_init_segments=max_init_segments,
                use_convex_hull=use_convex_hull,
            )
            cmask = color_masks[layer_i][mask_j] if color_masks is not None else mask
            mean_color = get_mean_color(target_img, cmask)
            path_group = pydiffvg.ShapeGroup(
                                    shape_ids=torch.LongTensor([len(shapes)]),
                                    fill_color=torch.tensor(list(mean_color)+[255])/255,
                                    stroke_color=torch.FloatTensor([0,0,0,1])
                                )
            shapes.append(path)
            shape_groups.append(path_group)
    return shapes,shape_groups


def color_fitting(shape_groups, target_img: np.ndarray, layerd_struct_masks: list, is_cluster: bool, k: int = 30):
    def combine_masks(masks_list):
        combined = np.zeros_like(masks_list[0])
        for m in masks_list:
            combined = cv2.bitwise_or(combined, m)
        return combined
    if is_cluster:
        data = target_img.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        target_img = segmented_data.reshape((target_img.shape))

    most_common_color_list = []
    for i, masks in enumerate(layerd_struct_masks):
        if i <= len(layerd_struct_masks) - 2:
            above_masks = [m for higher_layer in layerd_struct_masks[i + 1 :] for m in higher_layer]
            masks_combine = combine_masks(above_masks)
        else:
            masks_combine = np.zeros_like(masks[0])
        for j, mask in enumerate(masks):
            visible = (mask > 0) & (masks_combine == 0)
            colors = target_img[visible]
            color_tuples = [tuple(color) for color in colors]
            color_counts = Counter(color_tuples)
            if color_counts:
                most_common_color = color_counts.most_common(1)[0][0]
            else:
                # Fallback to full mask if completely occluded
                full_colors = target_img[mask > 0]
                if len(full_colors) > 0:
                    most_common_color = Counter([tuple(c) for c in full_colors]).most_common(1)[0][0]
                else:
                    most_common_color = (0, 0, 0)
            most_common_color_list.append(list(most_common_color))
    for i, shape_group in enumerate(shape_groups):
        shape_group.fill_color = torch.FloatTensor(most_common_color_list[i] + [255]) / 255
    return shape_groups, target_img

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


def compute_depth_map(
    device: torch.device,
    target_img: np.ndarray,
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
) -> np.ndarray:
    """Compute a normalized relative depth/disparity map in [0, 1] (higher = closer)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    pil_img = Image.fromarray(target_img.astype(np.uint8))
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    h, w = target_img.shape[:2]
    depth = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    del model, processor, inputs, outputs, predicted_depth
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros((h, w), dtype=np.float32)
    return depth.astype(np.float32)


def order_masks_by_depth(layerd_struct_masks: list, depth_map: np.ndarray) -> list:
    """Flatten deduplicated masks into a strict far->near z-order using median depth,
    while keeping the base canvas mask at index 0 and inner/contained masks above parents."""
    flat_masks = [m for layer in layerd_struct_masks for m in layer]
    if len(flat_masks) <= 2:
        return [[m] for m in flat_masks]

    base_mask = flat_masks[0]
    rest = flat_masks[1:]

    items = []
    for m in rest:
        fg = m > 0
        area = int(np.sum(fg))
        med_depth = float(np.median(depth_map[fg])) if area > 0 else 0.0
        items.append({"mask": m, "fg": fg, "area": area, "depth": med_depth})

    # Far -> near (ascending disparity), ties broken by larger area first
    items.sort(key=lambda x: (x["depth"], -x["area"]))

    # Enforce strict containment: if inner mask B is >=85% inside larger mask A, B must come after A
    n = len(items)
    changed = True
    max_passes = n
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1
        for i in range(n):
            for j in range(i + 1, n):
                if items[i]["area"] < items[j]["area"]:
                    inter = int(np.sum(items[i]["fg"] & items[j]["fg"]))
                    if items[i]["area"] > 0 and (inter / items[i]["area"]) >= 0.85:
                        # Move inner mask i right after enclosing mask j
                        inner = items.pop(i)
                        items.insert(j, inner)
                        changed = True
                        break
            if changed:
                break

    ordered = [base_mask] + [it["mask"] for it in items]
    return [[m] for m in ordered]


def refine_layers_with_depth_and_fill(
    layerd_struct_masks: list,
    depth_map: np.ndarray,
    target_img: np.ndarray,
    depth_far_k_px: int = 64,
    depth_erode_px: int = 4,
    depth_margin: float = 0.08,
    depth_touch_px: int = 5,
    fill_concavities: bool = True,
    fill_safety_erode_px: int = 3,
):
    """Build multi-shape layers using pairwise depth occlusion between touching masks,
    and fill higher-layer occluder notches inside each lower mask's convex hull.

    Returns `(raw_layered, filled_layered, debug_bgr)`:
    - `raw_layered`: masks grouped into multi-shape layers (for color fitting).
    - `filled_layered`: same structure with occluded notches filled (for path init & struct loss).
    - `debug_bgr`: side-by-side visualization of layer indices (left) and filled notches (right).
    """
    base_mask = layerd_struct_masks[0][0]
    masks_list = [m for layer in layerd_struct_masks[1:] for m in layer]
    m_count = len(masks_list)
    h, w = base_mask.shape[:2]

    if m_count == 0:
        raw_layered = [[base_mask]]
        filled_layered = [[base_mask.copy()]]
        debug_bgr = np.zeros((h, w * 2, 3), dtype=np.uint8)
        return raw_layered, filled_layered, debug_bgr

    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * depth_erode_px + 1, 2 * depth_erode_px + 1))
    small_erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    touch_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * depth_touch_px + 1, 2 * depth_touch_px + 1))

    fgs = []
    areas = []
    d_fars = []
    dils = []

    for m in masks_list:
        m_u8 = (m > 0).astype(np.uint8) * 255
        fg = m_u8 > 0
        area = int(fg.sum())
        core = cv2.erode(m_u8, erode_k) > 0 if depth_erode_px > 0 else fg
        if int(core.sum()) < depth_far_k_px:
            core = cv2.erode(m_u8, small_erode_k) > 0
        if int(core.sum()) < 8:
            core = fg
        if area > 0:
            sorted_d = np.sort(depth_map[core])
            k_take = min(max(1, depth_far_k_px), len(sorted_d))
            d_val = float(sorted_d[:k_take].mean())
        else:
            d_val = 0.0
        dil = cv2.dilate(m_u8, touch_k) > 0 if depth_touch_px > 0 else fg

        fgs.append(fg)
        areas.append(area)
        d_fars.append(d_val)
        dils.append(dil)

    # Directed graph: edge u -> v means u must sit in a strictly lower layer than v
    adj = [set() for _ in range(m_count)]

    def _can_reach(src: int, dst: int) -> bool:
        stack = [src]
        seen = {src}
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    # 1. Hard containment edges (smaller mask >=50% inside larger mask goes above it)
    for i in range(m_count):
        for j in range(i + 1, m_count):
            if areas[i] == 0 or areas[j] == 0:
                continue
            inter = int(np.sum(fgs[i] & fgs[j]))
            if inter == 0:
                continue
            if areas[j] < areas[i] and (inter / areas[j]) >= 0.50:
                if not _can_reach(j, i):
                    adj[i].add(j)
            elif areas[i] < areas[j] and (inter / areas[i]) >= 0.50:
                if not _can_reach(i, j):
                    adj[j].add(i)

    # 2. Pairwise depth occlusion edges between touching/overlapping masks (strongest |delta_d| first)
    depth_candidates = []
    for i in range(m_count):
        for j in range(i + 1, m_count):
            if (j in adj[i]) or (i in adj[j]):
                continue
            if not (np.any(dils[i] & fgs[j]) or np.any(dils[j] & fgs[i])):
                continue
            diff = d_fars[j] - d_fars[i]
            if abs(diff) >= depth_margin:
                if diff > 0:
                    depth_candidates.append((diff, i, j))
                else:
                    depth_candidates.append((-diff, j, i))

    depth_candidates.sort(key=lambda x: -x[0])
    added_depth_edges = 0
    for _, u, v in depth_candidates:
        if not _can_reach(v, u):
            adj[u].add(v)
            added_depth_edges += 1

    # 3. Minimum-layer assignment on DAG (disjoint / uncomparable masks share the same layer)
    indeg = [0] * m_count
    preds = [[] for _ in range(m_count)]
    for u in range(m_count):
        for v in adj[u]:
            indeg[v] += 1
            preds[v].append(u)

    queue = [i for i in range(m_count) if indeg[i] == 0]
    layer_id = [1] * m_count
    topo = []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        if preds[u]:
            layer_id[u] = max(layer_id[p] + 1 for p in preds[u])
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    max_layer = max(layer_id)
    raw_layered = [[base_mask]]
    for lid in range(1, max_layer + 1):
        members = [i for i in range(m_count) if layer_id[i] == lid]
        members.sort(key=lambda idx: -areas[idx])
        if members:
            raw_layered.append([masks_list[i] for i in members])

    # 4. Fill higher-layer occluder notches inside each lower mask's convex hull
    num_layers = len(raw_layered)
    filled_layered = [[m.copy() for m in layer] for layer in raw_layered]
    total_filled_masks = 0
    total_filled_px = 0

    if fill_concavities and num_layers > 2:
        higher_unions = [np.zeros((h, w), dtype=bool) for _ in range(num_layers)]
        running_union = np.zeros((h, w), dtype=bool)
        for lid in range(num_layers - 1, 0, -1):
            higher_unions[lid] = running_union.copy()
            for m in raw_layered[lid]:
                running_union |= m > 0

        safe_k = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fill_safety_erode_px + 1, 2 * fill_safety_erode_px + 1))
            if fill_safety_erode_px > 0
            else None
        )
        sep_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for lid in range(1, num_layers - 1):
            h_union = higher_unions[lid]
            if not np.any(h_union):
                continue
            layer_masks = raw_layered[lid]
            layer_fgs = [m > 0 for m in layer_masks]

            for idx, m in enumerate(layer_masks):
                m_fg = layer_fgs[idx]
                m_u8 = m_fg.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(cnts) == 0:
                    continue
                hull = cv2.convexHull(np.vstack(cnts))
                hull_u8 = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(hull_u8, [hull], -1, 255, thickness=cv2.FILLED)
                hull_fg = hull_u8 > 0

                allowed_u8 = (m_fg | h_union).astype(np.uint8) * 255
                safe_fg = (cv2.erode(allowed_u8, safe_k) > 0) if safe_k is not None else (m_fg | h_union)

                if len(layer_fgs) > 1:
                    other_fg = np.zeros((h, w), dtype=bool)
                    for k_idx, k_fg in enumerate(layer_fgs):
                        if k_idx != idx:
                            other_fg |= k_fg
                    other_dil = cv2.dilate(other_fg.astype(np.uint8) * 255, sep_k) > 0
                else:
                    other_dil = np.zeros((h, w), dtype=bool)

                cand_fg = m_fg | (hull_fg & safe_fg & (~other_dil))
                added_px = int(np.sum(cand_fg & (~m_fg)))
                if added_px <= 0:
                    continue

                # Keep only connected components touching the original mask
                num_cc, cc_labels = cv2.connectedComponents(cand_fg.astype(np.uint8) * 255)
                touching_labels = set(np.unique(cc_labels[m_fg])) - {0}
                clean_fg = np.isin(cc_labels, list(touching_labels))
                clean_u8 = filling_mask_holes(clean_fg.astype(np.uint8) * 255)

                real_added = int(np.sum((clean_u8 > 0) & (~m_fg)))
                if real_added > 0:
                    filled_layered[lid][idx] = clean_u8
                    total_filled_masks += 1
                    total_filled_px += real_added

    print(
        f"[depth-occlusion] {num_layers} layers (sizes {[len(l) for l in raw_layered]}), "
        f"{added_depth_edges} depth edges, filled {total_filled_masks} masks (+{total_filled_px} px)"
    )

    # 5. Build debug image (BGR for cv2.imwrite): left = layer map, right = notch fills in magenta
    gray = cv2.cvtColor(target_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    layer_idx_map = np.zeros((h, w), dtype=np.float32)
    for lid, layer in enumerate(raw_layered):
        for m in layer:
            layer_idx_map[m > 0] = float(lid) / max(num_layers - 1, 1)
    cmap_bgr = cv2.applyColorMap((layer_idx_map * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cmap_rgb = cv2.cvtColor(cmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    left_rgb = np.clip(0.30 * gray_rgb + 0.70 * cmap_rgb, 0, 255).astype(np.uint8)

    right_rgb = np.clip(target_img.astype(np.float32) * 0.75, 0, 255).astype(np.uint8)
    added_any = np.zeros((h, w), dtype=bool)
    for lid in range(1, num_layers):
        for r_m, f_m in zip(raw_layered[lid], filled_layered[lid]):
            diff_fg = (f_m > 0) & (~(r_m > 0))
            if np.any(diff_fg):
                added_any |= diff_fg
                cnts, _ = cv2.findContours((f_m > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(right_rgb, cnts, -1, (0, 255, 255), 1)
    right_rgb[added_any] = np.array([255, 0, 200], dtype=np.uint8)

    debug_rgb = np.concatenate([left_rgb, right_rgb], axis=1)
    debug_bgr = cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
    return raw_layered, filled_layered, debug_bgr


def init_composite_struct_target(ordered_masks: list, device: torch.device):
    """Build one bottom-to-top random-color composite target, matching ShapeGroups,
    and per-shape visible_masks (1 where not occluded by any higher mask j > i)."""
    h, w = ordered_masks[0].shape[:2]
    composite = torch.zeros((3, h, w), dtype=torch.float32)
    struct_shape_groups = []

    for i, mask in enumerate(ordered_masks):
        color = [float(0.2 + 0.8 * torch.rand(1).item()) for _ in range(3)]
        fg = torch.from_numpy((mask > 0).astype(np.float32))
        for c in range(3):
            composite[c] = composite[c] * (1.0 - fg) + color[c] * fg
        struct_shape_groups.append(
            pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([i]),
                fill_color=torch.FloatTensor(color + [1.0]),
                stroke_color=torch.FloatTensor([0.0, 0.0, 0.0, 1.0]),
            )
        )

    # Reverse cumulative union of higher masks -> visible_mask[i] = 1 - union(j > i)
    visible_masks = [None] * len(ordered_masks)
    occluded_above = np.zeros((h, w), dtype=bool)
    for i in range(len(ordered_masks) - 1, -1, -1):
        visible_masks[i] = torch.from_numpy((~occluded_above).astype(np.float32)).to(device)
        occluded_above |= ordered_masks[i] > 0

    return composite.to(device), struct_shape_groups, visible_masks


def compute_geometric_loss(
    shapes: list,
    is_opt_list: Optional[List[int]] = None,
    min_seg_ratio: float = 0.25,
    short_seg_weight: float = 0.01,
    handle_asym_weight: float = 0.005,
    smooth_cos_thresh: float = -0.5,
    elastica_weight: float = 0.0,
    elastica_length_weight: float = 0.2,
    elastica_curv_weight: float = 1.0,
):
    """Scale-proportional geometric regularization for closed cubic Bezier paths.

    1. Penalizes segments whose anchor-to-anchor length drops below
       `min_seg_ratio * mean_seg_len` (proportional to each shape's own size).
    2. Penalizes asymmetric incoming/outgoing handle lengths on smooth nodes.
    3. Optional scale-normalized Euler's Elastica smoothness (length + curvature^2)
       so occluded/bridged spans relax smoothly.
    """
    short_losses = []
    handle_losses = []
    elastica_losses = []
    t_samples = None

    for i, path in enumerate(shapes):
        if is_opt_list is not None and len(is_opt_list) > 0 and is_opt_list[i] == 0:
            continue
        pts = path.points
        if not pts.requires_grad:
            continue
        if pts.shape[0] < 9 or pts.shape[0] % 3 != 0:
            continue
        if not torch.all(path.num_control_points == 2):
            continue

        anchors = pts[0::3]  # (N, 2)
        next_anchors = torch.roll(anchors, shifts=-1, dims=0)
        seg_vecs = next_anchors - anchors
        seg_lens = torch.sqrt((seg_vecs ** 2).sum(dim=-1) + 1e-8)  # (N,)

        mean_seg_len = seg_lens.mean().detach().clamp(min=1e-3)
        min_len = min_seg_ratio * mean_seg_len
        short_pen = (F.relu(min_len - seg_lens) / mean_seg_len).pow(2).mean()
        short_losses.append(short_pen)

        h_out = pts[1::3] - anchors
        h_in = torch.roll(pts[2::3], shifts=1, dims=0) - anchors
        len_out = torch.sqrt((h_out ** 2).sum(dim=-1) + 1e-8)
        len_in = torch.sqrt((h_in ** 2).sum(dim=-1) + 1e-8)

        cos_angle = ((h_in * h_out).sum(dim=-1) / (len_in * len_out + 1e-8)).clamp(-1.0, 1.0)
        # 1.0 when handles are collinear & opposite (cos == -1), 0.0 at sharp corners
        denom = max(1.0 + smooth_cos_thresh, 1e-6)
        smooth_weight = (F.relu(-cos_angle + smooth_cos_thresh) / denom).detach()
        handle_pen = (smooth_weight * ((len_out - len_in) / mean_seg_len).pow(2)).mean()
        handle_losses.append(handle_pen)

        if elastica_weight > 0.0:
            if t_samples is None or t_samples.device != pts.device or t_samples.dtype != pts.dtype:
                t_samples = torch.linspace(0.1, 0.9, 8, device=pts.device, dtype=pts.dtype).view(1, -1, 1)
            omt = 1.0 - t_samples
            p0 = anchors.unsqueeze(1)
            p1 = pts[1::3].unsqueeze(1)
            p2 = pts[2::3].unsqueeze(1)
            p3 = next_anchors.unsqueeze(1)

            d1 = 3.0 * (omt ** 2) * (p1 - p0) + 6.0 * omt * t_samples * (p2 - p1) + 3.0 * (t_samples ** 2) * (p3 - p2)
            d2 = 6.0 * omt * (p2 - 2.0 * p1 + p0) + 6.0 * t_samples * (p3 - 2.0 * p2 + p1)

            d1_n = d1 / mean_seg_len
            d2_n = d2 / mean_seg_len
            speed_sq = (d1_n ** 2).sum(dim=-1) + 1e-4
            speed = torch.sqrt(speed_sq)
            cross = torch.abs(d1_n[..., 0] * d2_n[..., 1] - d1_n[..., 1] * d2_n[..., 0])
            kappa = (cross / (speed_sq * speed)).clamp(max=10.0)
            elastica_pen = ((elastica_length_weight + elastica_curv_weight * (kappa ** 2)) * speed).mean()
            elastica_losses.append(elastica_pen)

    if len(short_losses) == 0:
        return 0.0

    total = short_seg_weight * torch.stack(short_losses).mean() + handle_asym_weight * torch.stack(handle_losses).mean()
    if elastica_weight > 0.0 and len(elastica_losses) > 0:
        total = total + elastica_weight * torch.stack(elastica_losses).mean()
    return total


def subdivide_bezier_segment(
    path: pydiffvg.Path,
    seg_idx: int,
    snap_contour: Optional[np.ndarray] = None,
    snap_radius: float = 0.0,
) -> pydiffvg.Path:
    """Midpoint (t=0.5) De Casteljau split of one cubic segment in a closed path.

    The split itself preserves the curve exactly. If `snap_contour` (M, 2) is given, the new anchor
    (together with its two handles) is then translated onto the closest contour point, provided it is
    within `snap_radius` pixels, so the new control point starts on the silhouette instead of on the chord."""
    pts = path.points.detach()
    num_segs = pts.shape[0] // 3
    k = seg_idx % num_segs
    k_next = (k + 1) % num_segs

    p0 = pts[3 * k]
    p1 = pts[3 * k + 1]
    p2 = pts[3 * k + 2]
    p3 = pts[3 * k_next]

    q0 = 0.5 * (p0 + p1)
    q1 = 0.5 * (p1 + p2)
    q2 = 0.5 * (p2 + p3)
    r0 = 0.5 * (q0 + q1)
    r1 = 0.5 * (q1 + q2)
    mid = 0.5 * (r0 + r1)

    split_block = torch.stack([p0, q0, r0, mid, r1, q2], dim=0)
    new_pts = torch.cat([pts[: 3 * k], split_block, pts[3 * k + 3 :]], dim=0).clone()

    if snap_contour is not None and snap_radius > 0 and len(snap_contour) > 0:
        mid_idx = 3 * k + 3  # new anchor; handles sit at mid_idx - 1 and mid_idx + 1
        mid_np = new_pts[mid_idx].cpu().numpy()
        dist_sq = ((snap_contour - mid_np) ** 2).sum(axis=1)
        nearest = int(np.argmin(dist_sq))
        if dist_sq[nearest] <= snap_radius ** 2:
            delta = torch.from_numpy((snap_contour[nearest] - mid_np).astype(np.float32))
            new_pts[mid_idx - 1 : mid_idx + 2] += delta.to(device=new_pts.device, dtype=new_pts.dtype)

    new_path = pydiffvg.Path(
        num_control_points=torch.LongTensor([2] * (num_segs + 1)),
        points=new_pts,
        stroke_width=path.stroke_width.detach().clone(),
        is_closed=path.is_closed,
    )
    return new_path


@torch.no_grad()
def compute_segment_errors(
    path: pydiffvg.Path,
    err_map: torch.Tensor,
    num_samples: int = 9,
    normal_offsets: tuple = (0.0, -2.0, 2.0, -5.0, 5.0),
    visible_mask: Optional[torch.Tensor] = None,
):
    """Mean residual error of each cubic segment (tensor of shape (num_segs,)), obtained by sampling
    err_map along the curve and a small band around it. Returns None for degenerate paths.

    Reference values for the layer error maps (random colors in [0.2, 1] on black): a mismatched pixel
    costs ~0.4, and a perfectly fitted boundary still reads ~0.01-0.02 (antialiasing)."""
    pts = path.points.detach().to(err_map.device)
    num_segs = pts.shape[0] // 3
    if num_segs == 0 or pts.shape[0] % 3 != 0:
        return None

    eval_err = err_map * visible_mask if visible_mask is not None else err_map

    h, w = eval_err.shape
    t = torch.linspace(0.1, 0.9, num_samples, device=eval_err.device, dtype=pts.dtype).view(1, -1, 1)
    omt = 1.0 - t

    idx = torch.arange(num_segs, device=eval_err.device)
    p0 = pts[3 * idx].unsqueeze(1)
    p1 = pts[3 * idx + 1].unsqueeze(1)
    p2 = pts[3 * idx + 2].unsqueeze(1)
    p3 = pts[3 * ((idx + 1) % num_segs)].unsqueeze(1)

    # Cubic Bezier points B(t) and tangents B'(t): shape (N, S, 2)
    curve = (omt ** 3) * p0 + 3 * (omt ** 2) * t * p1 + 3 * omt * (t ** 2) * p2 + (t ** 3) * p3
    tangent = 3 * (omt ** 2) * (p1 - p0) + 6 * omt * t * (p2 - p1) + 3 * (t ** 2) * (p3 - p2)
    tang_norm = torch.sqrt((tangent ** 2).sum(dim=-1, keepdim=True) + 1e-8)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1) / tang_norm

    offsets = torch.tensor(normal_offsets, device=eval_err.device, dtype=pts.dtype).view(1, 1, -1, 1)
    sample_pts = curve.unsqueeze(2) + offsets * normal.unsqueeze(2)  # (N, S, O, 2)

    xs = sample_pts[..., 0].round().long().clamp(0, w - 1)
    ys = sample_pts[..., 1].round().long().clamp(0, h - 1)
    seg_errors = eval_err[ys, xs].mean(dim=(1, 2))  # (N,)

    return seg_errors