import tifffile as tif

import gc
import pandas as pd
import numpy as np
import cv2

from shapely import affinity
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from collections import defaultdict

import matplotlib.pyplot as plt
import global_vars
import os

def shw(im, y=6.0, x=6.0):
    plt.figure(figsize=(x,y))
    plt.imshow(im)
    plt.show()

# make paths relative etc
def load_train_names():
    train_labels = pd.read_csv(os.path.join(global_vars.DATA_DIR,'train_wkt_v4.csv'), index_col=0)
    return list(train_labels.index.unique())

def load_rgb(im_id):
    return tif.imread(os.path.join(global_vars.DATA_DIR,'three_band', im_id + '.tif')).transpose((1, 2, 0))

def load_m(im_id):
    return tif.imread(os.path.join(global_vars.DATA_DIR,'sixteen_band', im_id + '_M.tif')).transpose((1, 2, 0))

def load_labels(name, size, class_num):
    return tif.imread(os.path.join(global_vars.DATA_DIR, 'labels',
                                   name + '_' + str(size) +
                                   '_class_' + str(class_num) + '.tif'))


def load_all_lab(name, size):
    for i in range(1, 11):
        if i == 1:
            im = load_labels(name, size, i)
            ims = np.zeros(list(im.shape[:2]) + [10])
            ims[:, :, i - 1] = im[:, :, 0]
        else:
            ims[:, :, i - 1] = load_labels(name, size, i)[:, :, 0]

    return ims


def poly_label(image_id, class_type):
    train_wkt = pd.read_csv(os.path.join(global_vars.DATA_DIR,'train_wkt_v4.csv'))
    return wkt.loads(train_wkt.query('ClassType==@class_type and ImageId == @image_id').iloc[0, 2])

# generate wkt's
def make_poly(name, mask, epsilon, min_area):
    """creates a set of polygons from an image mask and scales it to submission coordinates"""
    shp = mask.shape
    mpoly = mask_to_polygons(mask, epsilon=epsilon, min_area=min_area)
    scaled = scale_multipolygon(name, mpoly, shp[0], shp[1])
    return scaled

def make_wkt(name, poly_lab_num, mask, epsilon, min_area):
    shp = mask.shape

    scaled = make_poly(name, mask, epsilon, min_area)

    wkt_tmp = wkt.dumps(scaled)
    pred = [name, poly_lab_num, wkt_tmp]
    return pred


def simple_to_wkt(threshed_masks, im_name, class_nums):
    all_preds = list()
    for e, c_num in enumerate(class_nums):
        mask = threshed_masks[:, :, e]

        mask = cv2.resize(mask.astype(np.uint8), (mask.shape[1] * 4, mask.shape[0] * 4), interpolation=0)
        tmp_wkt = make_wkt(im_name, c_num, mask, 1, 1)
        all_preds.append(tmp_wkt)
        gc.collect()

    all_preds = pd.DataFrame(all_preds, columns=['ImageId', 'ClassType', 'MultipolygonWKT'])
    all_preds = all_preds.replace(['POLYGON EMPTY', 'GEOMETRYCOLLECTION EMPTY'],
                                  ['MULTIPOLYGON EMPTY','MULTIPOLYGON EMPTY'])
    return all_preds

def make_masks(shp, im_id, label_nums):
    for e, num in enumerate(label_nums):
        if e == 0:
            mask = make_mask(shp, im_id, num)
            masks = mask.reshape(list(mask.shape) + [1])
        else:
            mask = make_mask(shp, im_id, num)
            mask = mask.reshape(list(mask.shape) + [1])
            masks = np.concatenate((masks, mask), axis=2)

    return masks


def make_mask(shp, im_id, label_num):
    grid_sizes = pd.read_csv(os.path.join(global_vars.DATA_DIR,'grid_sizes.csv'), index_col=0)
    train_wkt = pd.read_csv(os.path.join(global_vars.DATA_DIR,'train_wkt_v4.csv'), index_col=0)
    polys = get_polygon_list(train_wkt, im_id, label_num)
    x_max = grid_sizes.loc[im_id, 'Xmax']
    y_min = grid_sizes.loc[im_id, 'Ymin']
    plist, ilist = get_and_convert_contours(polys, shp, x_max, y_min)
    return plot_mask(shp, plist, ilist)


def find_thresh(masks, labels, lab_pos):
    thresh = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7, 75, 0.8]
    best_iou = 0
    best_thresh = -1
    for i in thresh:
        for e, j in enumerate(masks):
            if e == 0:
                tmp = j[:, :, lab_pos] > i
                tmp = tmp.reshape((tmp.shape[0] * tmp.shape[1], -1))

                lab = labels[e][:, :, lab_pos]
                lab = lab.reshape((tmp.shape[0] * tmp.shape[1], -1))

            else:
                t = (j[:, :, lab_pos] > i)
                t = t.reshape(t.shape[0] * t.shape[1], -1)
                tmp = np.concatenate((tmp, t), axis=0)

                l = labels[e][:, :, lab_pos]
                l = l.reshape(t.shape[0] * t.shape[1], -1)
                lab = np.concatenate((lab, l), axis=0) 

        total_iou = jaccard(lab, tmp)

        if total_iou > best_iou:
            best_thresh = i
            best_iou = total_iou

    return best_thresh


def scale_multipolygon(im_name, mpoly, height, width):
    """scales a multipolgon using the gridsizes provided by kaggle admins"""
    grid_sizes = pd.read_csv(os.path.join(global_vars.DATA_DIR,'grid_sizes.csv'), index_col=0)

    x_max = grid_sizes.loc[im_name, 'Xmax']
    y_min = grid_sizes.loc[im_name, 'Ymin']

    return affinity.scale(mpoly, xfact=x_max / width, yfact=y_min / height, origin=(0, 0))



def jaccard_poly(pred_mpoly, lab_mpoly):
    """intersection over union for two shapely multipolygons
    Thanks to shawn for posting this code on kaggle forums"""
    tp = pred_mpoly.intersection(lab_mpoly).area
    fp = pred_mpoly.area - tp
    fn = lab_mpoly.area - tp
    if fn + tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp + fn)


def jaccard(labels, preds):
    tp = (labels.flatten().astype(np.int) * preds.flatten().astype(np.int)).sum()
    fp = preds.sum() - tp
    fn = labels.sum() - tp

    if (tp + fp + fn) == 0:
        return 0
    else:
        return tp / (tp + fp + fn)


def augment_ims(x_train, y_train):
    for i in range(x_train.shape[0]):
        transp = np.random.randint(2)
        rotation = np.random.randint(4)
        x_tmp = x_train[i, :, :, :]
        y_tmp = y_train[i, :, :, :]
        if transp == 1:
            x_train[i, :, :, :] = np.rot90(x_tmp.transpose((1, 0, 2)), k=rotation)
            y_train[i, :, :, :] = np.rot90(y_tmp.transpose((1, 0, 2)), k=rotation)
        else:
            x_train[i, :, :, :] = np.rot90(x_tmp, k=rotation)
            y_train[i, :, :, :] = np.rot90(y_tmp, k=rotation)
    return x_train, y_train


def get_train_patches(im, lab_im, num_sample, label_edge, buff):
    ptch = list()
    labs = list()
    im_shp = im.shape
    for i in range(num_sample):
        uly = np.random.randint(buff, im_shp[0]-buff-label_edge)
        ulx = np.random.randint(buff, im_shp[1]-buff-label_edge)
        
        labs.append(lab_im[uly:uly+label_edge, ulx:ulx+label_edge,:])
        ptch.append(im[uly-buff:uly + label_edge + buff, ulx-buff:ulx + label_edge + buff, :])
    return ptch, labs

def print_scores(preds, y_test, lab_names, include_background=True):
    all_ious = list()
    if include_background:
        const=0
    else:
        const=1
    
    for e,i in enumerate(lab_names):
        preds1 = preds[:, :, :, e+const].flatten().astype(np.float)
        labs1 = y_test[:, :, :, e+const].flatten().astype(np.int)
        
        
        print('\n\nscores for class ' + str(i))
        print('iou for 0.4 thresh val images ',jaccard(labs1, (preds1 > 0.4)))

        iou_50 =  jaccard(labs1, (preds1 > 0.5))
        all_ious.append(iou_50)

        print('iou for 0.5 thresh val images ', iou_50)
        print('iou for 0.6 thresh val images ',jaccard(labs1, (preds1 > 0.6)))

    return all_ious
    

### all code below here is adapted from code provided by helpful people on kaggle.
### I have made slight modifacations, so any butchered code is likely mine;)

# https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
def convert_coordinates_to_raster(coords, img_size, x_max, y_min):
    # __author__ visoft

    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / x_max
    yf = H1 / y_min
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def get_polygon_list(train_wkt, image_id, class_type):
    # __author__ visoft

    df_image = train_wkt.loc[image_id, :]
    multipoly_def = df_image[df_image.ClassType == class_type].MultipolygonWKT
    polygon_list = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygon_list = wkt.loads(multipoly_def.values[0])
    return polygon_list


def get_and_convert_contours(polygon_list, img_size, x_max, y_min):
    # __author__ visoft

    perim_list = []
    interior_list = []
    if polygon_list is None:
        return None
    for poly in polygon_list:
        perim = np.array(list(poly.exterior.coords))
        perim_c = convert_coordinates_to_raster(perim, img_size, x_max, y_min)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = convert_coordinates_to_raster(interior, img_size, x_max, y_min)
            interior_list.append(interior_c)
    return perim_list, interior_list


def plot_mask(img_size, perim_list, interior_list, class_value=1):
    # __author__ visoft

    img_mask = np.zeros(img_size, np.uint8)
    if perim_list is None or interior_list is None:
        return img_mask
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def mask_to_polygons(mask, epsilon=1, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]

    if not approx_contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # need to re add in the check for type of all_polygons
        all_polygons = MultiPolygon(all_polygons)
    return all_polygons

# https://www.kaggle.com/chatcat/dstl-satellite-imagery-feature-detection/load-a-3-band-tif-image-and-overlay-dirt-tracks
def scl_prc(matrix):
    # __author__ = Alan Schoen
    """Fixes the pixel value range to 2%-98% original distribution of values"""
    orig_shape = matrix.shape
    matrix = np.reshape(matrix, [matrix.shape[0] * matrix.shape[1], 3]).astype(float)

    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 2, axis=0)
    maxs = np.percentile(matrix, 98, axis=0) - mins

    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, orig_shape)
    matrix = matrix.clip(0, 1)
    return matrix
