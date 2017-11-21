from skimage import io
import numpy as np
from skimage import morphology
from skimage import img_as_float
from skimage import filters
from skimage import restoration
from scipy import ndimage as ndi
from skimage import segmentation
from skimage import feature
from skimage import measure
from skimage import util

def roi(image):
    """Create an ROI for analysis """
    mask_roi = np.zeros(image.shape, dtype = bool)
    mask_roi[image>filters.threshold_otsu(image)] = 1
    roi_ilot = morphology.closing(mask_roi, morphology.ball(3))
    roi_ilot = morphology.closing(roi_ilot, morphology.ball(3))
    roi_ilot = morphology.dilation(roi_ilot, morphology.ball(3))
    roi_ilot = morphology.dilation(roi_ilot, morphology.ball(3))
    roi_ilot = morphology.remove_small_objects(roi_ilot, 100000)
    roi_ilot = morphology.remove_small_holes(roi_ilot, 100000)
    bbox = ndi.find_objects(roi_ilot)

    return bbox


def segment_nucleus(img_denoised):
    #find marker background
    mask_back = np.zeros(img_denoised.shape, dtype = bool)
    mask_back[img_denoised>filters.threshold_mean(img_denoised)*0.2] = 1
    #find marker foreground
    mask_forg = np.zeros(img_denoised.shape, dtype = bool)
    mask_forg[img_denoised>filters.threshold_minimum(img_denoised)*1.2] = 1

    marker_ws = np.zeros(img_denoised.shape, dtype = np.int)
    marker_ws[mask_back==0] = 1
    marker_ws[mask_forg==1] = 2

    segmented = segmentation.random_walker(img_denoised, marker_ws,
                                         beta = 2000, mode='cg_mg')

    labeled = measure.label(segmented)

    region = measure.regionprops(labeled)

    # remove small region
    small_region = []
    for prop in region:
        if prop.area < 10000:
            small_region.append(prop.label)

    cleaned = np.copy(labeled)
    for i in small_region:
        cleaned[np.where(cleaned == i)] = 1

    background = cleaned == 1
    background = morphology.erosion(background, morphology.ball(2))

    segmented_mask = np.zeros(img_denoised.shape, dtype = bool)
    segmented_mask[segmented==1] = True
    segmented_mask[segmented==2] = False

    return cleaned, segmented_mask ,background

def marker_ind_cells(actin_float_tvc, background):

    #Find markers individual cells

    mask_actin = np.zeros(actin_float_tvc.shape, dtype = bool)
    mask_actin[actin_float_tvc>filters.threshold_otsu(actin_float_tvc)] = 1
    mask_actin = morphology.erosion(~mask_actin, morphology.ball(3))
    mask_actin = morphology.dilation(mask_actin, morphology.ball(1))
    mask_actin[background] = 0

    # Limit plan in which markers can be found - pick image with highest variance
    variance = [np.var(image) for _, image in enumerate(mask_actin)]
    idx = np.argmax(variance)

    distance_a = ndi.distance_transform_edt(np.amax(mask_actin[idx-3:idx+3], axis = 0))
    distance_a = morphology.erosion(distance_a, morphology.disk(6))
    local_maxi = feature.peak_local_max(distance_a, indices=True,
                                        exclude_border=False,
                                        footprint=np.ones((30, 30)))

    markers_lm = np.zeros(distance_a.shape, dtype=np.int)
    markers_lm[local_maxi[:,0].astype(np.int),
               local_maxi[:,1].astype(np.int)] = np.arange(len(local_maxi[:,0])) + 1
    markers_lm = morphology.dilation(markers_lm, morphology.disk(7))
    markers_3D = np.zeros(actin_float_tvc.shape, dtype = np.int)
    markers_3D[idx] = markers_lm
    markers_3D = ndi.label(markers_3D)[0]

    return markers_3D, idx

def background_yap(yap_float_tvc):
    mask_yap = np.zeros(yap_float_tvc.shape, dtype = bool)
    mask_yap[yap_float_tvc>((filters.threshold_li(yap_float_tvc))*20/100)] = 1
    mask_yap = morphology.erosion(mask_yap, morphology.ball(3))
    mask_yap = morphology.dilation(mask_yap, morphology.ball(3))
    mask_yap = morphology.erosion(mask_yap, morphology.ball(3))

    return mask_yap

def segment_actin(actin_float_tvc, mask_yap, markers_3D):
    # Black tophat transformation (see https://en.wikipedia.org/wiki/Top-hat_transform)
    hat_actin = morphology.white_tophat(actin_float_tvc, morphology.cube(3))
    hat_actin = morphology.closing(hat_actin, morphology.ball(3))

    segmented = segmentation.watershed(hat_actin, markers_3D, compactness=0.001)
    segmented[~mask_yap] = 0

    return segmented

def regionprop_protein_in_mask(protein_roi, mask, segmented, inside = True):
    # find intensity image inside or outside mask (nucleus/cyto)

    protein = np.copy(protein_roi)
    if inside:
        protein[mask] = 0
    else:
        protein[~mask] = 0

    region_props = measure.regionprops(segmented, protein)

    return region_props

def outer_layer(mask_yap, segmented, idx):

    line_extern = morphology.erosion(
        mask_yap[idx],
        morphology.disk(5)) ^ morphology.dilation(mask_yap[idx],
        morphology.disk(1))
    line_extern = morphology.remove_small_objects(line_extern, 1000)

    mask_outside = np.copy(segmented[idx])
    mask_outside[~line_extern] = 0

    label_outside = measure.regionprops(mask_outside)

    label_outside_list = []
    for prop in label_outside:
        label_outside_list.append(prop.label)

    return label_outside_list

def list_mean_int_in_out(region_props, label_outside_list):

    mean_int_inside = []
    mean_int_outside = []

    for prop in region_props:
        if prop.label not in label_outside_list:
            int_inside = prop.intensity_image
            mean_int_inside.append(
                np.mean(int_inside[np.nonzero(int_inside)]))

        if prop.label in label_outside_list:
            int_outside = prop.intensity_image
            mean_int_outside.append(
                np.mean(int_outside[np.nonzero(int_outside)]))

    return mean_int_inside, mean_int_outside
