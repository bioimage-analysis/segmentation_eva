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
from sklearn.cluster import KMeans
from scipy.stats import norm
import scipy.ndimage as ndi
from scipy.signal import fftconvolve


def roi(image):
    """Create an ROI for analysis """
    mask_roi = np.zeros(image.shape, dtype = bool)
    mask_roi[image>filters.threshold_otsu(image)*0.1] = 1
    roi_ilot = morphology.remove_small_objects(mask_roi, 500000)
    bbox = ndi.find_objects(roi_ilot)

    return bbox[0]

def mask_nucl(img_denoised):
    """Mask the nucleus using local threshold"""
    mask_lst = []
    for img in img_denoised:
        mask_dapi = img_denoised[3] > filters.threshold_local(img_denoised[3],
                                                              81,
                                                              "mean")
        #d = morphology.selem.diamond(radius=5)
        #mask_dapi = morphology.closing(mask_dapi, d)
        mask_dapi = morphology.remove_small_holes(mask_dapi, 200)
        #mask_dapi = morphology.remove_small_objects(mask_dapi.astype(bool), 1000)
        mask_lst.append(mask_dapi)
    return(np.stack(mask_lst))

def mask_nucleus_hard_subs(dapi_roi_z):
    mask_lst = []
    for img in dapi_roi_z:
        mask_dapi = dapi_roi_z[3] > filters.threshold_local(dapi_roi_z[3], 81, "mean")
        d = morphology.selem.diamond(radius=2)
        mask_dapi = morphology.closing(mask_dapi, d)
        mask_dapi = morphology.remove_small_holes(mask_dapi, 1000)
        #mask_dapi = morphology.remove_small_objects(mask_dapi.astype(bool), 1000)
        mask_lst.append(mask_dapi)
    return(np.stack(mask_lst))

def segment_ind_cells(img_denoised, Edu_roi, yap_roi, dapi_roi, hard_sup = False):

    # Find highest variance in the image:
    mask_n = np.zeros(img_denoised.shape, dtype = bool)
    mask_n[img_denoised>filters.threshold_otsu(img_denoised)] = 1
    variance = [np.var(image) for _, image in enumerate(mask_n)]
    idx = np.argmax(variance)

    # Limit the z stack to +/- 3 steps around highest variance:
    nucleus_roi_z = img_denoised[idx-3:idx+3]
    nucleus_int = dapi_roi[idx-3:idx+3]
    Edu_int = Edu_roi[idx-3:idx+3]
    yap_roi = yap_roi[idx-3:idx+3]
    yap_float = img_as_float(yap_roi)
    yap_float_tvc = restoration.denoise_tv_chambolle(yap_float, weight = 0.02)


    # Adjust depending if on hard substrate or not:
    if not hard_sup:
        mask_nucleus =  mask_nucl(nucleus_roi_z)
        sigma = 4
        mask_yap = np.zeros(yap_float_tvc.shape, dtype = bool)
        mask_yap[yap_float_tvc>filters.threshold_otsu(yap_float_tvc)*0.1] = True
        mask_yap = morphology.remove_small_holes(mask_yap, 5000)
    else:
        mask_nucleus =  mask_nucleus_hard_subs(nucleus_roi_z)
        sigma = 10
        mask_yap = np.zeros(yap_float_tvc.shape, dtype = bool)
        mask_yap[yap_float_tvc>filters.threshold_otsu(yap_float_tvc)*0.3] = True
        mask_yap = morphology.remove_small_holes(mask_yap, 5000)
        mask_yap = morphology.remove_small_objects(mask_yap, 10000)

    # Calculate the distance map
    distance = ndi.distance_transform_edt(mask_nucleus[3])
    # Smooth the distance map to avoid oversegmentation:
    smooth_distance = filters.gaussian(distance, sigma=sigma)

    # From the distance mask find the markers for the cells:
    local_maxi = feature.peak_local_max(smooth_distance, indices=True,
                 exclude_border=False,footprint=np.ones((15, 15)))
    markers_lm = np.zeros(smooth_distance.shape, dtype=np.int)
    markers_lm[local_maxi[:,0].astype(np.int), \
    local_maxi[:,1].astype(np.int)] = np.arange(len(local_maxi[:,0])) + 1
    markers_lm = morphology.dilation(markers_lm, morphology.disk(5))

    markers_3D = np.zeros(nucleus_roi_z.shape, dtype = np.int)
    markers_3D[3] = markers_lm
    markers_3D = ndi.label(markers_3D)[0]

    # Edge detection with sobel operator:
    sobel = np.empty_like(nucleus_roi_z)

    for plane, image in enumerate(nucleus_roi_z):
        sobel[plane] = filters.sobel(image)

    # Segmentation cells using watershed
    seg_nucleus = segmentation.watershed(sobel, markers_3D, compactness = 0.001)

    # Remove background form segmentation:
    nucleus_seg_back = np.copy(seg_nucleus)
    nucleus_seg_back[~mask_yap] = 0

    return (mask_yap, idx,
            yap_roi, nucleus_roi_z, nucleus_int,
            nucleus_seg_back, mask_nucleus, Edu_int)

def seg_nuc_cyto(mask_nucleus, nucleus_roi_z, nucleus_seg_back):
    # find intensity image inside or outside mask (nucleus/cyto)

    bw = np.copy(mask_nucleus)
    bw[bw>0] = 1
    bw = bw.astype('bool')

    #First we do nucleus
    labeled_nucleus = measure.label(nucleus_seg_back)
    labeled_nucleus[~bw] = 0
    #re label to separate small objects not connected
    re_labeled_nucleus = labeled_nucleus
    prop_nucleus = measure.regionprops(re_labeled_nucleus,
                   intensity_image = nucleus_roi_z)
    nucleus_area = [prop.area for prop in prop_nucleus]
    nucleus_intensity = [prop.mean_intensity for prop in prop_nucleus]

    area_median = np.mean(np.asarray(nucleus_area)[np.where(np.asarray(nucleus_area) > 50)])/2

    for area, mean_int, region in zip(nucleus_area, nucleus_intensity, prop_nucleus):
        if area < area_median or mean_int < filters.threshold_otsu(nucleus_roi_z)*0.75 \
                       or mean_int > filters.threshold_otsu(nucleus_roi_z)*2.5:
            re_labeled_nucleus[tuple(region.coords.T)] = 0

    #Second, same thing for cyto
    labeled_cyto = measure.label(nucleus_seg_back)
    prop_cyto = measure.regionprops(labeled_cyto, re_labeled_nucleus)
    re_labeled_cyto = np.copy(labeled_cyto)

    for prop in prop_cyto:
        if prop.max_intensity == 0:
            re_labeled_cyto[tuple(prop.coords.T)] = 0
        else:
            re_labeled_cyto[tuple(prop.coords.T)] = prop.max_intensity

    cell_labeled = np.copy(re_labeled_cyto)
    re_labeled_cyto[bw] = 0

    #make sure num of cyto = numb of nucleus:
    prop_nuc = measure.regionprops(re_labeled_nucleus)
    prop_cyt = measure.regionprops(re_labeled_cyto)

    if len(prop_nuc) != len(prop_cyt):
        re_label = True
        while re_label:
            re_label = False
            for prop1, prop2 in zip(prop_nuc, prop_cyt):
                if prop1.label != prop2.label:
                    re_labeled_nucleus[tuple(prop1.coords.T)] = 0
                    prop_nuc = measure.regionprops(re_labeled_nucleus)
                    if len(prop_nuc) != len(prop_cyt):
                        re_label = True
                        break
                    else:
                        break
    return(re_labeled_nucleus, re_labeled_cyto, cell_labeled)

def outer_layer(mask_yap, cell_labeled, idx):

    line_extern = morphology.erosion(
        mask_yap[idx],
        morphology.disk(5)) ^ morphology.dilation(mask_yap[idx],
        morphology.disk(1))
    line_extern = morphology.remove_small_objects(line_extern, 1000)

    mask_outside = np.copy(cell_labeled[idx])
    mask_outside[~line_extern] = 0

    label_outside = measure.regionprops(mask_outside)

    label_outside_list = []
    for prop in label_outside:
        label_outside_list.append(prop.label)

    return label_outside_list

def list_mean_int_in_out(region_props, label_outside_list):

    mean_int_inside = []
    mean_int_outside = []
    median_int_inside = []
    median_int_outside = []

    for prop in region_props:
        if prop.label not in label_outside_list:
            int_inside = prop.intensity_image
            # median intensity cells inside "island"
            try:
                median_int_inside.append(
                    np.median(int_inside[np.nonzero(int_inside)]))
            except IndexError:
                median_int_inside.append(np.nan)
            # mean intensity cells inside "island"
            mean_int_inside.append(
                np.mean(int_inside[np.nonzero(int_inside)]))


        if prop.label in label_outside_list:
            # median intensity cells outside "island"
            int_outside = prop.intensity_image
            try:
                median_int_outside.append(
                    np.median(int_outside[np.nonzero(int_outside)]))
            except IndexError:
                median_int_outside.append(np.nan)
            # median intensity cells outide "island"
            mean_int_outside.append(
                np.mean(int_outside[np.nonzero(int_outside)]))

    return mean_int_inside, mean_int_outside, median_int_inside, median_int_outside
