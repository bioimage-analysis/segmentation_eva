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
    #roi_ilot = morphology.closing(mask_roi, morphology.ball(3))
    #roi_ilot = morphology.closing(roi_ilot, morphology.ball(3))
    #roi_ilot = morphology.dilation(roi_ilot, morphology.ball(3))
    #roi_ilot = morphology.dilation(roi_ilot, morphology.ball(3))
    #roi_ilot = morphology.remove_small_objects(roi_ilot, 1000000)
    roi_ilot = morphology.remove_small_objects(mask_roi, 500000)
    bbox = ndi.find_objects(roi_ilot)

    return bbox[0]


def mask_nucl(img_denoised):
    mask_lst = []
    for img in img_denoised:
        mask_dapi = img_denoised[3] > filters.threshold_local(img_denoised[3], 81, "mean")
        d = morphology.selem.diamond(radius=5)
        mask_dapi = morphology.closing(mask_dapi, d)
        mask_dapi = morphology.remove_small_holes(mask_dapi, 100)
        mask_dapi = morphology.remove_small_objects(mask_dapi.astype(bool), 1000)
        mask_lst.append(mask_dapi)
    return(np.stack(mask_lst))


def segment_ind_cells(img_denoised, Edu_roi, yap_roi, dapi_roi, flat = True):

    mask_nucleus = np.zeros(img_denoised.shape, dtype = bool)
    mask_nucleus[img_denoised>filters.threshold_otsu(img_denoised)] = 1
    variance = [np.var(image) for _, image in enumerate(mask_nucleus)]
    idx = np.argmax(variance)

    nucleus_roi_z = img_denoised[idx-3:idx+3]
    nucleus_int = dapi_roi[idx-3:idx+3]
    Edu_int = Edu_roi[idx-3:idx+3]

    flat_ln = np.empty_like(nucleus_roi_z)

    for plane, image in enumerate(nucleus_roi_z):
        flat_ln[plane] = locnormalize(image, 150, 150)

    if flat:

        mask_nucleus = mask_nucleus[idx-3:idx+3]
        #mask_nucleus = mask_nucl(nucleus_roi_z)
        distance = ndi.distance_transform_edt(mask_nucleus[3])

        local_maxi = feature.peak_local_max(distance, indices=True,
                                            exclude_border=False,footprint=np.ones((25, 25)))
        markers_lm = np.zeros(distance.shape, dtype=np.int)
        markers_lm[local_maxi[:,0].astype(np.int), local_maxi[:,1].astype(np.int)] = np.arange(len(local_maxi[:,0])) + 1
        markers_lm = morphology.dilation(markers_lm, morphology.disk(5))
        markers_3D = np.zeros(nucleus_roi_z.shape, dtype = np.int)
        markers_3D[3] = markers_lm
        markers_3D = ndi.label(markers_3D)[0]

    else:

        mask_nucleus= np.zeros(flat_ln.shape, dtype = bool)
        mask_nucleus[flat_ln>filters.threshold_otsu(flat_ln)] = 1
        mask_nucleus = morphology.closing(mask_nucleus, morphology.ball(5))
        mask_nucleus = morphology.remove_small_objects(mask_nucleus, 10000)

        distance = ndi.distance_transform_edt(mask_nucleus[3])

        local_maxi = feature.peak_local_max(distance, indices=True,
                                            exclude_border=False,footprint=np.ones((20, 20)))
        markers_lm = np.zeros(distance.shape, dtype=np.int)
        markers_lm[local_maxi[:,0].astype(np.int), local_maxi[:,1].astype(np.int)] = np.arange(len(local_maxi[:,0])) + 1
        markers_lm = morphology.dilation(markers_lm, morphology.disk(5))
        markers_3D = np.zeros(nucleus_roi_z.shape, dtype = np.int)
        markers_3D[3] = markers_lm
        markers_3D = ndi.label(markers_3D)[0]

    sobel = np.empty_like(nucleus_roi_z)

    for plane, image in enumerate(nucleus_roi_z):
        sobel[plane] = filters.sobel(image)

    seg_nucleus = segmentation.watershed(sobel, markers_3D, compactness = 0.1)

    yap_roi = yap_roi[idx-3:idx+3]
    yap_float = img_as_float(yap_roi)
    yap_float_tvc = restoration.denoise_tv_chambolle(yap_float, weight = 0.02)

    mask_yap = np.zeros(yap_float_tvc.shape, dtype = bool)
    mask_yap[yap_float_tvc>filters.threshold_otsu(yap_float_tvc)*0.1] = 1
    mask_yap = morphology.remove_small_holes(mask_yap, 5000)

    #USUALLY USED:
    #mask_yap = morphology.remove_small_holes(mask_yap, 20000)

    #mask_yap = morphology.binary_closing(mask_yap, morphology.ball(2))
    #mask_yap = morphology.remove_small_objects(mask_yap, 100000)
    #mask_yap = morphology.remove_small_holes(mask_yap, 100000)

    nucleus_seg_back = np.copy(seg_nucleus)
    nucleus_seg_back[~mask_yap] = 0

    return (nucleus_seg_back, mask_yap, idx,
            yap_roi, nucleus_roi_z, nucleus_int,
            seg_nucleus, flat_ln, mask_nucleus, Edu_int)

def idx_z(img_denoised):

    mask_nucleus = np.zeros(img_denoised.shape, dtype = bool)
    mask_nucleus[img_denoised>filters.threshold_otsu(img_denoised)] = 1
    variance = [np.var(image) for _, image in enumerate(mask_nucleus)]

    return np.argmax(variance)


def fspecial(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def locnormalize(image, sigma1, sigma2):
    """
    local noramlization, inspired from:
    http://bigwww.epfl.ch/demo/jlocalnormalization/
    """

    x_im, y_im = image.shape
    epsilon=1e-1
    halfsize1=np.ceil(-norm.ppf(epsilon/2, loc=0, scale=sigma1))
    size1=2*halfsize1+1
    halfsize2=np.ceil(-norm.ppf(epsilon/2, loc=0, scale=sigma2))
    size2=2*halfsize2+1
    gaussian1=fspecial(shape = (size1, size1), sigma = sigma1)

    # Need to pad the image to avoid border effect
    x_gau, y_gau = gaussian1.shape
    image_2 = np.lib.pad(image, (int((x_gau-1)/2), int((y_gau-1)/2)), 'edge')

    gaussian2=fspecial(shape = (size2, size2), sigma =sigma2)

    sub = fftconvolve(image_2, gaussian1, mode='valid')

    num = image - sub

    # Need to pad the image to avoid border effect
    x_gau2, y_gau2 = gaussian2.shape
    num_2 = np.lib.pad(num, (int((x_gau2-1)/2), int((y_gau2-1)/2)), 'edge')

    den = np.sqrt(fftconvolve(np.power(num_2, 2),gaussian2, mode='valid'))

    ln=num/den

    return (ln)
'''

def markers_watershed(nucleus_roi_z):

    segments_nuc = segmentation.slic(nucleus_roi_z, n_segments=500,
                   compactness=0.1, max_iter=100, spacing=(5, 1,1),
                   multichannel = False)
    # calculate flat image:

    flat_ln = np.empty_like(nucleus_roi_z)

    for plane, image in enumerate(nucleus_roi_z):
        flat_ln[plane] = locnormalize(image, 150, 150)


    # Get region mean intensity using flat image
    regions = measure.regionprops(segments_nuc, intensity_image = flat_ln)
    region_means = [r.mean_intensity for r in regions]
    #Use KMeans to separate region in 2 clusters
    model = KMeans(n_clusters=2)
    region_means = np.array(region_means).reshape(-1, 1)
    model.fit(np.array(region_means).reshape(-1, 1))

    bg_fg_labels = model.predict(region_means)
    # Label clusters:
    bg_fg_labels = np.where(bg_fg_labels==np.argmin(model.cluster_centers_), 0, 1)

    classified_labels = segments_nuc.copy()
    for bg_fg, region in zip(bg_fg_labels, regions):
        classified_labels[tuple(region.coords.T)] = bg_fg

    distance = ndi.distance_transform_edt(classified_labels[3])
    local_maxi = feature.peak_local_max(distance, indices=True,
                                        exclude_border=False,footprint=np.ones((30, 30)))
    markers_lm = np.zeros(distance.shape, dtype=np.int)
    markers_lm[local_maxi[:,0].astype(np.int), local_maxi[:,1].astype(np.int)] = np.arange(len(local_maxi[:,0])) + 1
    markers_lm = morphology.dilation(markers_lm, morphology.disk(5))
    markers_3D = np.zeros(nucleus_roi_z.shape, dtype = np.int)
    markers_3D[3] = markers_lm
    markers_3D = ndi.label(markers_3D)[0]

    return markers_3D, classified_labels, flat_ln




def segment_ind_cells_no_ilot(img_denoised, yap_roi, dapi_roi):
    # TO USE WHEN CELLS ARE NOT DENSELY PACKED INTO ISLAND

    idx = idx_z(img_denoised)

    nucleus_roi_z = img_denoised[idx-3:idx+3]
    nucleus_int = dapi_roi[idx-3:idx+3]

    markers_3D, classified_labels, flat_ln = markers_watershed(nucleus_roi_z)

    sobel = np.empty_like(nucleus_roi_z)

    for plane, image in enumerate(nucleus_roi_z):
        sobel[plane] = filters.sobel(image)

    seg_nucleus = segmentation.watershed(sobel, markers_3D, compactness = 0.1)

    yap_roi_z = yap_roi[idx-3:idx+3]
    yap_float = img_as_float(yap_roi_z)
    yap_float_tvc = restoration.denoise_tv_chambolle(yap_float, weight = 0.02)

    mask_yap = np.zeros(yap_float_tvc.shape, dtype = bool)
    mask_yap[yap_float_tvc>filters.threshold_otsu(yap_float_tvc)*0.04] = 1
    mask_yap = morphology.remove_small_holes(mask_yap, 10000)
    #mask_yap = morphology.binary_closing(mask_yap, morphology.ball(2))
    #mask_yap = morphology.remove_small_objects(mask_yap, 100000)
    #mask_yap = morphology.remove_small_holes(mask_yap, 100000)

    nucleus_seg_back = np.copy(seg_nucleus)
    nucleus_seg_back[~mask_yap] = 0

    classified_labels= classified_labels.astype('bool')

    return (nucleus_seg_back, mask_yap, idx,
            yap_roi_z, nucleus_roi_z, nucleus_int, seg_nucleus,
            classified_labels, flat_ln)

'''
def segment_nucleus(nucleus_roi_z, flat_ln):

    segments = segmentation.slic(nucleus_roi_z, n_segments=500,
               compactness=0.2, max_iter=50, spacing=(5, 1,1),
               multichannel = False)

    regions = measure.regionprops(segments,
                                  intensity_image = flat_ln)

    region_means = [r.mean_intensity for r in regions]

    # Find 2 clusters of data
    model = KMeans(n_clusters=3)
    region_means = np.array(region_means).reshape(-1, 1)
    model.fit(np.array(region_means).reshape(-1, 1))
    # Separate background from foreground
    bg_fg_labels = model.predict(region_means)
    bg_fg_labels = np.where(bg_fg_labels==np.argmin(model.cluster_centers_), 0, 1)
    # label bg_fg 0, 1
    classified_labels = segments.copy()
    for bg_fg, region in zip(bg_fg_labels, regions):
        classified_labels[tuple(region.coords.T)] = bg_fg

    return classified_labels.astype('bool')


'''
def segment_nucleus(nucleus_roi_z):

    segments = segmentation.slic(nucleus_roi_z, n_segments=500,
               compactness=0.2, multichannel = False)

    prop_test = measure.regionprops(segments, nucleus_roi_z)

    empty_region = []
    for prop in prop_test:
        if prop.mean_intensity < filters.threshold_otsu(nucleus_roi_z)*0.9 or \
           prop.mean_intensity > filters.threshold_otsu(nucleus_roi_z)*3:
            empty_region.append(prop.label)

    cleaned = np.copy(segments)
    for i in empty_region:
        cleaned[np.where(cleaned == i)] = 0

    cleaned[np.where(cleaned>1)] = 1
    cleaned = cleaned.astype('bool')

    return cleaned
'''

def regionprop_protein_in_mask(protein_roi, mask, segmented, outside_nu = True):
    # find intensity image inside or outside mask (nucleus/cyto)

    protein = np.copy(protein_roi)
    if outside_nu:
        protein[mask] = 0
    else:
        protein[~mask] = 0

    region_props = measure.regionprops(segmented, protein)

    return region_props, protein



def seg_nuc_cyto(mask_nucleus, nucleus_roi_z, nucleus_seg_back):
    # find intensity image inside or outside mask (nucleus/cyto)

    bw = np.copy(mask_nucleus)
    bw[bw>0] = 1
    bw = bw.astype('bool')

    #First we do nucleus
    labeled_nucleus = measure.label(nucleus_seg_back)
    labeled_nucleus[~bw] = 0
    #re label to separate small objects not connected
    re_labeled_nucleus = measure.label(labeled_nucleus)

    prop_nucleus = measure.regionprops(re_labeled_nucleus,
                   intensity_image = nucleus_roi_z)

    nucleus_area = [prop.area for prop in prop_nucleus]
    nucleus_intensity = [prop.mean_intensity for prop in prop_nucleus]

    #for area, mean_int, region in zip(nucleus_area, nucleus_intensity, prop_nucleus):
    #    if area < 11000 or mean_int < filters.threshold_otsu(nucleus_roi_z)*0.9 or mean_int > filters.threshold_otsu(nucleus_roi_z)*3:
    #        re_labeled_nucleus[tuple(region.coords.T)] = 0

    for area, mean_int, region in zip(nucleus_area, nucleus_intensity, prop_nucleus):
        if area < 2000 or mean_int < filters.threshold_otsu(nucleus_roi_z)*0.1 or mean_int > filters.threshold_otsu(nucleus_roi_z)*3:
            re_labeled_nucleus[tuple(region.coords.T)] = 0

    #Second same thing for cyto
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

    #region_prop_nucleus = measure.regionprops(re_labeled_nucleus)
    #region_prop_cyto = measure.regionprops(re_labeled_cyto)

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
            #if prop.mean_intensity > filters.threshold_li(yap_roi):
            #mean_int_inside.append(prop.mean_intensity)
            # median
            int_inside = prop.intensity_image
            #median_int_inside.append(
                #np.median(int_inside[np.nonzero(int_inside)]))
            try:
                median_int_inside.append(
                    np.median(int_inside[np.nonzero(int_inside)]))
            except IndexError:
                median_int_inside.append(np.nan)

            mean_int_inside.append(
                np.mean(int_inside[np.nonzero(int_inside)]))

            #print("mean_int_inside", prop.mean_intensity)
            #print("median_int_inside",np.median(int_inside[np.nonzero(int_inside)]))

        if prop.label in label_outside_list:
            #if prop.mean_intensity > filters.threshold_li(yap_roi):
            #mean_int_outside.append(prop.mean_intensity)

            #median outside
            int_outside = prop.intensity_image
            #median_int_outside.append(
                #np.median(int_outside[np.nonzero(int_outside)]))
            try:
                median_int_outside.append(
                    np.median(int_outside[np.nonzero(int_outside)]))
            except IndexError:
                median_int_outside.append(np.nan)

            mean_int_outside.append(
                np.mean(int_outside[np.nonzero(int_outside)]))

    return mean_int_inside, mean_int_outside, median_int_inside, median_int_outside
