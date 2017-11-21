import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np
from skimage import segmentation
from skimage import measure
from skimage import filters
from sklearn.cluster import KMeans
from skimage.color import label2rgb
import holoviews as hv
import holoviews.util

def display_im(img):
    z ,_ ,_ = img.shape
    plane = np.int(z/2)
    fig, ax = plt.subplots(1, 1, figsize = (8,8))
    ax.imshow(img[plane], cmap = "gray")
    plt.axis('off')

def loading(filename):
    return imread(filename)

def display_channels(img, cmap = 'gray'):
    z, y, x, c = img.shape

    fig, axes = plt.subplots(nrows=1, ncols=c, figsize = (16,8))
    im = np.swapaxes(img, 0, 3)

    for plane, image in enumerate(im):
        vmin = image[:,:,np.int(z/2)].min()
        vmax = image[:,:,np.int(z/2)].max()
        axes[plane].imshow(image[:,:,np.int(z/2)], cmap = cmap,
                           vmin=vmin, vmax=vmax)
        axes[plane].set_xticks([])
        axes[plane].set_yticks([])
        axes[plane].set_title('channel_{}'.format(plane+1))


def display_all_zsteps(
    img, cmap = 'gray',
    skip = False, skip_size = 2,
    save = False, name = 'cell_display.png'):

    #indexing ==> a[start_index:end_index:step]

    dim = img.ndim

    if dim == 4:
        z, _, _, _ = img.shape
    else:
        z, _, _ = img.shape

    if skip:

        z_rescaled = np.int(np.floor(z/skip_size))
        rescaled = np.zeros((z_rescaled, y, x), dtype = dapi.dtype)

        for index, plane in enumerate(dapi[::skip_size]):
            if  index == z_rescaled:
                break
            rescaled[index] = plane

            img = rescaled



    nrows=np.int(np.floor(np.sqrt(z)))
    ncols=np.int(np.ceil(np.sqrt(z)))

    if nrows * ncols < z:
        nrows = nrows + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize = (16,16))
    #fig.suptitle('Z stack', fontsize = 16)

    for x in range(z):

        #flooring division
        i = x // ncols
        #modulo
        j = x % ncols
        axes[i,j].imshow(img[x], cmap = cmap)
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
        axes[i,j].set_title('plane_{}'.format(x+1))

    if save == True:
        plt.savefig(name)


def display_seg(img, seg, save = False, name = 'cell_seg.png'):

    z, _, _ = img.shape

    nrows=np.int(np.floor(np.sqrt(z)))
    ncols=np.int(np.ceil(np.sqrt(z)))

    if nrows * ncols < z:
        nrows = nrows + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize = (16,16))

    for x in range(z):

        #flooring division
        i = x // ncols
        #modulo
        j = x % ncols

        axes[i,j].imshow(segmentation.mark_boundaries(img[x],
                                                      seg[x]))
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
        axes[i,j].set_title('plane_{}'.format(x+1))

    if save == True:
        plt.savefig(name)

def plot_list_int(region_props):
    # Print max int all int image

    list_int_img = []
    for prop in region_props:
        variance = [np.var(image) for _, image in enumerate(prop.intensity_image)]
        idx = np.argmax(variance)

        list_int_img.append(prop.intensity_image[idx])

    nrows=np.int(np.floor(np.sqrt(len(list_int_img))))
    ncols=np.int(np.ceil(np.sqrt(len(list_int_img))))

    if nrows * ncols < len(list_int_img):
        nrows = nrows + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize = (16,16))

    for x in range(len(list_int_img)):
        #flooring division
        i = x // ncols
        #modulo
        j = x % ncols
        axes[i,j].imshow(list_int_img[x])
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
        axes[i,j].set_title('region_{}'.format(x+1))

def plot_outer_cells(
        yap_roi, cell_labeled, regprops_yap_nucleus, label_outside_list,
        figure = "test1.png"):

    plt.ioff()
    #fig, ax = plt.subplots(figsize = (8,8))

    plt.imshow(segmentation.mark_boundaries(yap_roi[3],
                                            cell_labeled[3]))

    plt.axis('off')
    for j in regprops_yap_nucleus:
        if j.label in label_outside_list:
            plt.scatter(j.centroid[2], j.centroid[1], c = 'r')
        if j.label not in label_outside_list:
            plt.scatter(j.centroid[2], j.centroid[1], c = 'g')
    plt.savefig(figure, dpi = 300)
    plt.close()


def save_nuc_segmentation_no_ilot(cleaned, seg_nucleus, dapi_roi_z, flat_ln, figure = "test0.png"):

    segj = segmentation.join_segmentations(cleaned, seg_nucleus)
    regions = measure.regionprops(segj, intensity_image = flat_ln)
    region_means = [r.mean_intensity for r in regions]
    model = KMeans(n_clusters=3)

    region_means = np.array(region_means).reshape(-1, 1)
    model.fit(np.array(region_means).reshape(-1, 1))
    bg_fg_labels = model.predict(region_means)
    bg_fg_labels = np.where(bg_fg_labels==np.argmin(model.cluster_centers_), 0, 1)

    cleaned_nuc = segj.copy()
    for bg_fg, region in zip(bg_fg_labels, regions):
        if bg_fg == 0:
            cleaned_nuc[tuple(region.coords.T)] = bg_fg
        else:
            pass

    plt.ioff()
    plt.imshow(segmentation.mark_boundaries(dapi_roi_z[3],
                                        cleaned_nuc[3]))
    plt.savefig(figure, dpi = 300)
    plt.close()
    return cleaned_nuc

def save_nuc_segmentation(yap_int, Edu_int, nucleus_roi_z, re_labeled_nucleus, figure = "test0.png"):


    image_label_overlay = label2rgb(re_labeled_nucleus, bg_label=0)


    plt.ioff()

    fig, ax = plt.subplots(2,2,figsize = (16,16))

    ax[0,0].imshow(segmentation.mark_boundaries(nucleus_roi_z[3],
                                            re_labeled_nucleus[3]))
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_title('DAPI segmentation')

    ax[0,1].imshow(image_label_overlay[3])
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,1].set_title('DAPI segmentation (labels)')

    ax[1,0].imshow(segmentation.mark_boundaries(yap_int[3],
                                            re_labeled_nucleus[3]))
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title('YAP segmentation')

    ax[1,1].imshow(segmentation.mark_boundaries(Edu_int[3],
                                            re_labeled_nucleus[3]))

    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    ax[1,1].set_title('Edu Segmentation')

    plt.savefig(figure, dpi = 300)
    plt.close()
    #return labeled


def box_plot_mean(mean_int_inside, mean_int_outside, title = ''):

    data = [mean_int_inside,
                  mean_int_outside]

    fig, ax = plt.subplots(figsize = (8,8))


    bplot = ax.boxplot(data,
                         notch=False,  # notch shape
                         #whis = [5, 95],
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

    # fill with colors
    colors = ['pink', 'lightblue']

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines

    ax.yaxis.grid(True)
    ax.set_ylabel('mean YAP in {} intensity (A.U.)'.format(title), fontsize=14)

    # add x-tick labels
    xtickNames =plt.setp(ax, xticklabels=['inside', 'outside'])

    ax.set_title(title, fontsize=16)
    fig.tight_layout()
    plt.setp(xtickNames, fontsize=14)

def slice_exploreur(image_3D, i=1, label = ""):


    def image_slice(k):
        return(hv.Image(image_3D[k,::i,::i], bounds=(0,0,image_3D.shape[2],image_3D.shape[1]), label= label))
    keys = list(range(0, image_3D.shape[0]))
    return hv.HoloMap({N:image_slice(N) for N in keys},  kdims=['Zstep'])

def orthogonal_view(image_3D, i=1):
    def image_slice(k):
        return(hv.Image(image_3D[k,::i,::i], bounds=(0,0,image_3D.shape[2],image_3D.shape[1]), label= "XY view"))
    keys = list(range(0, image_3D.shape[0]))
    holomap = hv.HoloMap({N:image_slice(N) for N in keys},  kdims=['Zstep'])

    def image_slice1(k):
        return(hv.Image(image_3D[:,int(image_3D.shape[2]- k),:],
                        bounds=(0,0,image_3D.shape[2],image_3D.shape[1]),
                        label= "XZ view"))
    def image_slice2(k):
        return(hv.Image(np.flip(image_3D[:,:,int(k)], axis = 1),
                        bounds=(0,0,image_3D.shape[2],image_3D.shape[1]),
                        label= "YZ view"))

    plot_opts = {'height': 120}
    plot_opts2 = {'height': 300,
                 'width': 120}

    posy = hv.streams.PointerY(y=100)
    posx = hv.streams.PointerX(x=100)

    hline = hv.DynamicMap(lambda y: hv.HLine(y), streams=[posy])
    vline = hv.DynamicMap(lambda x: hv.VLine(x), streams=[posx])

    crosssection1 = hv.DynamicMap(lambda y: image_slice1(y), streams=[posy])
    crosssection2 = hv.DynamicMap(lambda x: image_slice2(x), streams=[posx])

    return ((holomap * hline * vline) << crosssection2.opts(plot=plot_opts2) << crosssection1.opts(plot=plot_opts))
