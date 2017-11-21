import os
import pandas as pd
from io_czi import cziReader
import argparse
import numpy as np
from skimage import img_as_float
from skimage import restoration
from segmentation_final import *
from plot_eva import *

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def batch_analysis(analysis = "ctrl", hard_sup = False, path = "directory/to/file"):

    path = path
    imageformat=".czi"
    directory = path + "/result2"

    if not os.path.exists(directory):
        os.makedirs(directory)

    # create a list of all the files that end with .czi
    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
    # Check which drug to analysis (analysis arg)
    data = [sentence for sentence in imfilelist if analysis in sentence]
    # Prepare all the different list I will need:

    mean_int_nucleus_inside_lst = []
    mean_int_nucleus_outside_lst = []

    mean_int_cyto_inside_lst = []
    mean_int_cyto_outside_lst = []

    mean_int_control_inside_lst = []
    mean_int_control_outside_lst = []

    mean_int_Edu_inside_lst = []
    mean_int_Edu_outside_lst = []

    median_int_nucleus_inside_lst = []
    median_int_nucleus_outside_lst = []

    median_int_cyto_inside_lst = []
    median_int_cyto_outside_lst = []

    median_int_control_inside_lst = []
    median_int_control_outside_lst = []

    median_int_Edu_inside_lst = []
    median_int_Edu_outside_lst = []

    # Looping through the different images:
    for numb, image in enumerate(data):

        #May have to go through Metadata to know the order of the channels
        reader = cziReader.CziReader(image)
        #Attention at the order of dye!
        yap, Edu, dapi = np.squeeze(reader.load()).transpose(1,0,2,3)

        if not hard_sup:
            # Calculate the bounding box on the dapi
            bbox = roi(dapi)

            # Apply the bbox to every channel
            dapi_roi = dapi[bbox]
            yap_roi = yap[bbox]
            Edu_roi = Edu[bbox]
        else:
            dapi_roi = dapi
            yap_roi = yap
            Edu_roi = Edu

        dapi_float = img_as_float(dapi_roi)
        dapi_float_tvc = restoration.denoise_tv_chambolle(dapi_float, weight = 0.05)

        # When cells are in ilot
        mask_yap, \
        idx, yap_int, dapi_roi_z, \
        dapi_int, dapi_seg, cleaned, Edu_int = segment_ind_cells(dapi_float_tvc, Edu_roi, yap_roi,
                                                       dapi_roi, hard_sup = hard_sup)

        re_labeled_nucleus, re_labeled_cyto, cell_labeled = seg_nuc_cyto(cleaned, dapi_roi_z, dapi_seg)

        # Calculate region property for yap ,dapi and edu:
        regprops_yap_nucleus = measure.regionprops(re_labeled_nucleus, yap_int)
        regprops_yap_cyto = measure.regionprops(re_labeled_cyto, yap_int)
        regprops_dapi = measure.regionprops(re_labeled_nucleus, dapi_int)
        regprops_Edu = measure.regionprops(re_labeled_nucleus, Edu_int)

        #cleaned_nuc = save_nuc_segmentation(cleaned, seg_nuc, dapi_roi_z, figure = analysis + "_nucleus_" + str(numb) +".png")
        save_nuc_segmentation(yap_int, Edu_int, dapi_roi_z,
                              re_labeled_nucleus, figure = path + "/" +"result2"+ "/" + analysis + "_nucleus_" + str(numb) +".png")


        #regprops_yap_nucleus =  measure.regionprops(dapi_seg, yap_int)
        label_outside_list = outer_layer(mask_yap, cell_labeled, 3)

        plot_outer_cells(yap_int, cell_labeled, regprops_yap_nucleus,
                         label_outside_list, figure = path + "/" +"result2" + "/" + analysis + "_layers_" + str(numb) + ".png")

        #print("nucleus")
        mean_int_nucleus_inside, mean_int_nucleus_outside, \
        median_int_nucleus_inside, median_int_nucleus_outside = list_mean_int_in_out(regprops_yap_nucleus,
                                                                                 label_outside_list)
        #print("cyto")
        mean_int_cyto_inside, mean_int_cyto_outside, \
        median_int_cyto_inside, median_int_cyto_outside = list_mean_int_in_out(regprops_yap_cyto,
                                                                                 label_outside_list)
        #print("controle(dapi)")
        mean_int_control_inside, mean_int_control_outside, \
        median_int_control_inside, median_int_control_outside = list_mean_int_in_out(regprops_dapi,
                                                                                 label_outside_list)

        #Edu
        mean_int_Edu_inside, mean_int_Edu_outside, \
        median_int_Edu_inside, median_int_Edu_outside = list_mean_int_in_out(regprops_Edu,
                                                                                 label_outside_list)

        #Mean
        mean_int_nucleus_inside_lst.extend(mean_int_nucleus_inside)
        mean_int_nucleus_outside_lst.extend(mean_int_nucleus_outside)

        mean_int_cyto_inside_lst.extend(mean_int_cyto_inside)
        mean_int_cyto_outside_lst.extend(mean_int_cyto_outside)

        mean_int_control_inside_lst.extend(mean_int_control_inside)
        mean_int_control_outside_lst.extend(mean_int_control_outside)

        mean_int_Edu_inside_lst.extend(mean_int_Edu_inside)
        mean_int_Edu_outside_lst.extend(mean_int_Edu_outside)

        #Median
        median_int_nucleus_inside_lst.extend(median_int_nucleus_inside)
        median_int_nucleus_outside_lst.extend(median_int_nucleus_outside)

        median_int_cyto_inside_lst.extend(median_int_cyto_inside)
        median_int_cyto_outside_lst.extend(median_int_cyto_outside)

        #DAPI
        median_int_control_inside_lst.extend(median_int_control_inside)
        median_int_control_outside_lst.extend(median_int_control_outside)

        #EDU
        median_int_Edu_inside_lst.extend(median_int_Edu_inside)
        median_int_Edu_outside_lst.extend(median_int_Edu_outside)

    return((mean_int_nucleus_inside_lst, mean_int_nucleus_outside_lst,
            mean_int_cyto_inside_lst, mean_int_cyto_outside_lst,
            mean_int_control_inside_lst, mean_int_control_outside_lst,
            median_int_nucleus_inside_lst, median_int_nucleus_outside_lst,
            median_int_cyto_inside_lst, median_int_cyto_outside_lst,
            median_int_control_inside_lst, median_int_control_outside_lst,
            mean_int_Edu_inside_lst, mean_int_Edu_outside_lst,
            median_int_Edu_inside_lst, median_int_Edu_outside_lst))



def main(analysis = "ctrl", hard_sup = False):

    parser = argparse.ArgumentParser(description='Example with non-optional arguments')

    parser.add_argument('--path_source', help='path to directory with file to analyse', required=True)
    parser.add_argument('-drug','--list', action='append', help='Drug to analyse as it appear in filename', required=True)
    parser.add_argument('--hard_sup', type=str2bool, help='true or false, are the cells grown on hard support')
    opts = parser.parse_args()

    to_analyse = opts.list

    df_final = []
    for file in to_analyse:
        print(file)
        result = batch_analysis(analysis = file, hard_sup = opts.hard_sup, path = opts.path_source)

        data_inside = {"mean intensity nucleus": result[0],
                       "mean intensity dapi": result[4],
                        "mean intensity cyto": result[2],
                        "mean intensity Edu": result[12],
                        "median intensity nucleus": result[6],
                        "median intensity dapi": result[10],
                        "median intensity cyto": result[8],
                        "median intensity Edu": result[14],
                        "where": ["inside"] * len(result[0]),
                        "drug": [file] * len(result[0])}
                        #"drug": "ctrl"}
        data_pd_ins = pd.DataFrame(data_inside)

        data_outside = {"mean intensity nucleus": result[1],
                        "mean intensity dapi": result[5],
                        "mean intensity cyto": result[3],
                        "mean intensity Edu": result[13],
                        "median intensity nucleus": result[7],
                        "median intensity dapi": result[11],
                        "median intensity cyto": result[9],
                        "median intensity Edu": result[15],
                        "where": ["outside"] * len(result[1]),
                        "drug": [file] * len(result[1])}
                        #"drug": "ctrl"}
        data_pd_out = pd.DataFrame(data_outside)

        frames = [data_pd_ins, data_pd_out]
        df = pd.concat(frames)
        df_final.append(df)

    path = opts.path_source
    df_result = pd.concat(df_final)
    df_result.to_csv(path + "/" +"result2"+ "/" + path.rsplit('/', 1)[-1] + '.csv')


if __name__ == '__main__':
    main()
