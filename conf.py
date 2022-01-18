import numpy as np

def get_image_info(image_name):
    """
    Returns the path of the input image and some more information for training.
    As written in the paper, the method uses repeating object window for rescaling the image and initialize the positive bucket.
    This was done by asking the user to mark a window, however, in this version it is hard-coded here.
    :param image_name: name of the image from the dataset (as written in the table in the paper)
    """
    th = 0.85
    number_of_patches = 8
    show_gray = 0
    max_num_cells = 400
    gt_color = [255,255,255]
    dir = "dataset/exp/"
    patch_sz = 21
    if image_name == "Water":
        path = dir+"Water.png"
        im_size = [276,404]
        window_loc = [[326,147]]
        gt_color = [255, 242, 0]
    elif image_name == "CellSml":
        path = dir+"CellSml.png"
        im_size = [400,400]
        window_loc = [[222,143]]
        show_gray = 1
    elif image_name == "CellLrg":
        path = dir+"CellLrg.png"
        im_size = [320,320]
        window_loc = [[74,158]]
        show_gray = 1
    elif image_name == "Beer":
        path = dir+"Beer.png"
        im_size = [336,494]
        window_loc = [[250,160]]
        gt_color = [163,73,164]
    elif image_name == "Pills":
        path = dir+"Pills.png"
        im_size = [100,280]
        window_loc = [[121,35]]
    elif image_name == "Flowers":
        path = dir+"Flowers.png"
        im_size = [200,424]
        window_loc = [[277,17],[64,61]]
        number_of_patches = 16
        gt_color = [0,162,232]
    elif image_name == "Sheep":
        path = dir+"Sheep.png"
        alpha = 0.8
        im_size = [300,666]
        window_loc = [[230,118],[318,152],[338,135],[350,188]]
        number_of_patches = 16
        gt_color = [255,0,0]
    elif image_name == "Cars":
        path = dir+"Cars.png"
        im_size = [188,412]
        window_loc = [[195,143],[194,80]]
        gt_color = [163,74,164]
    elif image_name == "Crowd":
        path = dir+"Crowd.png"
        im_size = [372,620]
        window_loc =[[209,277]]
        max_num_cells = 1000
        number_of_patches = 16
    elif image_name == "Crabs":
        path = dir+"Crabs.png"
        im_size = [400,700]
        window_loc = [[578,351]]
        gt_color = [0, 0, 0]
    elif image_name == "Matches":
        path = dir+"Matches.png"
        im_size = [272,380]
        window_loc = [[60,73]]
    elif image_name == "CarsBg":
        path = dir+"CarsBg.png"
        im_size = [612,568]
        window_loc = [[387,457]]
        max_num_cells = 1500
        gt_color = [34,177,76]
        patch_sz = 15
    elif image_name == "Birds":
        path = dir+"Birds.png"
        im_size = [320,480]
        window_loc = [[258,171],[98,194]]
        number_of_patches = 32
        gt_color = [255,242,0]
    elif image_name == "Parasol":
        path = dir+"Parasol.png"
        im_size = [244,196]
        window_loc = [[9,102]]
        number_of_patches = 16
        gt_color = [34,177,76]
    elif image_name == "Beach":
        path = dir+"Beach.png"
        im_size = [420,860]
        window_loc = [[239,123]]
        number_of_patches = 16
        gt_color = [34,177,76]
        patch_sz = 15
    elif image_name == "Wall":
        path = dir+"Wall.png"
        im_size = [270,270]
        window_loc = [[42,193]]
        gt_color = [255,255,255]
    elif image_name == "Cookies":
        path = dir+"Cookies.png"
        im_size = [240,312]
        window_loc = [[119,132]]
        gt_color = [34,177,76]
    elif image_name == "Chairs":
        path = dir+"Chairs.png"
        im_size = [360,480]
        window_loc = [[332,132]]
        max_num_cells = 1000
        gt_color = [0,0,0]
    elif image_name == "Candles":
        path = dir+"Candles.png"
        im_size = [160,300]
        window_loc = [[141,102]]
        gt_color = [34,177,76]
    elif image_name == "Logs":
        path = dir+"Logs.png"
        im_size = [236,324]
        window_loc = [[129,107]]
        gt_color = [0,162,232]
    elif image_name == "Peas":
        path = dir+"Peas.png"
        im_size = [152,228]
        window_loc = [[40,68]]
        number_of_patches = 16
        gt_color = [255,255,255]
    elif image_name == "CokeReg":
        path = dir+"CokeReg.png"
        im_size = [212,580]
        window_loc = [[250,139]]
        gt_color = [34,177,76]
    elif image_name == "CokeDiet":
        path = dir+"CokeDiet.png"
        im_size = [212,580]
        window_loc = [[321,140]]
        gt_color = [34,177,76]
    elif image_name == "Antarctica":
        path = dir+"Antarctica.png"
        im_size = [272,952]
        window_loc = [[28,72],[309,128]]
        number_of_patches = 32
        gt_color = [34,177,76]
    elif image_name == "Oranges":
        path = dir+"Oranges.png"
        im_size = [400,272]
        window_loc = [[87,283]]
        gt_color = [133,58,133]
    elif image_name == "Discussion":
        path = dir+"Discussion.png"
        im_size = [600,800]
        window_loc = [[394,212],[241,294],[217,147]]
        number_of_patches = 16
        gt_color = [83,0,13]
        patch_sz = 15
    elif image_name == "Hats":
        path = dir+"Hats.png"
        im_size = [304,600]
        window_loc = [[160,55],[331,49],[84,193]]
        number_of_patches = 16
        gt_color = [237,28,36]
    elif image_name == "Fish097":
        path = dir+"Fish097.png"
        im_size = [412,300]
        window_loc = [[15,47]]
        number_of_patches = 32
        gt_color = [255,255,255]
    elif image_name == "Fish107":
        path = dir+"Fish107.png"
        im_size = [412,300]
        window_loc = [[77,128]]
        number_of_patches = 32
        gt_color = [255,255,255]
    elif image_name == "Birds002":
        path = dir+"Birds002.png"
        im_size = [624,964]
        window_loc = [[559,485]]
        max_num_cells = 1000
        number_of_patches = 16
        gt_color = [255,255,255]
    elif image_name == "Bees":
        path = dir+"Bees.png"
        im_size = [248,980]
        window_loc = [[706,113]]
        number_of_patches = 16
        gt_color = [255,255,255]
    elif image_name == "Soldiers":
        path = dir+"Soldiers.png"
        im_size = [252,396]
        window_loc = [[94,23],[224,65],[146,95]]
        gt_color = [255,255,255]
    elif image_name == "RealCells":
        path = dir+"RealCells.png"
        im_size = [488,488]
        window_loc = [[161,298]]
        number_of_patches = 16
        gt_color = [237,28,36]
    else:
        print(image_name + ' not found image name')
        exit()

    return [im_size,window_loc,number_of_patches,th,path,show_gray,max_num_cells,gt_color,patch_sz]
