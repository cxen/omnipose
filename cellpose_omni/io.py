import os, datetime, gc, warnings, glob
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging, pathlib, sys
from pathlib import Path
from aicsimageio import AICSImage
from csv import reader, writer
import re

try:
    from omnipose.logger import LOGGER_FORMAT
    OMNI_INSTALLED = True
    import ncolor
except Exception as e:
    print(f"Error when importing omnipose or ncolor: {e}")
    OMNI_INSTALLED = False

from . import utils, plot, transforms

try:
    from PySide6 import QtGui, QtCore, Qt, QtWidgets
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False
    
try:
    from google.cloud import storage
    SERVER_UPLOAD = True
except:
    SERVER_UPLOAD = False

from omnipose.memory import force_cleanup

io_logger = logging.getLogger(__name__)

def logger_setup(verbose=False):
    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath('run.log')
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    logging.basicConfig(
                    level=logging.DEBUG if verbose else logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ]
                )
    logger = logging.getLogger(__name__)
    
    # logger.setLevel(logging.DEBUG) # does not fix CLI
    # logger.info(f'WRITING LOG OUTPUT TO {log_file}')
    #logger.handlers[1].stream = sys.stdout

    return logger, log_file

# helper function to check for a path; if it doesn't exist, make it 
def check_dir(path):
    if not os.path.isdir(path):
        # os.mkdir(path)
        os.makedirs(path,exist_ok=True)
        
        
def load_links(filename):
    """
    Read a txt or csv file with label links. 
    These should look like:
        1,2 
        1,3
        4,7
        6,19
        .
        .
        .
    Returns links as a set of tuples. 
    """
    if filename is not None and os.path.exists(filename):
        links = set()
        with open(filename, "r") as file:
            lines = reader(file)
            for l in lines:
                # Check if the line is not empty before processing
                if l:
                    links.add(tuple(int(num) for num in l))
        return links
    else:
        return set()

def write_links(savedir,basename,links):
    """
    Write label link file. See load_links() for its output format. 
    
    Parameters
    ----------
    savedir: string
        directory in which to save
    basename: string
        file name base to which _links.txt is appended. 
    links: set
        set of label tuples {(x,y),(z,w),...}

    """
    with open(os.path.join(savedir,basename+'_links.txt'), "w",newline='') as out:
        csv_out = writer(out)
        for row in links:
            csv_out.writerow(row)

def outlines_to_text(base, outlines):
    with open(base + '_cp_outlines.txt', 'w') as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ','.join(map(str, xy))
            f.write(xy_str)
            f.write('\n')

def imread(filename):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        img = tifffile.imread(filename)
        return img
    elif ext=='.npy':
        return np.load(filename)
    elif ext=='.npz':
        return np.load(filename)['arr_0']
    elif ext=='.czi':
        img = AICSImage(filename).data
        return img
    else:
        try:
            # Read image including alpha channel if present (-1 flag)
            img = cv2.imread(filename, -1)
            if img is None:
                raise ValueError("Failed to read image")
            # Check dimensions
            if img.ndim == 2:
                # Grayscale image, no conversion needed
                return img
            elif img.shape[2] == 3:
                # Convert 3-channel BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                # Convert 4-channel BGRA to RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return img
        except Exception as e:
            io_logger.critical('ERROR: could not read file, %s' % e)
            return None


def imwrite(filename, arr, **kwargs):
    # should transition to imagecodecs instead, faster for webp, probably others too 

    ext = os.path.splitext(filename)[-1].lower()
    if ext in ['.tif', '.tiff']:
        tifffile.imwrite(filename, arr, **kwargs)
    elif ext == '.npy':
        np.save(filename, arr, **kwargs)
    else:
        if ext == '.png':
            compression = kwargs.pop('compression', 9)        
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif ext in ['.jpg', '.jpeg', '.jp2']:
            quality = kwargs.pop('quality', 95)
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == '.webp':
            quality = kwargs.pop('quality', 0)
            # note: webp quality should be >100 for "lossless" compression, though a few pixels still differ 
            # 0 still looks really really good, though
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        else:
            # For any other extension, no special parameters are set.
            params = []
        
        # Handle color conversion for cv2
        if len(arr.shape) > 2:
            if arr.shape[-1] == 3:
                # If the user provided an image in RGB order, convert it to BGR for OpenCV.
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif arr.shape[-1] == 4:
                # For a 4-channel image, assume it is in RGBA order.
                # Convert RGBA to BGRA so that the alpha channel is preserved.
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        
        # Append any extra kwargs as key/value pairs.
        extra_params = []
        for key, value in kwargs.items():
            extra_params.extend([key, value])
        # Write the image with the combined parameters.
        cv2.imwrite(filename, arr, params + extra_params)
        
        

def imsave(filename, arr):
    io_logger.warning('WARNING: imsave is deprecated, use io.imwrite instead')
    return imwrite(filename, arr)

# now allows for any extension(s) to be specified, allowing exclusion if necessary, non-image files, etc. 
def get_image_files(folder, mask_filter='_masks', img_filter='', look_one_level_down=False,
                    extensions = ['png','jpg','jpeg','tif','tiff'], pattern=None):
    """ find all images in a folder and if look_one_level_down all subfolders """
    mask_filters = ['_cp_masks', '_cp_output', '_flows', mask_filter]
    image_names = []
    
    folders = []
    if look_one_level_down:
        folders = natsorted(glob.glob(os.path.join(folder, "*",'')))  
    folders.append(folder)

    for folder in folders:
        for ext in extensions:
            image_names.extend(glob.glob(folder + ('/*%s.'+ext)%img_filter))
    
    image_names = natsorted(image_names)
    imn = []
    for im in image_names:
        imfile = os.path.splitext(im)[0]
        igood = all([(len(imfile) > len(mask_filter) and imfile[-len(mask_filter):] != mask_filter) or len(imfile) < len(mask_filter) 
                        for mask_filter in mask_filters])
        if len(img_filter)>0:
            igood &= imfile[-len(img_filter):]==img_filter
        if pattern is not None:
            # igood &= bool(re.search(pattern, imfile))
            igood &= bool(re.search(pattern + r'$', imfile))
        if igood:
            imn.append(im)
    image_names = imn

    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')
    
    return image_names


def getname(path,suffix=''):
    return os.path.splitext(Path(path).name)[0].replace(suffix,'')

# I modified this work better with the save_masks function. Complexity added for subfolder and directory flexibility,
# and simplifications made because we can safely assume how output was saved.
# the one place it is needed internally 
def get_label_files(img_names, label_filter='_cp_masks', img_filter='', ext=None,
                    dir_above=False, subfolder='', parent=None, flows=False, links=False):
    """
    Get the corresponding labels and flows for the given file images. If no extension is given,
    looks for TIF, TIFF, and PNG. If multiple are found, the first in the list is returned. 
    If extension is given, no checks for file existence are made - useful for finding nonstandard output like txt or npy. 
    
    Parameters
    ----------
    img_names: list, str
        list of full image file paths
    label_filter: str
        the label filter sufix, defaults to _cp_masks
        can be _flows, _ncolor, etc. 
    ext: str
        the label extension
        can be .tif, .png, .txt, etc. 
    img_filter: str
        the image filter suffix, e.g. _img
    dir_above: bool
        whether or not masks are stored in the image parent folder    
    subfolder: str
        the name of the subfolder where the labels are stored
    parent: str
        parent folder or list of folders where masks are stored, if different from images 
    flows: Bool
        whether or not to search for and return stored flows
    links: bool
        whether or not to search for and return stored link files 
     
    Returns
    -------
    list of all absolute label paths (str)
    
    """

    nimg = len(img_names)
    label_base = [getname(i,suffix=img_filter) for i in img_names]
    
    # allow for the user to specify where the labels are stored, either as a single directory
    # or as a list of directories matching the length of the image list
    if parent is None:
        if dir_above: # for when masks are stored in the directory above (usually in subfolder)
            parent = [Path(i).parent.parent.absolute() for i in img_names]
        else: # for when masks are stored in the same containing forlder as images (usually not in subfolder)
            parent = [Path(i).parent.absolute() for i in img_names]
    
    elif not isinstance(label_folder, list):
        parent = [parent]*nimg
    
    if ext is None:
        label_paths = []
        extensions = ['.tif', '.tiff', '.png', '.npy', '.npz'] #order preference comes here 

        for p,b in zip(parent,label_base):            
            paths = [os.path.join(p,subfolder,b+label_filter+ext) for ext in extensions]
            found = [os.path.exists(path) for path in paths]
            nfound = np.sum(found)
            
            if nfound == 0:
                io_logger.warning('No TIF, TIFF, PNG, NPY, or NPZ labels of type {} found for image {}.'.format(label_filter, b))
            else:
                idx = np.nonzero(found)[0][0]
                label_paths.append(paths[idx])
                if nfound > 1:
                    io_logger.warning("""Multiple labels of type {} also 
                    found for image {}. Deferring to {} label.""".format(label_filter, b, extensions[idx]))
            
        
    else:
        label_paths = [os.path.join(p,subfolder,b+label_filter+ext) for p,b in zip(parent,label_base)]
    
    ret = [label_paths]

    if flows:
        flow_paths = []
        imfilters = ['',img_filter] # this allows both flow name conventions to exist in one folder 

        for p,b in zip(parent,label_base):            
            paths = [os.path.join(p,subfolder,b+imf+'_flows.tif') for imf in imfilters]
            found = [os.path.exists(path) for path in paths]
            nfound = np.sum(found)

            if nfound == 0:
                io_logger.info('not all flows are present, will run flow generation for all images') # this branch should be deprecated 
                flow_paths = None
                break
            else:
                idx = np.nonzero(found)[0][0]
                flow_paths.append(paths[idx])
        
        ret += [flow_paths]
        
    if links:
        link_paths = []
        imfilters = ['',img_filter] # this allows both flow name conventions to exist in one folder 
        
        for p,b in zip(parent,label_base):            
            paths = [os.path.join(p,subfolder,b+'_links.txt') for imf in imfilters]
            found = [os.path.exists(path) for path in paths]
            nfound = np.sum(found)

            if nfound == 0:
                link_paths.append(None)
            else:
                idx = np.nonzero(found)[0][0]
                link_paths.append(paths[idx])
            
        ret += [link_paths]
    return (*ret,) if len(ret)>1 else ret[0]

# edited to allow omni to not read in training flows if any exist; flows computed on-the-fly and code expects this 
# futher edited to automatically find link files for boundary or timelapse flow generation 
def load_train_test_data(train_dir, test_dir=None, image_filter='', mask_filter='_masks', 
                         unet=False, look_one_level_down=True, omni=False, do_links=True):
    """
    Loads the training and optional test data for training runs.
    """
    
    image_names = get_image_files(train_dir, mask_filter, image_filter, look_one_level_down)
    nimg_train = len(image_names)
    images = [imread(image_names[n]) for n in range(nimg_train)]

    label_names, flow_names, link_names = get_label_files(image_names, 
                                                          label_filter=mask_filter, 
                                                          img_filter=image_filter, 
                                                          flows=True, links=True)
    labels = [imread(l) for l in label_names]
    links = [load_links(l) for l in link_names]
    
    if flow_names is not None and not unet and not omni:
        for n in range(nimg_train):
            flows = imread(flow_names[n])
            if flows.shape[0]<4:
                labels[n] = np.concatenate((labels[n][np.newaxis,:,:], flows), axis=0) 
            else:
                labels[n] = flows
            
    # testing data
    nimg_test = 0
    test_images, test_labels, test_links, image_names_test = None,None,[None],None 
    if test_dir is not None:
        image_names_test = get_image_files(test_dir, mask_filter, image_filter, look_one_level_down)
        label_names_test, flow_names_test, link_names_test = get_label_files(image_names_test, 
                                                                            label_filter=mask_filter, 
                                                                            img_filter=image_filter, 
                                                                            flows=True, links=True)
        
        nimg_test = len(image_names_test)
        test_images = [imread(image_names_test[n]) for n in range(nimg_test)]
        test_labels = [imread(label_names_test[n]) for n in range(nimg_test)]
        test_links = [load_links(link_names_test[n]) for n in range(nimg_test)]
        if flow_names_test is not None and not unet:
            for n in range(nimg_test):
                flows = imread(flow_names_test[n])
                if flows.shape[0]<4:
                    test_labels[n] = np.concatenate((test_labels[n][np.newaxis,:,:], flows), axis=0) 
                else:
                    test_labels[n] = flows
    
    # Allow disabling the links even if link files were found 
    if not do_links:
        links = [None]*nimg_train
        test_links = [None]*nimg_test
    
    return images, labels, links, image_names, test_images, test_labels, test_links, image_names_test



def masks_flows_to_seg(images, masks, flows, diams, file_names, channels=None):
    """ save output of model eval to be loaded in GUI 

    can be list output (run on multiple images) or single output (run on single image)

    saved to file_names[k]+'_seg.npy'
    
    Parameters
    -------------

    images: (list of) 2D or 3D arrays
        images input into cellpose

    masks: (list of) 2D arrays, int
        masks output from cellpose_omni.eval, where 0=NO masks; 1,2,...=mask labels

    flows: (list of) list of ND arrays 
        flows output from cellpose_omni.eval

    diams: float array
        diameters used to run Cellpose

    file_names: (list of) str
        names of files of images

    channels: list of int (optional, default None)
        channels used to run Cellpose    
    
    """
    
    if channels is None:
        channels = [0,0]
    
    if isinstance(masks, list):
        if not isinstance(diams, (list, np.ndarray)):
            diams = diams * np.ones(len(masks), np.float32)
        for k, [image, mask, flow, diam, file_name] in enumerate(zip(images, masks, flows, diams, file_names)):
            channels_img = channels
            if channels_img is not None and len(channels) > 2:
                channels_img = channels[k]
            masks_flows_to_seg(image, mask, flow, diam, file_name, channels_img)
        return

    if len(channels)==1:
        channels = channels[0]
    
    flowi = []
    if flows[0].ndim==3:
        Ly, Lx = masks.shape[-2:]
        flowi.append(cv2.resize(flows[0], (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...])
    else:
        flowi.append(flows[0])
    
    if flows[0].ndim==3:
        cellprob = (np.clip(transforms.normalize99(flows[2]),0,1) * 255).astype(np.uint8)
        cellprob = cv2.resize(cellprob, (Lx, Ly), interpolation=cv2.INTER_NEAREST)
        flowi.append(cellprob[np.newaxis,...])
        flowi.append(np.zeros(flows[0].shape, dtype=np.uint8))
        flowi[-1] = flowi[-1][np.newaxis,...]
    else:
        flowi.append((np.clip(transforms.normalize99(flows[2]),0,1) * 255).astype(np.uint8))
        flowi.append((flows[1][0]/10 * 127 + 127).astype(np.uint8))
    if len(flows)>2:
        flowi.append(flows[3])
        flowi.append(np.concatenate((flows[1], flows[2][np.newaxis,...]), axis=0))
    outlines = masks * utils.masks_to_outlines(masks)
    base = os.path.splitext(file_names)[0]
    if masks.ndim==3:
        np.save(base+ '_seg.npy',
                    {'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                        'masks': masks.astype(np.uint16) if outlines.max()<2**16-1 else masks.astype(np.uint32),
                        'chan_choose': channels,
                        'img': images,
                        'ismanual': np.zeros(masks.max(), bool),
                        'filename': file_names,
                        'flows': flowi,
                        'est_diam': diams})
    else:
        if images.shape[0]<8:
            np.transpose(images, (1,2,0))
        np.save(base+ '_seg.npy',
                    {'img': images,
                        'outlines': outlines.astype(np.uint16) if outlines.max()<2**16-1 else outlines.astype(np.uint32),
                     'masks': masks.astype(np.uint16) if masks.max()<2**16-1 else masks.astype(np.uint32),
                     'chan_choose': channels,
                     'ismanual': np.zeros(masks.max().astype(int), bool),
                     'filename': file_names,
                     'flows': flowi,
                     'est_diam': diams})    

def save_to_png(images, masks, flows, file_names):
    """ deprecated (runs io.save_masks with png=True) 
    
        does not work for 3D images
    
    """
    save_masks(images, masks, flows, file_names, png=True)

# Now saves flows, masks, etc. to separate folders.
def _save_single_mask(image, mask, flow, file_name, png=True, tif=False,
                     suffix='', dir_above=None, save_flows=False, save_outlines=False, 
                     save_ncolor=False, save_txt=True, in_folders=False, savedir=None, 
                     save_jpeg=False, save_tif8bit=True, save_fol=True, channels=[0,0], 
                     verbose=False, primariesettings=''):
    """Save a single mask without recursion"""
    if mask is None:
        io_logger.warning(f"Cannot save None mask for {file_name}")
        return
        
    # Ensure mask is 2D
    if mask.ndim != 2:
        io_logger.warning(f"Mask has unexpected dimensions {mask.shape}, squeezing")
        mask = mask.squeeze()
        if mask.ndim != 2:
            io_logger.warning(f"After squeezing, mask still has {mask.ndim} dimensions")
            if mask.ndim > 2:
                # Take first slice or last dimension if it's 1
                if mask.shape[-1] == 1:
                    mask = mask[..., 0]
                else:
                    mask = mask[0]
    
    base = os.path.splitext(file_name)[0]
    basename = os.path.basename(base)
    dirname = os.path.dirname(base)
    if savedir is not None:
        dirname = savedir

    if in_folders and basename != '':
        dirname = os.path.join(dirname, basename)
        basename = ''
    
    if dir_above is not None:
        if isinstance(dir_above, str):
            dirname = os.path.join(dir_above, os.path.split(dirname)[1])
        else:
            dirname = dir_above
        
    # make save directory if it doesn't exist
    os.makedirs(dirname, exist_ok=True)

    # save masks
    outlines = None
    if png or save_jpeg or save_outlines or save_ncolor:
        outlines = masks_to_outlines(mask)
    
    # save txt file
    if save_txt:
        name = basename + suffix + '.txt'
        path = os.path.join(dirname, name)
        with open(path, 'w') as f:
            if primariesettings:
                f.write(primariesettings + '\n')

    # save outlines
    if save_outlines:
        name = basename + suffix + '_cp_outlines.png'
        path = os.path.join(dirname, name)
        imwrite(path, outlines)
    
    # Save masks as png
    if png:
        name = basename + suffix + '_cp_masks.png'
        path = os.path.join(dirname, name)
        imwrite(path, mask)
    
    # Save as jpeg
    if save_jpeg:
        name = basename + suffix + '_cp_masks.jpg'
        path = os.path.join(dirname, name)
        imwrite(path, mask)
    
    # save masks as tif
    if tif:
        name = basename + suffix + '_cp_masks.tif'
        path = os.path.join(dirname, name)
        imwrite(path, mask.astype(np.uint16))

    if save_tif8bit:
        name = basename + suffix + '_cp_masks_8bit.tif'
        path = os.path.join(dirname, name)
        imwrite(path, mask.astype(np.uint8))
    
    # save ncolor masks
    if save_ncolor and OMNI_INSTALLED:
        ncol = plot.apply_ncolor(mask)
        name = basename + suffix + '_cp_ncolor_masks.png'
        path = os.path.join(dirname, name)
        imwrite(path, (ncol * 255).astype(np.uint8))

    # save RGB flow
    if save_flows and flow is not None and flow.shape[0]>0:
        name = basename + suffix + '_cp_flows.tif'
        imwrite(os.path.join(dirname, name), flow.astype(np.float32))
    
    return

def save_masks(images, masks, flows, file_names, png=True, tif=False,
               suffix='', dir_above=None, save_flows=False, save_outlines=False, 
               save_ncolor=False, save_txt=True, in_folders=False, savedir=None, 
               save_jpeg=False, save_tif8bit=True, save_fol=True, channels=[0,0], 
               verbose=False, primariesettings=''):
    """Save masks to disk
    
    Parameters
    --------------
    
    images: list or array of ND-arrays
        images input into cellpose/omnipose
        
    masks: list or array of ND-arrays 
        masks output from cellpose/omnipose
        
    flows: list or array of ND-arrays 
        flows output from cellpose
    
    file_names: list or array of strings
        file names of images
    
    PNG options
    -------------------
    
    png: bool
        whether to save masks as PNG
        
    save_outlines: bool
        whether to save outlines of masks as PNG
        
    save_ncolor: bool
        whether to save masks as unique-colored PNG

    TIFF options
    -------------------

    tif: bool 
        whether to save masks as 16-bit TIFF

    save_tif8bit: bool
        whether to save masks as 8-bit TIFF

    save_flows: bool
        whether to save flows as TIFF
        
    save_jpeg: bool
        whether to save masks as JPEG
        
    file naming options
    ------------------
    suffix: str
        suffix for saved masks
        
    dir_above: str
        directory to save masks, defaults to image directory
        
    in_folders: bool
        whether to save masks in separate folders
        
    save_txt: bool 
        whether to save _seg.npy file as well
        
    savedir: str
        absolute path where images will be saved. Default is None
    """
    
    # Handle the case where masks is a single array
    if isinstance(masks, np.ndarray) and not isinstance(masks, list):
        if len(masks.shape) > 3:
            # If masks has more than 3 dimensions (likely a batch)
            io_logger.info(f"Converting single {masks.ndim}D array of masks to list")
        masks = list(masks)
    
    # Handle length mismatches
    if len(masks) != len(images):
        io_logger.warning(f"Length mismatch: {len(masks)} masks vs {len(images)} images")
        n_items = min(len(masks), len(images))
        masks = masks[:n_items]
        images = images[:n_items]
        file_names = file_names[:n_items]
    
    # Normalize flows to have same length as masks
    if flows is not None:
        if isinstance(flows, list) and len(flows) != len(masks):
            if len(flows) == 1:
                flows = flows * len(masks)
            else:
                flows = flows[:len(masks)]
    else:
        flows = [None] * len(masks)
    
    # Process each mask individually with non-recursive helper function
    for i, (image, mask, flow, file_name) in enumerate(zip(images, masks, flows, file_names)):
        try:
            _save_single_mask(
                image, mask, flow, file_name,
                png=png, tif=tif, suffix=suffix, dir_above=dir_above,
                save_flows=save_flows, save_outlines=save_outlines,
                save_ncolor=save_ncolor, save_txt=save_txt,
                in_folders=in_folders, savedir=savedir,
                save_jpeg=save_jpeg, save_tif8bit=save_tif8bit,
                save_fol=save_fol, channels=channels,
                verbose=verbose, primariesettings=primariesettings
            )
        except Exception as e:
            io_logger.error(f"Error saving mask {i} ({file_name}): {e}")
            continue
            
    # Clean up memory
    try:
        from omnipose.memory import force_cleanup
        force_cleanup(verbose=False)
    except ImportError:
        io_logger.debug("omnipose.memory.force_cleanup not available")

def save_server(parent=None, filename=None):
    """ Uploads a *_seg.npy file to the bucket.
    
    Parameters
    ----------------

    parent: PyQt.MainWindow (optional, default None)
        GUI window to grab file info from

    filename: str (optional, default None)
        if no GUI, send this file to server

    """
    if parent is not None:
        q = QtGui.QMessageBox.question(
                                    parent,
                                    "Send to server",
                                    "Are you sure? Only send complete and fully manually segmented data.\n (do not send partially automated segmentations)",
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No
                                  )
        if q != QtGui.QMessageBox.Yes:
            return
        else:
            filename = parent.filename

    if filename is not None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        'key/cellpose-data-writer.json')
        bucket_name = 'cellpose_data'
        base = os.path.splitext(filename)[0]
        source_file_name = base + '_seg.npy'
        io_logger.info(f'sending {source_file_name} to server')
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
        filestring = time + '.npy'
        io_logger.info(f'name on server: {filestring}')
        destination_blob_name = filestring
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        io_logger.info(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

from collections import defaultdict
def delete_old_models(directory, keep_last=10):
    # Dictionary to store lists of files by prefix
    files_by_prefix = defaultdict(list)

    # Create a search pattern for the model files
    pattern = os.path.join(directory, "*_epoch_*")
    
    # Get a list of all model files matching the pattern
    model_files = glob.glob(pattern)
    
    # Organize files by prefix
    for file in model_files:
        # Extract prefix (everything before '_epoch_')
        base_name = os.path.basename(file)
        prefix = base_name.split('_epoch_')[0]
        files_by_prefix[prefix].append(file)
    
    # Process each prefix
    for prefix, files in files_by_prefix.items():
        # Sort files by epoch number
        files.sort(key=lambda x: int(x.split('_epoch_')[-1]))
        
        # Determine how many files to delete
        num_files_to_delete = len(files) - keep_last
        
        # If there are files to delete, proceed
        if num_files_to_delete > 0:
            files_to_delete = files[:num_files_to_delete]

            for file in files_to_delete:
                os.remove(file)
                print(f"Deleted {file}")
        else:
            print(f"No files to delete for prefix '{prefix}', fewer than or equal to the last {keep_last} files are already present.")
        
    print("Deletion process completed.")
    
    
import platform

def adjust_file_path(file_path):
    """
    Adjust the file path based on the operating system.
    On macOS, replace '/home/user' with '/Volumes'.
    On Linux, replace '/Volumes' with the home directory path.

    Args:
        file_path (str): The original file path.

    Returns:
        str: The adjusted file path.
    """
    system = platform.system()
    if system == 'Darwin':  # macOS
        adjusted_path = re.sub(r'^/home/\w+', '/Volumes', file_path)
    elif system == 'Linux':  # Linux
        home_dir = os.path.expanduser('~')
        adjusted_path = re.sub(r'^/Volumes', home_dir, file_path)
    else:
        raise ValueError(f"Unsupported operating system: {system}")
    return adjusted_path