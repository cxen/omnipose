import os, datetime, gc, warnings, glob, shutil, copy
import numpy as np
import cv2
import fastremap 

from .. import utils, plot, transforms, models
from ..io import imread, imwrite, outlines_to_text, logger_setup

logger, log_file = logger_setup()

try:
    from PySide6.QtWidgets import QFileDialog
    GUI = True
except:
    GUI = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False

import ncolor
from omnipose.utils import sinebow
from omnipose import core

GC = True # global toggle for garbage collection

def _init_model_list(parent):
    models.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    parent.model_list_path = os.fspath(models.MODEL_DIR.joinpath('gui_models.txt'))
    parent.model_strings = []
    if not os.path.exists(parent.model_list_path):
        textfile = open(parent.model_list_path, 'w')
        textfile.close()
    else:
        with open(parent.model_list_path, 'r') as textfile:
            lines = [line.rstrip() for line in textfile]
            if len(lines) > 0:
                parent.model_strings.extend(lines)
    
def _add_model(parent, filename=None, load_model=True):
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Add model to GUI"
            )
        filename = name[0]
    fname = os.path.split(filename)[-1]
    try:
        shutil.copyfile(filename, os.fspath(models.MODEL_DIR.joinpath(fname)))
    except shutil.SameFileError:
        pass
    logger.info(f'{filename} copied to models folder {os.fspath(models.MODEL_DIR)}')
    with open(parent.model_list_path, 'a') as textfile:
        textfile.write(fname + '\n')
    parent.ModelChoose.addItems([fname])
    parent.model_strings.append(fname)
    if len(parent.model_strings) > 0:
        # parent.ModelButton.setStyleSheet(parent.styleUnpressed)
        parent.ModelButton.setEnabled(True)
    
    for ind, model_string in enumerate(parent.model_strings[:-1]):
        if model_string == fname:
            _remove_model(parent, ind=ind+1, verbose=False)

    parent.ModelChoose.setCurrentIndex(len(parent.model_strings))
    if load_model:
        # parent.model_choose(len(parent.model_strings))
        parent.model_choose()
        

def _remove_model(parent, ind=None, verbose=True):
    if ind is None:
        ind = parent.ModelChoose.currentIndex()
    if ind > 0:
        ind -= 1
        if verbose:
            logger.info(f'deleting {parent.model_strings[ind]} from GUI')
        parent.ModelChoose.removeItem(ind+1)
        del parent.model_strings[ind]
        custom_strings = parent.model_strings
        if len(custom_strings) > 0:
            with open(parent.model_list_path, 'w') as textfile:
                for fname in custom_strings:
                    textfile.write(fname + '\n')
            parent.ModelChoose.setCurrentIndex(len(parent.model_strings))
        else:
            # write empty file
            textfile = open(parent.model_list_path, 'w')
            textfile.close()
            parent.ModelChoose.setCurrentIndex(0)
            parent.ModelButton.setEnabled(False)
    else:
        print('ERROR: no model selected to delete')

    

def _get_train_set(image_names):
    """ get training data and labels for images in current folder image_names"""
    train_data, train_labels, train_files = [], [], []
    for image_name_full in image_names:
        image_name = os.path.splitext(image_name_full)[0]
        label_name = None
        if os.path.exists(image_name + '_seg.npy'):
            dat = np.load(image_name + '_seg.npy', allow_pickle=True).item()
            masks = dat['masks'].squeeze()
            if masks.ndim==2:
                fastremap.renumber(masks, in_place=True)
                label_name = image_name + '_seg.npy'
            else:
                logger.info(f'_seg.npy found for {image_name} but masks.ndim!=2')
        if label_name is not None:
            train_files.append(image_name_full)
            train_data.append(imread(image_name_full))
            train_labels.append(masks)
    return train_data, train_labels, train_files

def _load_image(parent, filename=None, load_seg=True):
    """ load image with filename; if None, open QFileDialog """
    
    
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load image"
            )
        filename = name[0]
        
    logger.info(f'called _load_image on {filename}')
    manual_file = os.path.splitext(filename)[0]+'_seg.npy'
    load_mask = False
    if load_seg:
        if os.path.isfile(manual_file) and not parent.autoloadMasks.isChecked():
            logger.info(f'segmentation npy file found: {manual_file}')
            _load_seg(parent, manual_file, image=imread(filename), image_file=filename)
            return # exit here, will not go on to load any mask files 
            
        elif os.path.isfile(os.path.splitext(filename)[0]+'_manual.npy'):
            logger.info(f'manual npy file found: {manual_file}')
            manual_file = os.path.splitext(filename)[0]+'_manual.npy'
            _load_seg(parent, manual_file, image=imread(filename), image_file=filename)
            return # likewise exit here, will not go on to load any mask files 
            # should merege this branch with the above? Not sure what use case manual npy is 
            
            
        elif parent.autoloadMasks.isChecked():
            logger.info('loading masks from _masks.tif file')
            mask_file = os.path.splitext(filename)[0]+'_masks'+os.path.splitext(filename)[-1]
            mask_file = os.path.splitext(filename)[0]+'_masks.tif' if not os.path.isfile(mask_file) else mask_file
            load_mask = True if os.path.isfile(mask_file) else False
    else:
        logger.info('not loading segmentation, just the image')
        
        
    # from here, we now will just be loading an image and a mask image file format, not npy 
    try:
        logger.info(f'loading image: {filename}')
        image = imread(filename)
        parent.loaded = True
    except Exception as e:
        print('ERROR: images not compatible')
        print(f'ERROR: {e}')
        

    if parent.loaded:
        logger.info(f'loaded image shape: {image.shape}')
        parent.reset()
        parent.filename = filename
        filename = os.path.split(parent.filename)[-1]
        _initialize_images(parent, image)
        parent.clear_all()
        parent.loaded = True
        parent.enable_buttons()
        if load_mask:
            print('loading masks')
            _load_masks(parent, filename=mask_file)
        # parent.threshslider.setEnabled(False)
        # parent.probslider.setEnabled(False)
            


def _initialize_images(parent, image):
    """ format image for GUI """
    logger.info(f'initializing image, shape {image.shape}')
    parent.onechan=False
    parent.shape = image.shape
    if image.ndim > 3:
        # make tiff Z x channels x W x H
        if image.shape[0]<4:
            # tiff is channels x Z x W x H
            image = np.transpose(image, (1,0,2,3))
        elif image.shape[-1]<4:
            # tiff is Z x W x H x channels
            image = np.transpose(image, (0,3,1,2))
        # fill in with blank channels to make 3 channels
        if image.shape[1] < 3:
            shape = image.shape
            image = np.concatenate((image,
                            np.zeros((shape[0], 3-shape[1], shape[2], shape[3]), dtype=np.uint8)), axis=1)
            if 3-shape[1]>1:
                parent.onechan=True
        image = np.transpose(image, (0,2,3,1))
    elif image.ndim==3:
        if image.shape[0] < 5:
            image = np.transpose(image, (1,2,0))
        if image.shape[-1] < 3:
            shape = image.shape
            #if parent.autochannelbtn.isChecked():
            #    image = normalize99(image) * 255
            image = np.concatenate((image,np.zeros((shape[0], shape[1], 3-shape[2]),dtype=type(image[0,0,0]))), axis=-1)
            if 3-shape[2]>1:
                parent.onechan=True
            image = image[np.newaxis,...]
        elif image.shape[-1]<5 and image.shape[-1]>2:
            image = image[:,:,:3]
            #if parent.autochannelbtn.isChecked():
            #    image = normalize99(image) * 255
            image = image[np.newaxis,...]
    else:
        image = image[np.newaxis,...]
        
    logger.info(f'loaded image shape: {image.shape}')
    
    img_min = image.min() 
    img_max = image.max()
    parent.stack = image
    parent.NZ = len(parent.stack)
    parent.scroll.setMaximum(parent.NZ-1)
    parent.stack = parent.stack.astype(np.float32)
    parent.stack -= img_min
    if img_max > img_min + 1e-3:
        parent.stack /= (img_max - img_min)
    parent.stack *= 255
    if parent.NZ>1:
        logger.info('converted to float and normalized values to 0.0->255.0')
    del image
    
    if GC: gc.collect() # not sure if these are necessary and if they cause a slowdown

    #parent.stack = list(parent.stack)

    if parent.stack.ndim < 4:
        parent.onechan=True
        parent.stack = parent.stack[:,:,:,np.newaxis]
    parent.imask=0
    parent.Ly, parent.Lx = parent.stack.shape[1:3]
    parent.layerz = 0 * np.ones((parent.Ly,parent.Lx,4), 'uint8')
    
    # print(parent.layerz.shape)
    # if parent.autobtn.isChecked():
    #     parent.compute_saturation()
    # elif len(parent.saturation) != parent.NZ:
    #     parent.saturation = []
    #     for n in range(parent.NZ):
    #         parent.saturation.append([0, 255])
    #     parent.slider.setMinimum(0)
    #     parent.slider.setMinimum(100)
    if parent.autobtn.isChecked() or len(parent.saturation)!=parent.NZ:
        parent.compute_saturation()
    parent.compute_scale()
    parent.currentZ = int(np.floor(parent.NZ/2))
    parent.scroll.setValue(parent.currentZ)
    parent.zpos.setText(str(parent.currentZ))
    parent.track_changes = []
    parent.recenter()
    

def _load_seg(parent, filename=None, image=None, image_file=None):
    """ load *_seg.npy with filename; if None, open QFileDialog """
    
    logger.info(f'loading segmentation: {filename}')
    
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load labelled data", filter="*.npy"
            )
        filename = name[0]
    try:
        dat = np.load(filename, allow_pickle=True).item()
        dat['masks'] # test if masks are present
        parent.loaded = True
    except:
        parent.loaded = False
        print('ERROR: not NPY')
        return

    # this puts in some defaults if they are not present in the npy file
    parent.reset()
    
    if image is None:
        logger.info(f'loading image in _load_seg')
        found_image = False
        if 'filename' in dat:
            parent.filename = dat['filename']
            if os.path.isfile(parent.filename):
                parent.filename = dat['filename']
                found_image = True
            else:
                imgname = os.path.split(parent.filename)[1]
                root = os.path.split(filename)[0]
                parent.filename = root+'/'+imgname
                if os.path.isfile(parent.filename):
                    found_image = True
        if found_image:
            try:
                image = imread(parent.filename)
            except:
                parent.loaded = False
                found_image = False
                print('ERROR: cannot find image file, loading from npy')
        if not found_image:
            parent.filename = filename[:-11]
            if 'img' in dat:
                image = dat['img']
            else:
                print('ERROR: no image file found and no image in npy')
                return
    else:
        parent.filename = image_file
    
    if 'X2' in dat:
        parent.X2 = dat['X2']
    else:
        parent.X2 = 0
    # if 'resize' in dat:
    #     parent.resize = dat['resize']
    # elif 'img' in dat:
    #     if max(image.shape) > max(dat['img'].shape):
    #         parent.resize = max(dat['img'].shape)
    # else:
    #     parent.resize = -1
    
    logger.info(f'loading image in _load_seg with shape {image.shape}')
    _initialize_images(parent, image)
    
    if 'chan_choose' in dat:
        parent.ChannelChoose[0].setCurrentIndex(dat['chan_choose'][0])
        parent.ChannelChoose[1].setCurrentIndex(dat['chan_choose'][1])
    
    
    # Transfer fields from dat to parent directly
    exclude = ['runstring', 'img'] # these are not to be transferred because their formats are different when saved vs in the GUI
    for key, value in dat.items():
        if key not in exclude:
            setattr(parent, key, value)
            print('setting',key)
            
    if 'runstring' in dat:
        parent.runstring.setPlainText(dat['runstring'])
    
    if 'outlines' in dat:
        parent.bounds = parent.outlines = dat['outlines']

    # print('A', parent.masks.shape, parent.coords[0].shape, 
    #       parent.affinity_graph.shape, parent.boundary.shape)
    
    # fix formats using -1 as background
    if parent.masks.min()==-1:
        logger.warning('-1 found in masks, running formatting')
        parent.masks = ncolor.format_labels(parent.masks)
        
    
    parent.initialize_seg()
    
    # Update masks and outlines to ZYX format stored as parent.cellpix and parent.outpix
    if parent.masks.ndim == 2:
        parent.cellpix = parent.masks[np.newaxis, :, :]
        parent.outpix = parent.outlines[np.newaxis, :, :]
        
        
    if not hasattr(parent, 'links'):
        parent.links = None
    
    # we want to initialize the segmentation infrastructure like steps and coords, 
    # this also will create the affinity graph if not present 
    parent.initialize_seg()   
    parent.ncells = parent.masks.max()
    
    # handle colors - I feel like this needs improvement 
    if 'colors' in dat and len(dat['colors'])>=dat['masks'].max(): #== too sctrict, >= is fine 
        colors = dat['colors']
    else:
        colors = parent.colormap[:parent.ncells,:3]
    parent.cellcolors = np.append(parent.cellcolors, colors, axis=0)
    

    if 'est_diam' in dat:
        parent.Diameter.setText('%0.1f'%dat['est_diam'])
        parent.diameter = dat['est_diam']
        parent.compute_scale()
        
    if 'manual_changes' in dat: 
        parent.track_changes = dat['manual_changes']
        logger.info('loaded in previous changes')    
    if 'zdraw' in dat:
        parent.zdraw = dat['zdraw']
    else:
        parent.zdraw = [None for n in range(parent.ncells)]

    # print('dat contents',dat.keys())
    # ['outlines', 'colors', 'masks', 'chan_choose', 'img', 'filename', 'flows', 'ismanual', 'manual_changes', 'model_path', 'flow_threshold', 'cellprob_threshold', 'runstring'])
    
    parent.ismanual = np.zeros(parent.ncells, bool)
    if 'ismanual' in dat:
        if len(dat['ismanual']) == parent.ncells:
            parent.ismanual = dat['ismanual']

    if 'current_channel' in dat:
        logger.info(f'current channel: {dat["current_channel"]}')
        parent.color = (dat['current_channel']+2)%5
        parent.RGBDropDown.setCurrentIndex(parent.color)

    if 'flows' in dat:
        parent.flows = dat['flows']
        
        try:
            if parent.flows[0].shape[-3]!=dat['masks'].shape[-2]:
                Ly, Lx = dat['masks'].shape[-2:]
                
                for i in range[3]:
                    parent.flows[i] = cv2.resize(parent.flows[i].squeeze(), (Lx, Ly), interpolation=cv2.INTER_NEAREST)[np.newaxis,...]

            if parent.NZ==1:
                parent.recompute_masks = True
            else:
                parent.recompute_masks = False
                
        except:
            try:
                if len(parent.flows[0])>0:
                    parent.flows = parent.flows[0]
            except:
                parent.flows = [[],[],[],[],[[]]]
            parent.recompute_masks = False
            

    # Added functionality to jump right back into parameter tuning from saved flows 
    if 'model_path' in dat:
        parent.current_model = dat['model_path']
        # parent.initialize_model() 
    
    
    parent.enable_buttons()
    parent.update_layer()
    logger.info('loaded segmentation, enabling buttons')
    
    
        # important for enabling things  
    parent.loaded = True  
    
    del dat
    if GC: gc.collect()

def _load_masks(parent, filename=None):
    """ load zero-based masks (0=no cell, 1=cell one, ...) """
    if filename is None:
        name = QFileDialog.getOpenFileName(
            parent, "Load masks (PNG or TIFF)"
            )
        filename = name[0]
    logger.info(f'loading masks: {filename}')
    masks = imread(filename)
    parent.masks = masks
    # parent.initialize_seg() # redudnat if initialize_seg is called in _masks_to_gui

    if masks.shape[0]!=parent.NZ:
        print('ERROR: masks are not same depth (number of planes) as image stack')
        return
    
    _masks_to_gui(parent)
    
    # del masks 
    if GC: gc.collect()
    parent.update_layer()
    parent.update_plot()
    
    
def _masks_to_gui(parent, format_labels=False):
    """ masks loaded into GUI """
    masks = parent.masks
    shape = masks.shape 
    ndim = masks.ndim
    # if format_labels:
    #     masks = ncolor.format_labels(masks,clean=True)
    # else:
    #     fastremap.renumber(masks, in_place=True)
    logger.info(f'{parent.ncells} masks found')
    
    # print('calling masks to gui',masks.shape)
    parent.ncells = masks.max() #try to grab the cell count before ncolor


    if parent.ncolor:
        try:
            # Check if masks has any content before calling ncolor
            if parent.ncells > 0:
                masks, ncol = ncolor.label(masks, return_n=True, max_depth=5) 
            else:
                # Handle the case where there are no masks
                ncol = 0
                # Keep masks as is - already empty/zeros
        except Exception as e:
            logger.warning(f"Error applying ncolor: {e}")
            ncol = 0
            # Fallback to basic formatting
            masks = np.reshape(masks, shape)
            masks = masks.astype(np.uint16) if masks.max()<(2**16-1) else masks.astype(np.uint32)
    else:
        masks = np.reshape(masks, shape)
        masks = masks.astype(np.uint16) if masks.max()<(2**16-1) else masks.astype(np.uint32)
        
        # the intrinsic values are masks and bounds, but I will use the old lingo
    # of cellpix and outpix for the draw_layer function expecting ZYX stacks 
    if ndim==2:
        # print('reshaping masks to cellpix stack')
        parent.cellpix = masks[np.newaxis,:,:]
        parent.outpix = parent.bounds[np.newaxis,:,:]



    np.random.seed(42) #try to make a bit more stable 
    
    if parent.ncolor:
        # Approach 1: use a dictionary to color cells but keep their original label
        # Approach 2: actually change the masks to n-color
        # 2 is easier and more stable for editing. Only downside is that exporting will
        # require formatting and users may need to shuffle or add a color to avoid like
        # colors touching 
        # colors = parent.colormap[np.linspace(0,255,parent.ncells+1).astype(int), :3]
        c = sinebow(ncol+1)
        colors = (np.array(list(c.values()))[1:,:3] * (2**8-1) ).astype(np.uint8)

    else:
        colors = parent.colormap[:parent.ncells, :3]
        
    logger.info('creating cell colors and drawing masks')
    parent.cellcolors = np.concatenate((np.array([[255,255,255]]), colors), axis=0).astype(np.uint8)
    
    parent.draw_layer()
    # parent.redraw_masks(masks=parent.masksOn, outlines=parent.outlinesOn) # add to obey outline/mask setting upon recomputing, missing outlines otherwise
    if parent.ncells>0:
        parent.toggle_mask_ops()
    parent.ismanual = np.zeros(parent.ncells, bool)
    parent.zdraw = list(-1*np.ones(parent.ncells, np.int16))
    
    parent.update_layer()
    # parent.update_plot()
    parent.update_shape()
    parent.initialize_seg()
    
    
def _save_png(parent):
    """ save masks to png or tiff (if 3D) """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        if parent.cellpix[0].max() > 65534:
            logger.info('saving 2D masks to tif (too many masks for PNG)')
            imwrite(base + '_cp_masks.tif', parent.cellpix[0])
        else:
            logger.info('saving 2D masks to png')
            imwrite(base + '_cp_masks.png', parent.cellpix[0].astype(np.uint16))
    else:
        logger.info('saving 3D masks to tiff')
        imwrite(base + '_cp_masks.tif', parent.cellpix)

def _save_outlines(parent):
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    if parent.NZ==1:
        logger.info('saving 2D outlines to text file, see docs for info to load into ImageJ')    
        outlines = utils.outlines_list(parent.cellpix[0])
        outlines_to_text(base, outlines)
    else:
        print('ERROR: cannot save 3D outlines')
    

def _save_sets(parent):
    """ save masks to *_seg.npy """
    filename = parent.filename
    base = os.path.splitext(filename)[0]
    flow_threshold, cellprob_threshold = parent.get_thresholds()
    
    # print(parent.cellcolors,'color')
    
    if parent.NZ > 1 and parent.is_stack:
        np.save(base + '_seg.npy',
                {'outlines': parent.outpix,
                 'colors': parent.cellcolors[1:],
                 'masks': parent.cellpix,
                 'current_channel': (parent.color-2)%5,
                 'filename': parent.filename,
                 'flows': parent.flows,
                 'zdraw': parent.zdraw,
                 'model_path': parent.current_model_path if hasattr(parent, 'current_model_path') else 0,
                 'flow_threshold': flow_threshold,
                 'cellprob_threshold': cellprob_threshold,
                 'runstring': parent.runstring.toPlainText()
                 })
    else:
        image = parent.chanchoose(parent.stack[parent.currentZ].copy())
        if image.ndim < 4:
            image = image[np.newaxis,...]
        np.save(base + '_seg.npy',
                {'outlines': parent.outpix.squeeze(),
                 'colors': parent.cellcolors[1:],
                 'masks': parent.cellpix.squeeze(),
                 'chan_choose': [parent.ChannelChoose[0].currentIndex(),
                                 parent.ChannelChoose[1].currentIndex()],
                 'img': image.squeeze(),
                 'filename': parent.filename,
                 'flows': parent.flows,
                 'ismanual': parent.ismanual,
                 'manual_changes': parent.track_changes,
                 'model_path': parent.current_model_path if hasattr(parent, 'current_model_path') else 0,
                 'flow_threshold': flow_threshold,
                 'cellprob_threshold': cellprob_threshold,
                 'runstring': parent.runstring.toPlainText()
                })
    #print(parent.point_sets)
    logger.info('%d RoIs saved to %s'%(parent.ncells, base + '_seg.npy'))
