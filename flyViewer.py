# -*- coding: utf-8 -*-
"""
Simple example of loading UI template created with Qt Designer.

This example uses uic.loadUiType to parse and load the ui at runtime. It is also
possible to pre-compile the .ui file using pyuic (see VideoSpeedTest and 
ScatterPlotSpeedTest examples; these .ui files have been compiled with the
tools/rebuildUi.py script).
"""
#import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui#QStringList,QString
import numpy as np
import os

pg.mkQApp()

## Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'flyViewer.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)

import tifffile
import numpy as np

import muscle_model as mm

default_rframe_data = {'a1': np.array([ 51.5848967 ,  -5.93928407]),
                       'a2': np.array([ -0.09151179,  88.42505672]),
                       'p': np.array([ 26.66908747,  34.43488385])}

#stacked_muscles = tifffile.TiffFile('stacked_muscles.tiff')
#overlay = np.transpose(stacked_muscles.asarray(),(1,0,2))[:,::-1].astype(np.float32)
def get_muscle_list():
    #line_database = get_line_database()
    muscle_names = ['b1','b2','b3','i1','i2','iii1','iii24','iii3',
                    'hg1','hg2','hg3','hg4','tpd','tpv','ttm']
    muscle_names = sorted(muscle_names)
    #muscle_names = sorted(get_muscle_list(line_name))
    return muscle_names


def fit_to_model(imchunk,model, mode = 'pinv',fit_pix_mask = None,baseline = None):
    import numpy as np
    #im_array = (imchunk-baseline)#/baseline
    im_array = imchunk-baseline#/baseline
    imshape = np.shape(im_array[0])
    im_array = im_array.reshape((-1,imshape[0]*imshape[1]))
    if mode == 'nnls':
        fits = np.empty((np.shape(model)[0],np.shape(im_array)[0]))
        for i,im2 in enumerate(im_array):
            im = im2.copy()
            im[~np.isfinite(im)] = 0
            from scipy.optimize import nnls
            if not(fit_pix_mask is None):
                fits[:,i] = nnls(model[:,fit_pix_mask].T,im[fit_pix_mask])[0]
            else:
                fits[:,i] = nnls(model.T,im)[0]
    else:
        im = im_array
        print np.shape(im_array)
        from numpy.linalg import pinv
        if not(fit_pix_mask is None):
            fits = np.dot(pinv(model[:,fit_pix_mask]).T,im[:,fit_pix_mask].T)
        else:
            fits = np.dot(pinv(model).T,im)
    return fits

#extract the data give the fly_path and 'line_name'

    #return np.hstack(fits),muscles

class MuscleModelView(object):

    def __init__(self,model):
        import copy
        self.model = model
        self.plot_frame = copy.copy(model.frame)
        self.curves = None
        #self.element_list = ['b2', 'b1', 'ttm', 'b3', 'pr', 'nm', 
        #                    'i1', 'iii24', 'A', 'C', 'B', 'E', 'D',
        #                     'G', 'F', 'I', 'H', 'K', 'i2', 'J', 'tpd', 
        #                     'iii1', 'iii3', 'hg2', 'hg3', 'hg1', 'tpv', 
        #                     'DVM1', 'hg4', 'DVM3', 'DVM2']
        self.element_list = ['b2', 'b1', 'ttm', 'b3', 'pr', 'nm', 
                             'i1', 'i2','iii24',  'tpd', 'iii1', 'iii3', 'hg2',
                             'hg3', 'hg1', 'tpv',]
        
    def plot(self,basis,plotobject):
        lines = self.model.coords_from_frame(basis)
        self.curves = list()
        print self.model.lines.keys()
        for element_name, line in lines.items():
            if element_name in self.element_list:
                self.curves.append(plotobject.plot(line[0,:],line[1,:]))

    def update_basis(self,basis):
        lines = self.model.coords_from_frame(basis)
        lines = [l for k,l in lines.items() if k in self.element_list]
        if self.curves:
            for curve,line in zip(self.curves,lines):#lines.values()):
                curve.setData(line[0,:],line[1,:])

    def basis_changed(self,roi):
        pnts = roi.saveState()['points']
        p = np.array(pnts[1])

        a1 = np.array(pnts[0])-p
        a2 = np.array(pnts[2])-p

        self.plot_frame['p'] = p
        self.plot_frame['a1'] = a1
        self.plot_frame['a2'] = a2
        self.update_basis(self.plot_frame)

class RefrenceFrameROI(pg.ROI):
    
    def __init__(self, basis, closed=False, pos=None, **args):
        
        pos = [0,0]
        
        self.closed = closed
        self.segments = []
        pg.ROI.__init__(self, pos, **args)
        
        self.addFreeHandle((basis['p'][0]+basis['a1'][0],basis['p'][1]+basis['a1'][1]))
        self.addFreeHandle((basis['p'][0],basis['p'][1]))
        self.addFreeHandle((basis['p'][0]+basis['a2'][0],basis['p'][1]+basis['a2'][1]))

        for i in range(0, len(self.handles)-1):
            self.addSegment(self.handles[i]['item'], self.handles[i+1]['item'])
            
    def addSegment(self, h1, h2, index=None):
        seg = pg.LineSegmentROI(handles=(h1, h2), pen=self.pen, parent=self, movable=False)
        if index is None:
            self.segments.append(seg)
        else:      #transform image
        #self.transformPlt = pg.PlotItem()
        #self.ui.transformImage.setCentralItem(self.transformPlt)
        #self.transformImage = pg.ImageItem()
        #self.transformPlt.addItem(self.transformImage)
            self.segments.insert(index, seg)
        #seg.sigClicked.connect(self.segmentClicked)
        #seg.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        seg.setZValue(self.zValue()+1)
        for h in seg.handles:
            h['item'].setDeletable(False)
        
    def saveState(self):
        state = pg.ROI.saveState(self)
        state['closed'] = self.closed
        state['points'] = [tuple(h.pos()) for h in self.getHandles()]
        return state

    def setState(self,state):
        pg.ROI.setState(self,state,update = False)
        #state = pg.ROI.saveState(self)
        for h,p in zip(self.getHandles(),state['points']):
            self.movePoint(h,p)

        self.stateChanged(finish=True)
        return state

class MainWindow(TemplateBaseClass):  
    
    def __init__(self):
        TemplateBaseClass.__init__(self)
        self.setWindowTitle('muscle imaging browser')
        # Create the main window
        self.ui = WindowTemplate()
        #initialize the items created in designer
        self.ui.setupUi(self)
        
        #frame view
        self.plt = pg.PlotItem()
        self.ui.frameView.setCentralItem(self.plt)
        self.frameView = pg.ImageItem()
        self.plt.addItem(self.frameView)

        #gama plot
        self.gammaPlt = pg.PlotItem()
        self.ui.gammaPlot.setCentralItem(self.gammaPlt)
        self.ui.gammaSlider.valueChanged.connect(self.gammaChange)
        
        #default gama
        self.gammaf = lambda x: x**1
        self.gammax = np.linspace(0,2,100)
        self.gammaCurve = self.gammaPlt.plot(self.gammax,self.gammaf(self.gammax))

        #timeSeries
        self.timeSeriesPlt = pg.PlotItem()
        self.ui.timeSeriesPlt.setCentralItem(self.timeSeriesPlt)
        #self.tserTrace = self.timeSeriesPlt.plot(np.ones(1000))
        self.tpointLine = pg.InfiniteLine(pos = 0,movable = True)
        self.tpointLine.sigPositionChanged.connect(self.tpointLineMoved)
        self.timeSeriesPlt.addItem(self.tpointLine)

        #load frames button
        self.ui.loadFrames.clicked.connect(self.loadFrames)

        self.ui.applyDemix.clicked.connect(self.extract_signals)
        #save data button
        self.ui.saveFit.clicked.connect(self.saveFit)
        self.ui.loadFit.clicked.connect(self.loadFit)

        ##scroll bar
        self.ui.frameScrollBar.valueChanged.connect(self.frameScrollBar_valueChanged)

        # Contrast/color control
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.frameView)
        self.ui.frameHist.setCentralItem(self.hist)

        #load data
        self.loadModel()
        self.current_frame = 0
        self.show()
        
        #self.ui.commentBox
        self.ui.frameNumber.setText(str(self.current_frame))
        self.ui.frameNumber.textEdited.connect(self.frameInput)

        #addEpoch
        self.epochPlots = dict()
        self.epoch_dict = dict()
        self.ui.newEpoch.clicked.connect(self.newEpoch)
        self.ui.saveEpoch.clicked.connect(self.saveEpoch)

        self.ui.epochStart.textEdited.connect(self.updateEpochFromText)
        self.ui.epochEnd.textEdited.connect(self.updateEpochFromText)

    def add_model_signals(self):
        self.lst_model = QtGui.QStandardItemModel(self.ui.signalView)
        self.ui.signalView.setModel(self.lst_model)
        self.color_dict = dict()
        import shelve
        fname = os.path.join(self.CurrentDirPath,'model_fits.shelve')
        self.signalshelf = shelve.open(fname)
        if len(self.signalshelf.keys()) == 0:
            self.signalshelf['pxmean'] = np.mean(np.mean(self.images,axis = 0),axis = 0)
        print np.shape(self.images)
        #[self.signalshelf.update({str(mname):sig}) for mname,sig in zip(fits,muscles)]
        for n,key in enumerate(self.signalshelf.keys()):                   
            item = QtGui.QStandardItem(key)
            check = 1 if np.random.randint(0, 1) == 1 else 0
            item.setCheckState(check)
            item.setCheckable(True)
            self.lst_model.appendRow(item)
            self.color_dict[key] = 'r'
        self.lst_model.itemChanged.connect(self.on_model_list_changed)

    def on_model_list_changed(self,item):
        # If the changed item is not checked, don't bother checking others
        #if not item.checkState():
        #    return
     
        # Loop through the items until you get None, which
        # means you've passed the end of the list
        i = 0
        item_list = list()
        while self.lst_model.item(i):
            if self.lst_model.item(i).checkState():
                item_list.append(i)
                #return
            i += 1
        skeys = self.signalshelf.keys()
        self.checked_signals = [skeys[i] for i in item_list]
        self.update_tser_plot()
    
    def update_tser_plot(self):
        [self.timeSeriesPlt.removeItem(pitem) for pitem in self.timeSeriesPlt.listDataItems()]
        pitems = [self.timeSeriesPlt.plot(self.signalshelf[skey]/np.nanstd(self.signalshelf[skey]), 
                                            clickable= True,
                                            pen = self.color_dict[skey]) for skey in self.checked_signals]
        print np.shape(self.signalshelf[skey])
        for pitem,nm in zip(pitems,self.checked_signals):
            pitem.curve.setClickable(True)
            pitem.sigClicked.connect(self.traceClicked)
            pitem.mname = nm

    def traceClicked(self,item):
        print item.mname
        print 'here'
        color = pg.QtGui.QColorDialog.getColor()
        self.color_dict[item.mname] = color
        self.update_tser_plot()

    def newEpoch(self):
        name = str(self.ui.epochName.text())
        print name
        if (not(name in self.epoch_dict.keys()) and not(name == '')):
            epoch_range = [self.current_frame,self.current_frame + 100]
            self.epoch_dict[name] = epoch_range
            self.plotEpoch(name)
            ep_plot = self.epochPlots[name]
            sta,stp = ep_plot.getRegion()
            self.ui.epochStart.setText(str(int(sta)))
            self.ui.epochEnd.setText(str(int(stp)))

    def clearEpochs(self):
        for k in self.epoch_dict.keys():
            self.timeSeriesPlt.removeItem(self.epochPlots[k])
            self.epochPlots.pop(k)
            self.epoch_dict.pop(k)

    def plotEpoch(self,k):
        ep = pg.LinearRegionItem(values= self.epoch_dict[k])
        ep.epoch_name = k
        ep.sigRegionChanged.connect(self.updateEpochPlot)
        self.epochPlots[k] = ep
        self.timeSeriesPlt.addItem(ep)
        self.tpointLine.setZValue(ep.zValue()+1)

    def updateEpochPlot(self,ep):
        self.ui.epochName.setText(ep.epoch_name)
        self.updateCurrentEpochState()

    def updateEpochFromText(self):
        k = str(self.ui.epochName.text())
        ep_plot = self.epochPlots[k]
        sta = int(self.ui.epochStart.text())
        stp = int(self.ui.epochEnd.text())
        ep_plot.setRegion((sta,stp))
        self.epoch_dict[k] = [sta,stp]

    def updateCurrentEpochState(self):
        k = str(self.ui.epochName.text())
        ep = self.epoch_dict[k]
        ep_plot = self.epochPlots[k]
        sta,stp = ep_plot.getRegion()
        self.ui.epochStart.setText(str(int(sta)))
        self.ui.epochEnd.setText(str(int(stp)))
        self.epoch_dict[k] = [int(sta),int(stp)]

    def saveEpoch(self):
        #flydir = '%s%s/'%(dba.root_dir,self.current_fly)
        fname = os.path.join(self.CurrentDirPath,'epoch_data.cpkl')
        with open(fname,'wb') as f:
            import cPickle
            cPickle.dump(self.epoch_dict,f)
            print self.epoch_dict

    def frameInput(self,value):
        self.current_frame = int(value)
        self.showFrame()

    def tpointLineMoved(self):
        self.current_frame = int(self.tpointLine.value())
        self.showFrame()

    def gammaChange(self,value):
        gamma = value/50.0
        self.gammaf = lambda x: x**gamma
        #print gamma
        self.gammaCurve.setData(self.gammax,self.gammaf(self.gammax))
        self.showFrame()

    def loadModel(self):
        import cPickle
        f = open('anatomy_outlines.cpkl','rb')
        ###f = open('/media/flyranch/ICRA_2015/model_data.cpkl','rb')
        anatomy_outlines = cPickle.load(f)
        f.close()

        ########################
        #model_keys = []
        e1 = anatomy_outlines['e1']
        e2 = anatomy_outlines['e2']

        muscle_dict = dict()
        for key in anatomy_outlines.keys():
            if not(key in ['e1','e2']):
                muscle_dict[key] = anatomy_outlines[key]

        frame = mm.Frame()
        frame['a2'] = e1[1]-e2[0]
        frame['a1'] = e2[1]-e2[0]
        frame['p'] = e2[0]
        thorax = mm.GeometricModel(muscle_dict,frame)


        self.thorax_view = MuscleModelView(thorax)
        self.roi = RefrenceFrameROI(thorax.frame)
        self.roi.sigRegionChanged.connect(self.thorax_view.basis_changed)
        #self.roi.sigRegionChanged.connect(self.affineWarp)

        self.plt.disableAutoRange('xy')
        
        state = self.roi.getState()
        rf = default_rframe_data
        pnts = [(rf['p'][0]+rf['a1'][0],rf['p'][1]+rf['a1'][1]),
                 (rf['p'][0],rf['p'][1]),
                 (rf['p'][0]+rf['a2'][0],rf['p'][1]+rf['a2'][1])]
        state['points'] = pnts
        self.roi.setState(state)
        self.roi.stateChanged()
        self.plt.addItem(self.roi)

        self.thorax_view.plot(self.thorax_view.plot_frame,self.plt)

    def loadFrames(self):

        self.CurrentTiffFileName = str(QtGui.QFileDialog.getOpenFileName(self, 
                                        'Dialog Title', 
                                        '',
                                        selectedFilter='*.tif'))
        #self.images = np.array(fly_db[fnum]['experiments'].values()[0]['tiff_data']['images'])
        import os
        self.CurrentDirPath = os.path.split(self.CurrentTiffFileName)[0]
        print self.CurrentDirPath
        tfile = tifffile.TiffFile(self.CurrentTiffFileName)
        print 
        self.images = tfile.asarray()

        import cPickle
        #with open('tseries_data.cpkl','rb') as f:
        #    tser_data = cPickle.load(f)
        #    self.tserTrace.setData(tser_data)
        
        try:
            fname = os.path.join(self.CurrentDirPath,'frame_fits.cpkl')
            with open(fname,'rb') as f:
                import cPickle
                basis = cPickle.load(f)
            state = self.roi.getState()
            pnts = [(basis['p'][0]+basis['a1'][0],basis['p'][1]+basis['a1'][1]),
                    (basis['p'][0],basis['p'][1]),
                    (basis['p'][0]+basis['a2'][0],basis['p'][1]+basis['a2'][1])]
            state['points'] = pnts
            self.roi.setState(state)
            self.roi.stateChanged()
            self.ui.commentBox.setPlainText(basis['commentBox'])

        except IOError:
            print 'no file'
            self.ui.commentBox.setPlainText('')

        self.clearEpochs()

        try:
            fname = os.path.join(self.CurrentDirPath,'epoch_data.cpkl')
            with open(fname,'rb') as f:
                import cPickle
                self.epoch_dict = cPickle.load(f)
            for k in self.epoch_dict.keys():
                self.plotEpoch(k)
            self.ui.epochName.setText(self.epoch_dict.keys()[0])
            self.updateCurrentEpochState()
        except IOError:
            print 'no epoch file'
            self.ui.epochName.setText('')
            self.ui.epochStart.setText('')
            self.ui.epochEnd.setText('')

        #self.frameView.setImage(self.images[0,:,:])
        self.current_frame = 0
        self.showFrame()
        #self.transformImage.setImage(self.transform_img.astype(np.float32))
        self.ui.frameScrollBar.setMaximum(np.shape(self.images)[0])
        self.plt.autoRange()
        #set transformImage
        self.add_model_signals()

    def showFrame(self):
        img = self.gammaf(self.images[self.current_frame,:,:].astype(np.float32))
        self.frameView.setImage(img.astype(np.float32))
        self.ui.frameNumber.setText(str(self.current_frame))
        self.ui.frameScrollBar.setValue(self.current_frame)
        self.tpointLine.setValue(self.current_frame)

    def frameScrollBar_valueChanged(self,value):
        #self.frameView.setImage(self.images[value,:,:])
        self.current_frame = value
        self.showFrame()
        
    def saveFit(self):
        import cPickle
        savedata = dict(self.thorax_view.plot_frame)
        comment_text = self.ui.commentBox.toPlainText()
        savedata['commentBox'] = comment_text
        fname = os.path.join(self.CurrentDirPath,'frame_fits.cpkl')
        with open(fname,'wb') as f:
            cPickle.dump(savedata,f)

    def loadFit(self):
        pass
        #print self.ui.fileTree.selectedItems()[0].data(0,QtCore.Qt.UserRole).toPyObject()

    def extract_signals(self):
        import muscle_model as mm
        import numpy as np
        import h5py
        import cv2
        model_type = 'volumetric'
        #model_type = 'masks'
        #load the reference frame of the cofocal data and that of the imaged fly
        confocal_model = mm.GeometricModel(filepath = 'anatomy_outlines.cpkl')
        #confocal_view = mm.ModelViewMPL(confocal_model)
        pkname = os.path.join(self.CurrentDirPath,'frame_fits.cpkl')
        fly_frame = mm.Frame();fly_frame.load(pkname)
        #get the transformation matrix A and compose with a scaling of s
        #to construct a transformation for homogenious vectors
        s = 1 #scale
        A = fly_frame.get_transform(confocal_model.frame)
        Ap = np.dot([[s,0.0,0],[0,s,0],[0,0,1]],A)
        #parse the GMR genotype to get the line name
        #line_name = parse_GMR_genotype(fly.get_genotype())['gal4']
        #get the list of muscles for a given line
        muscles = get_muscle_list()
        muscles = [m for m in muscles if not('DVM' in m) and not('DLM' in m) and not('ps' in m)]
        #get a reference to the image data
        #fly_record = h5py.File(fly.fly_path + 'fly_record.hdf5','r')
        #exp_record = fly_record['experiments'].values()[0]
        #tfile = tifffile.TiffFile('image_stack.tif')
        #imgs = tfile.asarray()
        imgs = self.images
        #the output shape of the warped model
        output_shape = np.shape(imgs[0])
        if model_type == 'masks':
            #get the mask of all the muscles for fit
            masks = confocal_model.get_masks(fly_frame,np.shape(imgs[0]))
            #create the model using only the muscles that express in a given line
            model = np.vstack([masks[mask_key].T.ravel().astype(float) for mask_key in muscles])
            #construct a mask do reduce the projection to just the data within the model
            fit_pix_mask = np.sum(model,axis=0) > 0
        if model_type == 'volumetric':
            model_data = h5py.File('flatened_model_20x_eo.hdf5','r')
            #muscles = model_data.keys()
            model_muscles = [np.array(model_data[muscle]) for muscle in muscles]
            output_shapes = [output_shape for muscle in muscles]
            transforms = [Ap[:-1,:] for muscle in muscles]
            model = map(cv2.warpAffine,model_muscles,transforms,output_shapes)
            model.append(np.ones_like(model[0]))
            muscles.append('bkg')
            model = np.vstack([muscle.T.ravel() for muscle in model])
            fit_pix_mask = np.ones_like(model[0]) > 0

        print muscles
        print np.shape(model)
        fname = os.path.join(self.CurrentDirPath,'epoch_data.cpkl')
        with open(fname,'rb') as f:
            import cPickle
            baseline_range = cPickle.load(f)['baseline_F']

        baseln = np.mean(imgs[baseline_range],axis = 0)
        
        chnk_sz = 100
        num_samps = np.shape(imgs)[0]
        print num_samps
        chunks = [slice(x,x+chnk_sz if x+chnk_sz < num_samps else num_samps) for x in range(0,num_samps,chnk_sz)]
        
        img_chunks = [np.array(imgs[chunk]) for chunk in chunks]
        models = [model for chunk in chunks]
        modes = ['nnls' for chunk in chunks]
        fit_pix_masks = [fit_pix_mask for chunk in chunks]
        baselines = [baseln for chunk in chunks]
        
        fits = map(fit_to_model,img_chunks,models,modes,fit_pix_masks,baselines)
        #fit = fit_to_model(imchunk,model,mode = 'nnls',fit_pix_mask = fit_pix_mask)
        fname = os.path.join(self.CurrentDirPath,'model_fits.cpkl')
        savedict = dict()
        with open(fname,'wb') as f:
            [savedict.update({str(mname):sig}) for sig,mname in zip(np.hstack(fits),muscles)]
            cPickle.dump(savedict,f)

        #import shelve
        #fname = os.path.join(self.CurrentDirPath,'model_fits.shelve')
        #self.signalshelf = shelve.open(fname) 
        print np.shape(np.hstack(fits))
        print muscles
        [self.signalshelf.update({str(mname):sig}) for sig,mname in zip(np.hstack(fits),muscles)]
        self.add_model_signals()

win = MainWindow()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()