from pymodis import downmodis
import glob
from osgeo import gdal
import subprocess
import os
import rasterio
import numpy as np 
from pyproj import Proj, transform
import calendar
import glob
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage.color as skc
import scipy.ndimage.filters as scipyf
import skimage.filters as skf
import cv2
import pickle
import getopt
import sys
import datetime
"""
TODO:
- Add date
- add isDownsampled method
"""
IMG_BANDS = [11,13,14]
QA_BANDS = [1]
TILES = ['h14v00', 'h15v00', 'h16v00', 'h17v00', 'h18v00', 'h19v00', 'h20v00', 'h21v00', 'h11v01', 'h12v01', 'h13v01', 'h14v01', 'h15v01', 'h16v01', 'h17v01', 'h18v01', 'h19v01', 'h20v01', 'h21v01', 'h22v01', 'h23v01', 'h24v01']


def main(date, projection_sys, isDownsampled, removeDataFolder, outputMask):
    download_hdf = False
    convert_hdf_to_tif = False
    combine_channels = False
    stitch = True
    new_cloud_mask = True
    
    if(download_hdf): 
        print("Attempting HDF file download")
        downloadFiles(date)
        print("HDF file download complete")
    if(convert_hdf_to_tif):
        print("Attemping to convert HDF files to GeoTIFF")
        process(IMG_BANDS + QA_BANDS)
        print("File conversion complete")
    if(combine_channels): 
        print("Attempting to combine RGB channels")
        combineChannels(TILES, IMG_BANDS)
        print("RBG channels successfully combined")
    if(stitch):
        # stitching image together
        print("Attempting to stitch RGB tiles")
        joined_image = join(True, IMG_BANDS, isStacked=True)
        outputImage("stitched.tif", joined_image)
        print("Tiles successfully stitched")
    else: joined_image = skio.imread("stitched.tif")
    if(new_cloud_mask): 
        print("Attempting to stitch together cloud mask")
        joined_qa = join(False, QA_BANDS, isStacked=False)
        cloudMask = cloudDetector(joined_qa)
        if(outputMask): plt.imsave('cloud_mask.png', cloudMask)

        print("Successfully created cloud mask")

        # cloudMask = cloudDetector(convolved)
        # plt.imsave('cloud_mask.png', cloudMask)
        print("Attempting to apply cloud mask")
        joined_image[cloudMask] = [255, 0, 0]
        outputImage("stitched_masked.tif", joined_image)
        print("Successfully applied cloud mask")

        print("Reprojecting stitched image to Polar Stereographic")
        projection_call = ["gdalwarp", "-t_srs", "EPSG:3414"]
        if(projection_sys != None): projection_call = ["gdalwarp", "-t_srs", "projection_sys"]
        if(isDownsampled): projection_call += ["-tr", "10000", "10000", "-r", "average"]

        subprocess.run(projection_call + ["stitched.tif", "reproj.tif"])
        subprocess.run(projection_call + ["stitched.tif", "reproj_masked.tif"])

    


def downloadFiles(day):
    # Variables for data download
    dest = "data/" # This directory must already exist BTW

    # enddate = "2018-10-11" # The download works backward, so that enddate is anterior to day=
    product = "MOD09GA.006"

    # Instantiate download class, connect and download
    modis_down = downmodis.downModis(destinationFolder=dest, tiles=TILES, user="lukealvoeiro", password="Burnout1", today=day, delta=1, product=product)
    modis_down.connect()
    modis_down.downloadsAllDay()

    # Check that the data has been downloaded
    MODIS_files = glob.glob(dest + '*.hdf')

def process(bands):
    owd = os.getcwd()
    dest = "data/" 
    MODIS_files = [i[5:] for i in glob.glob(dest + '*.hdf')]
    os.chdir(dest)
    getTiffFiles(MODIS_files, bands)    

def getTiffFiles(MODIS_files, bands):
    with open("list_modis_files.txt", "w") as output_file:
        for file in MODIS_files:
            for band in bands:
                sds = gdal.Open(file, gdal.GA_ReadOnly).GetSubDatasets()
                src = gdal.Open(sds[band][0])

                filename =  "band" + str(band) + "_" + file.split(".")[2] +".tif"
                gdal.Translate(filename, src)
                output_file.write(filename + "\n")
    
def combineChannels(tiles, bands):
    """
    Given a list of n files, stacks them on top of each other producing a geotiff with
    the same metadata, but with all bands available.
    """
    
    for file_start in tiles:
        files = []
        for band in bands:
            files = files + glob.glob("data/*" + str(band) + "_" + file_start + '.tif')

        if(len(files) != len(bands)): 
            print("Ya done f***ed up again")
            return
        
        print("Combining...", files)
        with rasterio.open(files[0]) as src0:
            meta = src0.meta

        meta.update(count = len(files))
        # Read each layer and write it to stack
        with rasterio.open("data/stacked_" + file_start + ".tif", 'w', **meta) as dst:
            for id, layer in enumerate(files, start=1):
                with rasterio.open(layer) as src:
                    dst.write_band(id, src.read(1))

def join(isPreprocessed, bands, isStacked):
    rows = list(range(0, 2))
    res = None
    itr = True
    for row in rows:
        files = None
        if(isStacked): files = sorted(glob.glob("data/stacked*v0" + str(row) + '.tif'))
        else: files = sorted(glob.glob("data/band" + str(bands[0]) + "_*v0" + str(row) + '.tif'))
        
        base = None
        if(isPreprocessed): base = processImageBeforeStitching(skio.imread(files[0]))
        else: base = skio.imread(files[0])
        

        for col_img in files[1:]:
            tmp = None
            if(isPreprocessed): tmp = processImageBeforeStitching(skio.imread(col_img))
            else: tmp = skio.imread(col_img)
            base = np.hstack((base, tmp))
        if(itr): 
            if(len(bands) > 1): filler = np.zeros((2400, 7200, len(bands)))
            else: filler = np.zeros((1200, 3600))
            base = np.hstack((base, filler))

            res = np.hstack((filler, base))
            itr = False
        else: res = np.vstack((res, base))
    return res

def outputImage(outputFilename, outputImage, rows = 4800, cols = 33600, inputFilename='mosaic.vrt'):
    numBands = None
    if(len(outputImage.shape) == 2): numBands = 1
    else: numBands = outputImage.shape[2]
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outputFilename, cols, rows, numBands, gdal.GDT_Byte)
    base = gdal.Open(inputFilename)
    outdata.SetGeoTransform(base.GetGeoTransform())
    outdata.SetProjection(base.GetProjection())
    if numBands == 1:
        outdata.GetRasterBand(1).WriteArray(outputImage)
    else:
        for band in range(numBands):
            outdata.GetRasterBand(band+1).WriteArray(outputImage[:,:,band])
    outdata = None
    base = None

def outputQA(outputFilename, outputImage, rows = 2400, cols = 16800, inputFilename='mosaic.vrt'):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outputFilename, cols, rows, 1, gdal.GDT_Int16)
    base = gdal.Open(inputFilename)
    outdata.SetGeoTransform(base.GetGeoTransform())
    outdata.SetProjection(base.GetProjection())
    outdata.GetRasterBand(1).WriteArray(outputImage)
    outdata = None
    base = None

def processImageBeforeStitching(img):
    img[img == -28672] = -100
    img += 101
    img = img / 16100
    # img = (img - img.min()) / (img.max() - img.min())
    img *= 255
    img = img.astype(np.uint8)
    tmp = img[:,:,1]
    img[:,:,1] = img[:,:,2]
    img[:,:,2] = tmp
    return img

def cloudDetector(img):
    # to preserve first two bits
    # img[img == 65535.0] = 4.0
    img = img.astype(np.int16)
    clouds = np.bitwise_and(img, 0b0000000000000011)
    clouds[img == 0b0000000000000001] = 1
    clouds = np.repeat(clouds, 2, axis=0)
    cloudMask = np.repeat(clouds, 2, axis=1)
    cloudMask = cloudMask > 0
    return cloudMask
    


if __name__ == '__main__':
#  try:
    opts, args = getopt.getopt(sys.argv[1:], "p:d:srm")
    opts = dict(opts)
    date = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    projection_sys, isDownsampled, removeDataFolder, outputMask = None, False, False, False
    if("-p" in opts.keys()): projection_sys = opts["-p"]
    if("-d" in opts.keys()): date = opts["-d"]
    if("-s" in opts.keys()): isDownsampled = True
    if("-r" in opts.keys()): removeDataFolder = True
    if("-m" in opts.keys()): outputMask = True

    main(date, projection_sys, isDownsampled, removeDataFolder, outputMask)
    
    # except:
    #     print("Something went wrong!")


