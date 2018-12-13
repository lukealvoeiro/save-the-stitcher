def getHorizontalBorderCoords(filename_imgA, filename_imgB):
    imgA = skio.imread(filename_imgA)
    imgB = skio.imread(filename_imgB)
    if(imgA.shape != imgB.shape): return False
    rowIndicesToCheck = list(range(imgA.shape[0]))
    imgACoords = np.array(getCoordinatesFromIndices(filename_imgA, rowIndicesToCheck, [imgA.shape[1]]*len(rowIndicesToCheck)))
    imgBCoords = np.array(getCoordinatesFromIndices(filename_imgB, rowIndicesToCheck, [0]*len(rowIndicesToCheck)))
    return imgACoords, imgBCoords

def getVerticalBorderCoords(filename_imgA, filename_imgB):
    imgA = skio.imread(filename_imgA)
    imgB = skio.imread(filename_imgB)
    if(imgA.shape != imgB.shape): return False
    colIndicesToCheck = list(range(imgA.shape[1]))
    imgACoords = np.array(getCoordinatesFromIndices(filename_imgA, [imgA.shape[0]]*len(colIndicesToCheck), colIndicesToCheck))
    imgBCoords = np.array(getCoordinatesFromIndices(filename_imgB, [0]*len(colIndicesToCheck), colIndicesToCheck))
    return imgACoords, imgBCoords

def checkDifference(filename_imgA, filename_imgB, typeDiff='H'):
    """
    Checks if there is a difference between the two files starting and ending points. Make sure images are placed in the correct order
    typeDiff can be either 'H' or 'V' (Horizontal or Vertical)
    """
    if(typeDiff == 'H'): coordsA, coordsB = getHorizontalBorderCoords(filename_imgA, filename_imgB)
    else: coordsA, coordsB = getVerticalBorderCoords(filename_imgA, filename_imgB)
    diff = coordsA - coordsB
    print(np.average(diff, axis=0))


def getCoordinatesFromIndices(filename, x_index, y_index):
    """
    Given a file and indices x and y, get their position in real 
    latitude and longitude coordinates
    """
    if(len(x_index) != len(y_index) or len(y_index) == 0 or len(x_index) == 0):
        return []

    # Read raster
    with rasterio.open(filename, 'r') as src:
        if(src.crs.is_valid):
            # Determine Affine object and CRS projection
            trans = src.transform
            inProj = Proj(src.crs)
            outProj = Proj(init='epsg:4326')

            res = []
            for i in range(len(x_index)):
                curr_x, curr_y = x_index[i], y_index[i]
                # Determines East/Northing 
                x, y = rasterio.transform.xy(trans, curr_x, curr_y)
                # Convert these to latitude / longitude
                tmp = transform(inProj,outProj,x,y)
                res.append(tmp)
            
            return res   

def getIndicesOfCoordinate(filename, x_coord, y_coord):
    """
    Given an image file and latitude and longitude coordinates x and y, 
    get their corresponding indices in the image
    """
    
    if(len(x_coord) != len(y_coord) or len(y_coord) == 0 or len(x_coord) == 0):
        return []
    
    # Read raster
    with rasterio.open(filename, 'r') as src:
        # Determine Affine object and CRS projection
        trans = src.transform
        inProj = Proj(init='epsg:4326')
        outProj = Proj(src.crs)

        res = []
        for i in range(len(x_coord)):
            curr_x, curr_y = x_coord[i], y_coord[i]
            # Determines East/Northing 
            x, y = transform(inProj, outProj, curr_x, curr_y)
            # Convert the point to indices in the image
            tmp = rasterio.transform.rowcol(trans, x, y)
            res.append(tmp)
        
        return res

def transformFiles(files):
    outfile = 'mosaic.vrt'
    outfile2 = 'testProject.tif'
    gdal.BuildVRT(outfile, files)
    opt= gdal.WarpOptions(format = 'GTiff')
    gdal.Warp(outfile2, outfile, options=opt)

def produceRGBtif(filename, countBands):
    """
    Given a image file, combines all the bands into a numPy array
    where each pixel corresponds to a list of n values, where n is the number
    channels in the image provided
    """
    with rasterio.open(filename, 'r') as src:
        base = src.read(1)
        for band_num in range(2, countBands):
            base = np.dstack((base, src.read(band_num)))
        return base



    # 

    # 
    # clouds[(img > 178) & (img < 204)] = 255
    # cloudMask = skc.rgb2gray(clouds) > 0

    # clouds = np.zeros((rows, cols, bands))
    # clouds[(img[:,:,3] > 255*0.35)] = 255
    # cloudMask = cloudMask | (skc.rgb2gray(clouds) > 0)

    # # clouds = np.zeros((rows, cols, bands))
    # derp = (img[:,:,1] - img[:,:,3]) / (img[:,:,1] + img[:,:,3]) # less than 0.4 for clouds
    # plt.imsave('derp.png', derp)








    # plt.imsave('norm_after_mask', non_project_RGB)
    # non_proj = skio.imread("output_nonproj.tif")

    # # output image
    # outputImage("output_nonproj.tif", joined_image)
    # joined_image = rgbAsStackedArray("output_nonproj.tif")
    
    
    # outputClouds("output_clouds_nonproj.tif", cloudMask)
    # plt.imsave('clouds.png', cloudMask)
    
    
    # # get mask that predicts clouds (1 if cloud, 0 else)
    # cloudMask = cloudDetector(res)
    # # join mask and clouds to remove clouds from image
    # resClouds = res & ~cloudMask


        
    
    # # normalized_image = skio.imread("normalized_image.tif")
    # cloudSDS = joined_image[:,:,4].astype(np.uint8) & 0b00001100
    # print(cloudSDS[:3, :3])
    # # cloudMask, convolved = None, None
    
    
    
    # if(convolve):
    #     strel = np.zeros((9,9))
    #     strel[4,4] = 1
    #     strel = skf.gaussian(strel)
    #     convolved = np.zeros_like(joined_image)
    #     for layer in range(joined_image.shape[2]):
    #         convolved[:,:,layer] = scipyf.convolve(joined_image[:,:,layer], strel)
    #         print("Done with layer", layer)
    #     plt.imsave('convolved.png', convolved)
    # else:
    #     convolved = skio.imread('convolved.png')