"""This module wraps the Grid/Collection stitching plugin in ImageJ (FIJI)
using the command line headless interface of ImageJ2"""

import os, stat, re
from datetime import datetime
import requests as rq
from zipfile import ZipFile
import subprocess as sp
import errno

class IJ_Stitch:
    def __init__(self, input_dir, output_dir, imagej_path = None, Type = 'Grid: row-by-row', 
                 Order = 'Left & Up', layout_file = None, grid_size_x = 3, grid_size_y = 3,
                 tile_overlap = 20, first_file_index_i = 0, file_names = 'image_{ii}.tif',
                 output_textfile_name = 'TileConfiguration.txt', fusion_method = 'Intensity of random input tile',
                 regression_threshold = 0.3, max_Over_avg_displacement_threshold = 2.50,
                 absolute_displacement_threshold = 3.5, compute_overlap = True,
                 invert_x_coordinates = False, invert_y_coordinates = False, subpixel_accuracy = False,
                 downsample_tiles = False, computation_parameters = 'Save memory (but be slower)', 
                 image_output = 'Write to disk', macroName = None, output_name = None):
        """ input_dir: where the images are located. The TileConfiguration files, if generated,
                will be saved here. 
            output_dir: where the stitched image will be saved. 
            imagej_path: path to imagej linux executables. If None, FIJI will be downloaded. 
            Type: the Type drop-down menu when the plugin is opened. Select from 
                ['Grid: row-by-row', 'Grid: column-by-column', 'Grid: snake by rows', 
                 'Grid: snake by columns', 'Filename defined position', 'Unknown position', 
                 'Positions from file', 'Sequential Images']
                Note that, not necessarily all grid types are implemented.
            Order: the Order drop-down menu when the plugin is opened. Order depends on
                what Type is chosen. For example if Type is 'Grid: row-by-row',
                then order could be 'Left & Up'. Or if Type is 'Positions from file', then
                order could be 'Defined by TileConfiguration'.
            layout_file: if Type is 'Positions from file' and Order is 'Defined by TileConfiguration', 
                the layout file should the TileConfiguration file.
            grid_size_x and grid_size_y: If Type is a grid, these two parameters specify
                the number of tiles in x and y axes.
            tile_overlap: for a grid Type, it specifies the percent overlap between adjacent tiles.
            file_names: a string representing the names of the sequence of images. It should 
                have one variable part, the tile number, shown with i in brackets,
                i.e, {iii} for 3-digit tile number. 
            fusion_method: choose from 'Linear Blending', 'Intensity of random input tile', 
                'Do not fuse images (only write TileConfiguration)', ... (look at the plugin)
            compute_overlap: If False, the tiles will be hard-codedly placed into a grid 
                specified by other parameter. If True, the tile positions will be locally optimized. 
            macroName: the name of the macro. If None, the date and time will be used.
        """
        self.input_dir = input_dir # where the macro will be saved
        self.imagej_path = imagej_path
        self.type = Type
        self.output_dir = output_dir
        self.file_names = file_names
        self.macroname = macroName
        self.output_name = output_name
#         # check if ImageJ file exists or download it
#         if imagej_path is None:
#             print("Downloading FIJI in {}".format(self.input_dir))
#             self.getImageJ(self.input_dir)
#         elif os.path.isdir(imagej_path):
#             print('A directory is given for ImageJ. FIJI will be downloaded there ({}).'.format(imagej_path))
#             self.getImageJ(imagej_path)
#         elif os.path.isfile(imagej_path):
#             if os.path.basename(imagej_path) != 'ImageJ-linux64':
#                 raise FileNotFoundError('Invalid address for ImageJ. The file name needs to be "ImageJ-linux64".')
#             self.imagej_path = imagej_path
        
        if 'Grid' in Type:
            # a grid stitching
            self.typeArgs = self.prepGridStitching(Type, Order, grid_size_x, grid_size_y, tile_overlap, 
                          first_file_index_i, input_dir, file_names, output_textfile_name)
        elif Type == 'Positions from file':
            # stitching using positions from files
            self.typeArgs = self.prepPositionsFromFile(Type, Order, input_dir, layout_file)
        self.commonArgs = self.prepCommons(fusion_method, regression_threshold, max_Over_avg_displacement_threshold,
                    absolute_displacement_threshold, compute_overlap, 
                    invert_x_coordinates, invert_y_coordinates, subpixel_accuracy,
                    downsample_tiles, computation_parameters, image_output, output_dir)
   
#         print(self.typeArgs)
#         print(self.commonArgs)
        
    def run(self):
        if self.macroname is None:
            dtn = datetime.now()
            macroFile = "{0}-{1}-{2}_{3}:{4}:{5}_{6}.ijm".format(dtn.year, dtn.month, dtn.day,
                                                    dtn.hour, dtn.minute, dtn.second, self.type)
            macroFile = os.path.join(self.input_dir, macroFile)
        else:
            macroFile = os.path.join(self.input_dir, self.macroname)
            
        self.saveMacro(macroFile)
        
        shellCommand = []
        if self.imagej_path.startswith('/'):
            shellCommand.append('' + self.imagej_path)
        else:
            shellCommand.append('/' + self.imagej_path)
        shellCommand = shellCommand + ['--ij2'] +  ['--headless'] + ['--console'] + ['-macro'] + [macroFile]

        commandOut = sp.run(shellCommand, stdout = sp.PIPE, stderr = sp.PIPE, text = True)
        try: 
            self.changeOutputName(self.output_dir, self.output_name)
        except OSError as err:
            # Suppress the exception if it is a file not found error.
            # Otherwise, re-raise the exception.
            if err.errno == errno.ENOENT:
                print('The stitching output was not generated.')
        return commandOut
    
    def saveMacro(self, filepath):
        with open(filepath, 'w') as writer:
            writer.write(self.assembleCommand())
    
    def assembleCommand(self):
        macro = """run("Grid/Collection stitching", "{}");""".format(self.typeArgs + ' ' + self.commonArgs)
        return macro 
    
    def prepGridStitching(self, Type, Order, grid_size_x, grid_size_y, tile_overlap, 
                          first_file_index_i, input_dir, file_names, output_textfile_name):
        grids = 'type=[{}]'.format(Type)
        grids = grids + ' ' + 'order=[{}]'.format(Order)
        grids = grids + ' ' + 'grid_size_x={}'.format(grid_size_x)
        grids = grids + ' ' + 'grid_size_y={}'.format(grid_size_y)
        grids = grids + ' ' + 'tile_overlap={}'.format(tile_overlap)
        grids = grids + ' ' + 'first_file_index_i={}'.format(first_file_index_i)
        grids = grids + ' ' + 'directory={}'.format(input_dir)
        grids = grids + ' ' + 'file_names={}'.format(file_names)
        grids = grids + ' ' + 'output_textfile_name={}'.format(output_textfile_name)
        return grids
    
    def prepPositionsFromFile(self, Type, Order, input_dir, layout_file):
        if Order != 'Defined by TileConfiguration':
            raise TypeError('Order {0} is not implemented for Type {1}'.format(Order, Type))
            
        args = 'type=[{}]'.format(Type)
        args = args + ' ' + 'order=[{}]'.format(Order)
        args = args + ' ' + 'directory={}'.format(input_dir)
        args = args + ' ' + 'layout_file={}'.format(layout_file)
        return args
    
    def prepCommons(self, fusion_method, regression_threshold, max_Over_avg_displacement_threshold,
                    absolute_displacement_threshold, compute_overlap, 
                    invert_x_coordinates, invert_y_coordinates, subpixel_accuracy,
                    downsample_tiles, computation_parameters, image_output, output_dir):
        coms = 'fusion_method=[{}]'.format(fusion_method)
        coms = coms + ' ' + 'regression_threshold={}'.format(regression_threshold)
        coms = coms + ' ' + 'max/avg_displacement_threshold={}'.format(max_Over_avg_displacement_threshold)
        coms = coms + ' ' + 'absolute_displacement_threshold={}'.format(absolute_displacement_threshold)
        if compute_overlap:
            coms = coms + ' ' + 'compute_overlap'
        if invert_x_coordinates:
            coms = coms + ' ' + 'invert_x'
        if invert_y_coordinates:
            coms = coms + ' ' + 'invert_y'
        if subpixel_accuracy:
            coms = coms + ' ' + 'subpixel_accuracy'
        if downsample_tiles:
            coms = coms + ' ' + 'downsample_tiles'
            
        # The output settings
        coms = coms + ' ' + 'computation_parameters=[{}]'.format(computation_parameters)
        coms = coms + ' ' + 'image_output=[{}]'.format(image_output)
        coms = coms + ' ' + 'output_directory={}'.format(output_dir)
        return coms

    
    def changeOutputName(self, output_dir, new_name = None):
        if new_name is None:
            new_name = re.sub(r"_[^_]+{[i]+}", '', self.file_names) # removing _FOV{iii} from the name
        os.rename(os.path.join(output_dir, 'img_t1_z1_c1'), os.path.join(output_dir, new_name))
        
    @staticmethod
    def getImageJ(pathOrFile):
        """ Downloads FIJI. If given a directory, will save a FIJI there. 
        If given a file, will save FIJI in the directory of that file"""
        if os.path.isdir(pathOrFile):
            savingDir = pathOrFile
        elif os.path.isdir(os.path.dirname(pathOrFile)):
            savingDir = os.path.dirname(pathOrFile)
        else:
            raise FileNotFoundError('Invalid directory or file for ImageJ: \n {}'.format(pathOrFile))
            
        # downloading FIJI
        ijfile = rq.get('https://downloads.imagej.net/fiji/archive/20200928-2004/fiji-linux64.zip')
        with open(os.path.join(savingDir, 'fiji-linux64.zip'), 'wb') as writer:
            writer.write(ijfile.content)

        # unzipping FIJI
        with ZipFile(os.path.join(savingDir, 'fiji-linux64.zip'), 'r') as unzipper: 
            unzipper.extractall(savingDir)
        
        imagej_path = os.path.join(savingDir, 'Fiji.app', 'ImageJ-linux64')
        
        # changing the permission
        st = os.stat(imagej_path)
        os.chmod(imagej_path, st.st_mode | stat.S_IEXEC)
        
        return imagej_path
