#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:16:56 2018

@author: alexeedm
"""

# https://raw.githubusercontent.com/mikedh/trimesh/master/examples/offscreen_render.py

import sys
import numpy as np
import trimesh
import glob



if __name__ == '__main__':
    # print logged messages
    #trimesh.util.attach_to_log()

    def draw(Re, kappa, lbd, Ca, y):
    # load a mesh
        files = sorted(glob.glob('/home/alexeedm/extern/daint/project/alexeedm/focusing_soft/' +
                                 'newcase_' + str(int(Re)) + '_' + str(kappa) +
                                 '/case_*_' + str(lbd) + '_' + str(Ca) + '*__' + str(y) + '/ply/capsule_*.ply'))
    
        fname = 'shape_'  + str(int(Re)) + '_' + str(kappa) + '_' + str(lbd) + '_' + str(Ca) + '__' + str(y) + '.png'
        print(fname)
        
        try:
            mesh = trimesh.load(files[-1])
        except:
            print('Skipping!')
            return
        
        scene = mesh.scene()
        scene.set_camera()
        
        png = scene.save_image(resolution=[800, 600],
                               visible=True)
        
        with open('soft_circle_shapes' + fname, 'wb') as f:
            f.write(png)
            f.close()
            
    
    for Re in [50, 100, 200]:
        for kappa in [0.15, 0.22, 0.3]:
            for lbd in [1.0, 5.0, 25.0]:
                for Ca in [1.0, 0.2, 0.05, 0.01]:
                    for y in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                        draw(Re, kappa, lbd, Ca, y)
    
    
    #mesh.show(smooth=True)

#    # get a scene object containing the mesh, this is equivalent to:
#    # scene = trimesh.scene.Scene(mesh)
#    scene = mesh.scene()
#
#    # add a base transform for the camera which just
#    # centers the mesh into the FOV of the camera
#    scene.set_camera()
#
#    # a 45 degree homogenous rotation matrix around
#    # the Y axis at the scene centroid
#    rotate = trimesh.transformations.rotation_matrix(
#        angle=np.radians(45.0),
#        direction=[0, 1, 0],
#        point=scene.centroid)
#
#    for i in range(4):
#        trimesh.constants.log.info('Saving image %d', i)
#
#        # rotate the camera view transform
#        camera_old, _geometry = scene.graph['camera']
#        camera_new = np.dot(camera_old, rotate)
#
#        # apply the new transform
#        scene.graph['camera'] = camera_new
#
#        # saving an image requires an opengl context, so if -nw
#        # is passed don't save the image
#        try:
#            # increment the file name
#            file_name = 'render_' + str(i) + '.png'
#            # save a render of the object as a png
#            png = scene.save_image(resolution=[1920, 1080],
#                                   visible=True)
#            with open(file_name, 'wb') as f:
#                f.write(png)
#                f.close()
#
#        except BaseException as E:
#            print("unable to save image", str(E))