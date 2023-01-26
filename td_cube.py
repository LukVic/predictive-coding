import pygame
import cv2
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from skimage.io import imshow, imread
from skimage.color import rgb2yuv, rgb2hsv, rgb2gray, yuv2rgb, hsv2rgb, gray2rgb
from scipy.signal import convolve2d
#from wand.image import Image

dataset_size = 0

PHASE = 'TEST'
file_str = ''

if PHASE == 'TEST':
    dataset_size = 500
    file_str = 'test.txt'
if PHASE == 'TRAIN':
    dataset_size = 5000
    file_str = 'train.txt'
if PHASE == 'VALID':
    dataset_size = 2000
    file_str = 'valid.txt'

def main():

    path1 = 'train_data/illusion_cube_tmp'
    path2 = 'valid_data/illusion_cube_tmp'
    path3 = 'train_data/cube_tmp'
    path4 = 'valid_data/cube_tmp'

    path5 = 'test_data/illusion_cube_tmp'
    path6 = 'test_data/cube_tmp'
    
    plt_ill_cube(path1)
    #plt_ill_cube(path2)
    #plt_ill_cube(path5)
    plt_cube(path3)
    #plt_cube(path4)
    #plt_cube(path6)


def plt_ill_cube(path):
    path = path
    # Rotate Image By 180 Degree
    size_rands = np.random.randint(450, 550, dataset_size)
    angle_rands = np.random.randint(0, 360, dataset_size)

    save_cube(angle_rands, size_rands, path, 0)

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e    
    

def plt_cube(path):
    #cube = np.ones((1,1,1), dtype='bool')
    #fig = plt.figure()
 
    # ax.set_facecolor('k')
    # upscale the above voxel image, leaving gaps

    # Shrink the gaps
    #ax.voxels(cube, facecolor='k', edgecolor='w')

    
    # build up the numpy logo
    n_voxels = np.zeros((18, 18, 18), dtype=bool)
    #front side
    n_voxels[0, 0, :] = True
    n_voxels[-1, 0, :] = True
    n_voxels[:, 0, -1] = True
    n_voxels[:, 0, 0] = True

    #left side
    n_voxels[0, -1, :] = True
    n_voxels[0, :, 0] = True
    n_voxels[0, :, -1] = True

    #right side
    n_voxels[-1, :, 0] = True
    n_voxels[-1, :, -1] = True
    n_voxels[-1, -1, :] = True

    #back side
    n_voxels[:, -1, -1] = True
    n_voxels[:, -1, 0] = True

   # n_voxels[1, 0, 2] = True
   # n_voxels[2, 0, 1] = True
    #facecolors = np.where(n_voxels, '#FFD65DC0', '#7A88CCC0')
    #edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
    facecolors = np.where(n_voxels, 'white', 'white')
    edgecolors = np.where(n_voxels, 'white', 'white')
    
    
    filled = np.ones(n_voxels.shape)

    #transparency
    filled[0:-1,1:-1,1:-1] = False
    filled[-1,1:-1,1:-1] = False
    filled[1,-1:0,1] = False
    filled[1:-1,1:-1,0] = False
    filled[1:-1,0,1:-1] = False
    filled[1:-1,-1,1:-1] = False
    filled[1:-1,1:-1,-1] = False

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.001
    y[:, 0::2, :] += 0.001
    z[:, :, 0::2] += 0.001
    x[1::2, :, :] += 0.999
    y[:, 1::2, :] += 0.999
    z[:, :, 1::2] += 0.999

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ##alpha canal
    #alpha_can = np.zeros() 

    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2,shade=False)
    #ax.axis('off')
    ax.grid(False)
    # Get rid of colored axes planes
    # First remove fill
    ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
    ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # Now set color to white (or whatever is "invisible")
    # ax.set_xlim3d(-5, 25)
    # ax.set_ylim3d(-5, 20)
    # ax.set_zlim3d(-5, 18)
    ax.set_xlim3d(-5, 25)
    ax.set_ylim3d(-5, 25)
    ax.set_zlim3d(-5, 25)
    #plt.show()


    #angles1 = np.random.randint(0, 360, 1)
    #angles2 = np.random.randint(0, 360, 1)
    angles1 = [17]
    angles2 = [109]
    path = path
    ax.view_init(angles1[0], angles2[0])
    plt.savefig(path+".jpg")
    
    # Rotate Image By 180 Degree
    size_rands = np.random.randint(950, 1100, dataset_size)
    angle_rands = np.random.randint(0, 360, dataset_size)
    color_inv = np.random.randint(0,1,1)
    save_cube(angle_rands, size_rands, path, 1)


    #ax.axis('off')
    #plt.savefig("cub_data/cube_"+ str(idx)+".png",dpi=100,bbox_inches='tight')


    # for idx, (angle1, angle2) in enumerate(zip(angles1, angles2)):
    #     ax.view_init(angle1, angle2)
    #     plt.pause(.5)
    #     #ax.axis('off')
    #     #plt.savefig("cub_data/cube_"+ str(idx)+".png",dpi=100,bbox_inches='tight')
    #     plt.savefig("cub_data/cube_"+ str(idx)+".png")
    #     img = Image.open("cub_data/cube_"+ str(idx)+".png")
    #     box = (200, 100, 500, 400)
    #     img2 = img.crop(box)
    #     #img = img.rotate(45,fillcolor='black', expand=True)
    #     img2.save("cub_data/cube_cropped_"+ str(idx)+".png")
    # plt.show()

def save_cube(angles, sizes, path, type):
    for idx, (size_rand, angle_rand) in enumerate(zip(sizes, angles)):
        color_inv = random.randint(0,1)
        #size_rand = 450
        # angle = 45
        move_x = 0
        move_y = 0
        label = None
        noise = 0.5

        if size_rand <= 450:
            move_x = random.randint(0,int((500-size_rand)/2))
            move_y = random.randint(0,int((500-size_rand)/2))

        before = "/media/lucas/ADATA SE760/KYR/KSY/Illusory-Contour-Predictive-Networks/"
        before1 = '/home.nfs/vicenluk/KSY/Illusory-Contour-Predictive-Networks/'
        after1 = ".jpg"
        after2 = ".png"
        Original_Image = Image.open(path+after1)
        changed_image1 = Original_Image.rotate(angle_rand,fillcolor='white', expand=True)
        #changed_image1 = changed_image1.resize((size_rand,size_rand), Image.ANTIALIAS)
        
        changed_image1 = changed_image1.resize((size_rand,size_rand), Image.ANTIALIAS)
        box = ((size_rand - 300)/2 + move_x, (size_rand - 300)/2 + move_y, 300 + (size_rand - 300)/2 + move_x, 300 + (size_rand - 300)/2 + move_y)
        changed_image1 = changed_image1.crop(box)
        changed_image1 = sharpeninig(changed_image1)
        #changed_image1 = changed_image1.convert('L')
        if color_inv == 1 : changed_image1 = ImageOps.invert(changed_image1)     
        changed_image1 = changed_image1.resize((32,32), Image.ANTIALIAS)
        box = (0,0,32,32)
        changed_image1 = sharpeninig(changed_image1)

        changed_image1 = changed_image1.convert('L')

        changed_image1 = changed_image1.crop(box)

        changed_image1 = recoloring(changed_image1)

        # plt.imshow(changed_image1)
        # plt.show()

        #changed_image1 = noisy(changed_image1)
        # fig = plt.figure()
        # plt.imshow(changed_image1)
        # plt.show()
        

        if type == 0: label = 0 # illusion
        else: label = 1 #real
        if PHASE == 'TEST':
            full_path = before + path + str(idx) + after2 + " " +str(label)+ " " + "\n"
        else:
            full_path = before + path + str(idx) + after2 + " " +str(label)+ " " + str(noise) + "\n"

        f = open(before+file_str, "a")
        f.write(full_path)
        f.close()

        changed_image1.save(path+str(idx)+after2)
        print("Figure number {0} finished".format(idx))
    print("Whole dataset generated")

def sharpeninig(image):
    image = np.asarray(image)
    image[image <= int(255/2)] = 0
    image[image > int(255/2)] = 255
    image = Image.fromarray(image)
    return image

def recoloring(image):
    color_1 = random.randint(0,255)
    color_2 =  (color_1 + 128) if color_1 < int(255/2) else (color_1 - 128)
    color_2 = color_2 % 255
    image = np.asarray(image)
    if color_1 > color_2:
        image[image <= int(255/2)] = color_2
        image[image > int(255/2)] = color_1
    else:
        image[image <= int(255/2)] = color_1
        image[image > int(255/2)] = color_2
    image = Image.fromarray(image)
    return image

def noisy(image):
    image = np.asarray(image)
    row,col= image.shape
    gauss_noise=np.zeros((row,col),dtype=np.uint8)
    cv2.randn(gauss_noise,128,20)
    gauss_noise=(gauss_noise*0.7).astype(np.uint8)
    noisy=cv2.add(image,(1)*gauss_noise)
    noisy = Image.fromarray(noisy)
    return noisy

if __name__ == "__main__":
    #global win
    main()


        # pygame.init()
    # display = (800,600)
    # win = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)


    # glRotatef(1, 45, 1, 200)
    # # glTranslatef(0,0.0, -5)

    # # idx = 0

    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         pygame.quit()
    #         quit()

   
    
    
    # Cube()
    # pygame.display.flip()
    # x3 = pygame.surfarray.pixels3d(win)
    # array = np.uint8(x3)
    # #print(array[array>0])
    # # im = Image.fromarray(array)
    # # im.show()
    # pygame.image.save(win, "cub_data/cube_illusion.png")
    # pygame.time.wait(1000)
    # idx += 1
    # glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             quit()

    #     glRotatef(1, 1, 1, 1)
        
        
    #     Cube()
    #     pygame.display.flip()
    #     x3 = pygame.surfarray.pixels3d(win)
    #     array = np.uint8(x3)
    #     #print(array[array>0])
    #     # im = Image.fromarray(array)
    #     # im.show()
    #     pygame.image.save(win, "cub_data/cube_illusion.png")
    #     pygame.time.wait(1000)
    #     idx += 1
    #     glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    # vertices= (
#     (1, -1, -1),
#     (1, 1, -1),
#     (-1, 1, -1),
#     (-1, -1, -1),
#     (1, -1, 1),
#     (1, 1, 1),
#     (-1, -1, 1),
#     (-1, 1, 1)
#     )
# edges = (
#     (0,1),
#     (0,3),
#     (0,4),
#     (2,1),
#     (2,3),
#     (2,7),
#     (6,3),
#     (6,4),
#     (6,7),
#     (5,1),
#     (5,4),
#     (5,7)
#     )

# def Cube():
#     glBegin(GL_LINES)
#     for edge in edges:
#         for vertex in edge:
#             glVertex3fv(vertices[vertex])
#     glEnd()
#     return   
