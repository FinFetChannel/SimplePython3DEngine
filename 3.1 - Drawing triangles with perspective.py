import pygame as pg
import numpy as np

SCREEN_W, SCREEN_H = 1800, 900
RED, GREEN, BLUE = [255,0,0], [0,255,0], [0,0,255]

def get_slopes(num_start, num_middle, num_stop, den_start, den_middle, den_stop):
    slope_1 = (num_stop - num_start)/(den_stop - den_start + 1e-32)
    slope_2 = (num_middle - num_start)/(den_middle - den_start + 1e-32)
    slope_3 = (num_stop - num_middle)/(den_stop - den_middle + 1e-32)

    return slope_1, slope_2, slope_3

pg.init()
screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
running = True

# triangle = np.random.randint(0, SCREEN_H/2, (3,3))
triangle =  np.asarray([[0, SCREEN_H, 1], [int(SCREEN_W/2), SCREEN_H, 3], [int(SCREEN_W/4), 0, 6]])
offset = np.asarray([[int(SCREEN_W/2),0,0], [int(SCREEN_W/2),0,0], [int(SCREEN_W/2),0,0]])

texture = pg.surfarray.array3d(pg.image.load('finfet.png'))

texture_uv = np.asarray([[0,0.85], [1,0.85], [0.5,0]])

text_size = [len(texture), len(texture[0])]

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT: running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE: running = False
            if event.key == pg.K_SPACE: triangle = np.random.randint(0, SCREEN_H/2, (3,3))

    frame = np.zeros((SCREEN_W, SCREEN_H, 3)).astype('uint8')

    # start with perspective correct triangle

    sorted_y = triangle[:,1].argsort()

    x_start, y_start, z_start = triangle[sorted_y[0]]
    x_middle, y_middle, z_middle = triangle[sorted_y[1]]
    x_stop, y_stop, z_stop = triangle[sorted_y[2]]

    x_slope_1, x_slope_2, x_slope_3 = get_slopes(x_start, x_middle, x_stop, y_start, y_middle, y_stop)

    max_z = max(z_start, z_middle, z_stop) #for shading
    # invert z for interpolation
    z_start, z_middle, z_stop = 1/(z_start +1e-32), 1/(z_middle + 1e-32), 1/(z_stop +1e-32)
    z_slope_1, z_slope_2, z_slope_3 = get_slopes(z_start, z_middle, z_stop, y_start, y_middle, y_stop)

    # uv coordinates multiplied by inverted z to account for perspective
    uv_start = texture_uv[sorted_y[0]]*z_start 
    uv_middle = texture_uv[sorted_y[1]]*z_middle
    uv_stop = texture_uv[sorted_y[2]]*z_stop
    uv_slope_1, uv_slope_2, uv_slope_3 = get_slopes(uv_start, uv_middle, uv_stop, y_start, y_middle, y_stop)

    for y in range(max(0, int(y_start)), min(SCREEN_H, int(y_stop+1))):
        delta_y = y - y_start
        x1 = x_start + int(delta_y*x_slope_1)
        z1 = z_start + delta_y*z_slope_1
        uv1 = uv_start + delta_y*uv_slope_1

        if y < y_middle:
            x2 = x_start + int(delta_y*x_slope_2)
            z2 = z_start + delta_y*z_slope_2
            uv2 = uv_start + delta_y*uv_slope_2

        else:
            delta_y = y - y_middle
            x2 = x_middle + int(delta_y*x_slope_3)
            z2 = z_middle + delta_y*z_slope_3
            uv2 = uv_middle + delta_y*uv_slope_3
        
        # lower x should be on the left
        if x1 > x2:
            x1, x2 = x2, x1
            z1, z2 = z2, z1
            uv1, uv2 = uv2, uv1

        uv_slope = (uv2 - uv1)/(x2 - x1 + 1e-32) # + 1e-32 to avoid zero division ¯\_(ツ)_/¯
        z_slope = (z2 - z1)/(x2 - x1 + 1e-32)

        for x in range(max(0, int(x1)), min(SCREEN_W, int(x2+1))):
            z = 1/(z1 + (x - x1)*z_slope + 1e-32) # retrive z
            uv = (uv1 + (x - x1)*uv_slope)*z # multiply by z to go back to uv space
            shade = 1 - z/max_z
            if min(uv) >= 0 and max(uv) < 1: # don't render out of bounds
                frame[x, y] = texture[int(uv[0]*text_size[0])][int(uv[1]*text_size[1])]*shade

    # offset triangle without perspective (affine)
    
    offset_triangle = triangle + offset
    sorted_y = offset_triangle[:,1].argsort()

    x_start, y_start, z_start = offset_triangle[sorted_y[0]]
    x_middle, y_middle, z_middle = offset_triangle[sorted_y[1]]
    x_stop, y_stop, z_stop = offset_triangle[sorted_y[2]]

    x_slope_1 = (x_stop - x_start)/(y_stop - y_start + 1e-32)
    x_slope_2 = (x_middle - x_start)/(y_middle - y_start + 1e-32)
    x_slope_3 = (x_stop - x_middle)/(y_stop - y_middle + 1e-32)

    uv_start = texture_uv[sorted_y[0]]
    uv_middle = texture_uv[sorted_y[1]]
    uv_stop = texture_uv[sorted_y[2]]

    uv_slope_1 = (uv_stop - uv_start)/(y_stop - y_start + 1e-32)
    uv_slope_2 = (uv_middle - uv_start)/(y_middle - y_start + 1e-32)
    uv_slope_3 = (uv_stop - uv_middle)/(y_stop - y_middle + 1e-32)

    for y in range(y_start, y_stop+1):

        x1 = x_start + int((y-y_start)*x_slope_1)
        uv1 = uv_start + (y-y_start)*uv_slope_1

        if y < y_middle:
            x2 = x_start + int((y-y_start)*x_slope_2)
            uv2 = uv_start + (y-y_start)*uv_slope_2

        else:
            x2 = x_middle + int((y-y_middle)*x_slope_3)
            uv2 = uv_middle + (y-y_middle)*uv_slope_3

        
        if x1 > x2:
            x1, x2 = x2, x1
            uv1, uv2 = uv2, uv1

        uv_slope = (uv2 - uv1)/(x2 - x1 +1e-32)

        for x in range(x1, x2+1):
            uv = uv1 + (x - x1)*uv_slope
            u = int(uv[0]*text_size[0])%text_size[0]
            v = int(uv[1]*text_size[1])%text_size[1]
            frame[x, y] = texture[u][v]

    surf = pg.surfarray.make_surface(frame)

    # pg.draw.polygon(surf, RED, triangle + offset)

    screen.blit(surf, (0,0)); pg.display.update()

pg.quit()
