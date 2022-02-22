import pygame as pg
import numpy as np

SCREEN_W, SCREEN_H = 800, 600
RED, GREEN, BLUE = [255,0,0], [0,255,0], [0,0,255]

pg.init()
screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
running = True
triangle = np.random.randint(0, SCREEN_H/2, (3,2))
offset = np.asarray([[int(SCREEN_W/2),0], [int(SCREEN_W/2),0], [int(SCREEN_W/2),0]])

texture = pg.surfarray.array3d(pg.image.load('finfet.png'))
texture_uv = np.asarray([[0.5,0], [1,0.85], [0,0.85]])
text_size = [len(texture)-1, len(texture[0])-1]

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT: running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE: running = False
            if event.key == pg.K_SPACE:
                triangle = np.random.randint(0, SCREEN_H/2, (3,2))

    frame = np.zeros((SCREEN_W, SCREEN_H, 3)).astype('uint8')

    sorted_y = triangle[:,1].argsort()
    x_start, y_start = triangle[sorted_y[0]]
    x_middle, y_middle = triangle[sorted_y[1]]
    x_stop, y_stop = triangle[sorted_y[2]]
    
    x_slope_1 = (x_stop - x_start)/(y_stop - y_start + 1e-16)
    x_slope_2 = (x_middle - x_start)/(y_middle - y_start + 1e-16)
    x_slope_3 = (x_stop - x_middle)/(y_stop - y_middle + 1e-16)

    uv_start = texture_uv[sorted_y[0]]
    uv_middle = texture_uv[sorted_y[1]]
    uv_stop = texture_uv[sorted_y[2]]

    uv_slope_1 = (uv_stop - uv_start)/(y_stop - y_start + 1e-16)
    uv_slope_2 = (uv_middle - uv_start)/(y_middle - y_start + 1e-16)
    uv_slope_3 = (uv_stop - uv_middle)/(y_stop - y_middle + 1e-16)

    for y in range(y_start, y_stop):
        
        x1 = x_start + int((y-y_start)*x_slope_1)
        uv1 = uv_start + (y-y_start)*uv_slope_1

        if y < y_middle:
            x2 = x_start + int((y-y_start)*x_slope_2)
            uv2 = uv_start + (y-y_start)*uv_slope_2
            # color = GREEN
        else:
            x2 = x_middle + int((y-y_middle)*x_slope_3)
            uv2 = uv_middle + (y-y_middle)*uv_slope_3
            # color = BLUE
        
        if x1 > x2:
            x1, x2 = x2, x1
            uv1, uv2 = uv2, uv1
        
        uv_slope = (uv2 - uv1)/(x2 - x1 +1e-16)
        
        # frame[x1:x2,y] = color
        for x in range(x1, x2):
            uv = uv1 + (x - x1)*uv_slope
            frame[x, y] = texture[int(uv[0]*text_size[0])][int(uv[1]*text_size[1])]

    surf = pg.surfarray.make_surface(frame)   
    pg.draw.polygon(surf, [255, 0, 0], triangle + offset)
    screen.blit(surf, (0,0)); pg.display.update()

pg.quit()
