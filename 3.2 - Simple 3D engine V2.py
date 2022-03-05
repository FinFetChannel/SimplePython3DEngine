import pygame as pg
import numpy as np
from numba import njit
from objLoader import read_obj

SCREEN_W, SCREEN_H = 800, 600
FOV_V = np.pi/4 # 45 degrees vertical fov
FOV_H = FOV_V*SCREEN_W/SCREEN_H
SKY_BLUE = np.asarray([50,127,200]).astype('uint8')


def main():
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    running = True
    clock = pg.time.Clock()
    surf = pg.surface.Surface((SCREEN_W, SCREEN_H))
    frame= np.ones((SCREEN_W, SCREEN_H, 3)).astype('uint8')
    z_buffer = np.ones((SCREEN_W, SCREEN_H))

    # points, triangles, texture_uv, texture_map, textured =  read_obj('obj models/teapot.obj')
    # points, triangles, texture_uv, texture_map, textured =  read_obj('obj models/mountains.obj')
    # points, triangles, texture_uv, texture_map, textured =  read_obj('obj models/finfet.obj')
    
    # textured = True
    # points = 10.1*np.asarray([[0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1], 
    #                         [0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1]])
    # triangles = np.asarray([[0,1,2], [0,2,3],[3,2,6], [3,6,7], [1,5,6], [1,6,2], [0,3,7], [0,7,4], [1,0,4], [1,4,5], [6,5,4], [6,4,7]])
    # texture_uv = np.asarray([[0,1], [1,1], [0,0], [1,0]])
    # texture_map = np.asarray([[2,0,1], [2,1,3], [2,0,1], [2,1,3], [2,0,1], [2,1,3], [2,0,1], [2,1,3], [2,0,1], [2,1,3], [2,0,1], [2,1,3],])
    # texture = pg.surfarray.array3d(pg.image.load('finfet.png'))

    # points, triangles, texture_uv, texture_map, textured =  read_obj('obj models/cube text.obj')

    # points, triangles, texture_uv, texture_map, textured =  read_obj('obj models/Babycrocodile.obj')
    # texture = pg.surfarray.array3d(pg.image.load('obj models/BabyCrocodileGreen.png'))

    # points, triangles, texture_uv, texture_map, textured =  read_obj('obj models/cottage_obj.obj')
    # texture = pg.surfarray.array3d(pg.image.load('obj models/cottage_diffuse.png'))

    points, triangles, texture_uv, texture_map, textured =  read_obj('obj models/ah64d.obj')
    texture = pg.surfarray.array3d(pg.image.load('obj models/ah64.png'))

    camera = np.asarray([13, 0.5, 2, 3.3, 0])
    
    pg.mouse.set_visible(0)

    while running:
        pg.mouse.set_pos(SCREEN_W/2, SCREEN_H/2)
        elapsed_time = clock.tick()*0.001

        light_dir = np.asarray([np.sin(pg.time.get_ticks()/1000), 1, 1])
        light_dir = light_dir/np.linalg.norm(light_dir)

        for event in pg.event.get():
            if event.type == pg.QUIT: running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_DELETE:
                camera = np.asarray([13, 0.5, 2, 3.3, 0]) # reset camera
        
        frame[:,:,:] = SKY_BLUE
        z_buffer[:,:] = 1e32 # start with some big value
        
        project_points(points, camera)
        if textured:
            draw_triangles(frame, points, triangles, camera, light_dir, z_buffer, texture_uv, texture_map, texture)
        else:
            draw_flat_triangles(frame, points, triangles, camera, light_dir, z_buffer)
        
        surf = pg.surfarray.make_surface(frame)
        screen.blit(surf, (0,0)); pg.display.update()
        pg.display.set_caption(str(round(1/(elapsed_time+1e-16), 1)) + ' ' + str(camera))
        movement(camera, min(elapsed_time*10, 1))

def movement(camera, elapsed_time):

    if pg.mouse.get_focused():
        p_mouse = pg.mouse.get_pos()
        camera[3] = (camera[3] + 10*elapsed_time*np.clip((p_mouse[0]-SCREEN_W/2)/SCREEN_W, -0.2, .2))%(2*np.pi)
        camera[4] = camera[4] + 10*elapsed_time*np.clip((p_mouse[1]-SCREEN_H/2)/SCREEN_H, -0.2, .2)
        camera[4] = np.clip(camera[4], -1.57, 1.57) # limit to +- 180°
    
    pressed_keys = pg.key.get_pressed()

    if pressed_keys[ord('e')]: camera[1] += elapsed_time
    elif pressed_keys[ord('q')]: camera[1] -= elapsed_time

    if (pressed_keys[ord('w')] or pressed_keys[ord('s')]) and (pressed_keys[ord('a')] or pressed_keys[ord('d')]):
        elapsed_time *= 0.707 # keep speed for diagonals
        
    if pressed_keys[pg.K_UP] or pressed_keys[ord('w')]:
        camera[0] += elapsed_time*np.cos(camera[3])
        camera[2] += elapsed_time*np.sin(camera[3])

    elif pressed_keys[pg.K_DOWN] or pressed_keys[ord('s')]:
        camera[0] -= elapsed_time*np.cos(camera[3])
        camera[2] -= elapsed_time*np.sin(camera[3])
        
    if pressed_keys[pg.K_LEFT] or pressed_keys[ord('a')]:
        camera[0] += elapsed_time*np.sin(camera[3])
        camera[2] -= elapsed_time*np.cos(camera[3])
        
    elif pressed_keys[pg.K_RIGHT] or pressed_keys[ord('d')]:
        camera[0] -= elapsed_time*np.sin(camera[3])
        camera[2] += elapsed_time*np.cos(camera[3])



@njit()
def project_points(points, camera):

    cos_hor = np.cos(-camera[3]+np.pi/2) # add 90° to align with z axis
    sin_hor = np.sin(-camera[3]+np.pi/2) # negative 
    
    cos_ver = np.cos(-camera[4])
    sin_ver = np.sin(-camera[4])

    hor_fov_adjust = 0.5*SCREEN_W/ np.tan(FOV_H * 0.5) 
    ver_fov_adjust = 0.5*SCREEN_H/ np.tan(FOV_V * 0.5)
    
    for point in points:

        # translate to have camera as origin
        translate = point[:3] - camera[:3]

        # rotate to camera horizontal direction
        new_x = translate[0]*cos_hor - translate[2]*sin_hor
        new_z = translate[0]*sin_hor + translate[2]*cos_hor
        translate[0], translate[2] = new_x, new_z

        # rotate to camera vertical direction
        new_y = translate[1]*cos_ver - translate[2]*sin_ver
        new_z = translate[1]*sin_ver + translate[2]*cos_ver
        translate[1], translate[2] = new_y, new_z
        
        if translate[2] <  0.001 and translate[2] >  - 0.001: # jump over 0 to avoid zero division ¯\_(ツ)_/¯
            translate[2] = - 0.001

        point[3] = int(-hor_fov_adjust*translate[0]/translate[2] + 0.5*SCREEN_W)
        point[4] = int(-ver_fov_adjust*translate[1]/translate[2] + 0.5*SCREEN_H)
        point[5] = translate[2] # np.sqrt(translate[0]*translate[0] + translate[1]*translate[1] + translate[2]*translate[2])


@njit()
def dot_3d(arr1, arr2):
    return arr1[0]*arr2[0] + arr1[1]*arr2[1] + arr1[2]*arr2[2]

@njit()
def draw_triangles(frame, points, triangles, camera, light_dir, z_buffer, texture_uv, texture_map, texture):
    text_size = [len(texture)-1, len(texture[0])-1]
    for index in range(len(triangles)):
        
        triangle = triangles[index]

        # Use Cross-Product to get surface normal
        vet1 = points[triangle[1]][:3]  - points[triangle[0]][:3]
        vet2 = points[triangle[2]][:3] - points[triangle[0]][:3]

        # backface culling with dot product between normal and camera ray
        normal = np.cross(vet1, vet2)
        normal = normal/np.sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
        CameraRay = (points[triangle[0]][:3] - camera[:3])/points[triangle[0]][5]

        # get projected 2d points for crude filtering of offscreen triangles
        xxs = [points[triangle[0]][3],  points[triangle[1]][3],  points[triangle[2]][3]]
        yys = [points[triangle[0]][4],  points[triangle[1]][4],  points[triangle[2]][4]]
        z_min = min([points[triangle[0]][5],  points[triangle[1]][5],  points[triangle[2]][5]])

        # check valid values
        if filter_triangles(z_min, normal, CameraRay, xxs, yys):

            shade = 0.5*dot_3d(light_dir, normal) + 0.5 #  directional lighting

            proj_points = points[triangle][:,3:] #
            uv_points = texture_uv[texture_map[index]]

            sorted_y = proj_points[:,1].argsort()

            x_start, y_start, z_start = proj_points[sorted_y[0]]
            x_middle, y_middle, z_middle = proj_points[sorted_y[1]]
            x_stop, y_stop, z_stop = proj_points[sorted_y[2]]

            x_slope_1, x_slope_2, x_slope_3 = get_slopes(x_start, x_middle, x_stop, y_start, y_middle, y_stop)

            # invert z for interpolation
            z_start, z_middle, z_stop = 1/(z_start +1e-32), 1/(z_middle + 1e-32), 1/(z_stop +1e-32)
            z_slope_1, z_slope_2, z_slope_3 = get_slopes(z_start, z_middle, z_stop, y_start, y_middle, y_stop)

            # uv coordinates multiplied by inverted z to account for perspective
            uv_start = uv_points[sorted_y[0]]*z_start 
            uv_middle = uv_points[sorted_y[1]]*z_middle
            uv_stop = uv_points[sorted_y[2]]*z_stop
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

                xx1, xx2 = max(0, min(SCREEN_W-1, int(x1))), max(0, min(SCREEN_W-1, int(x2+1)))
                if xx1 != xx2:
                    uv_slope = (uv2 - uv1)/(x2 - x1 + 1e-32)
                    z_slope = (z2 - z1)/(x2 - x1 + 1e-32)

                    for x in range(xx1, xx2):
                        z = 1/(z1 + (x - x1)*z_slope + 1e-32) # retrive z
                        if z < z_buffer[x][y]: # check z buffer
                            uv = (uv1 + (x - x1)*uv_slope)*z # multiply by z to go back to uv space
                            if min(uv) >= 0 and max(uv) <= 1: # don't render out of bounds
                                z_buffer[x][y] = z
                                frame[x, y] = shade*texture[int(uv[0]*text_size[0])][int(uv[1]*text_size[1])]

@njit()
def get_slopes(num_start, num_middle, num_stop, den_start, den_middle, den_stop):
    slope_1 = (num_stop - num_start)/(den_stop - den_start + 1e-32) # + 1e-32 avoid zero division ¯\_(ツ)_/¯
    slope_2 = (num_middle - num_start)/(den_middle - den_start + 1e-32)
    slope_3 = (num_stop - num_middle)/(den_stop - den_middle + 1e-32)

    return slope_1, slope_2, slope_3

@njit()
def filter_triangles(z_min, normal, CameraRay, xxs, yys): #TODO replace filtering with proper clipping
    if z_min > 0 and dot_3d(normal, CameraRay) < 0   and (
        (xxs[0] >= -SCREEN_W and xxs[0] < 2*SCREEN_W and yys[0] >= -SCREEN_H and yys[0] < 2*SCREEN_H) or
        (xxs[1] >= -SCREEN_W and xxs[1] < 2*SCREEN_W and yys[1] >= -SCREEN_H and yys[1] < 2*SCREEN_H) or
        (xxs[2] >= -SCREEN_W and xxs[2] < 2*SCREEN_W and yys[2] >= -SCREEN_H and yys[2] < 2*SCREEN_H)):
        return True

    else:
        return False

@njit()
def draw_flat_triangles(frame, points, triangles, camera, light_dir, z_buffer):
    color_scale = 230/np.max(np.abs(points[:,:3]))

    for index in range(len(triangles)):
        
        triangle = triangles[index]

        # Use Cross-Product to get surface normal
        vet1 = points[triangle[1]][:3]  - points[triangle[0]][:3]
        vet2 = points[triangle[2]][:3] - points[triangle[0]][:3]

        # backface culling with dot product between normal and camera ray
        normal = np.cross(vet1, vet2)
        normal = normal/np.sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
        CameraRay = (points[triangle[0]][:3] - camera[:3])/points[triangle[0]][5]

        # get projected 2d points for crude filtering of offscreen triangles
        xxs = [points[triangle[0]][3],  points[triangle[1]][3],  points[triangle[2]][3]]
        yys = [points[triangle[0]][4],  points[triangle[1]][4],  points[triangle[2]][4]]
        z_min = min([points[triangle[0]][5],  points[triangle[1]][5],  points[triangle[2]][5]])

        # check valid values
        if filter_triangles(z_min, normal, CameraRay, xxs, yys): 
            
            shade = 0.5*dot_3d(light_dir, normal) + 0.5 # directional lighting
            color = shade*np.abs(points[triangles[index][0]][:3])*color_scale + 25

            proj_points = points[triangle][:,3:]
            sorted_y = proj_points[:,1].argsort()

            x_start, y_start, z_start = proj_points[sorted_y[0]]
            x_middle, y_middle, z_middle = proj_points[sorted_y[1]]
            x_stop, y_stop, z_stop = proj_points[sorted_y[2]]

            x_slope_1, x_slope_2, x_slope_3 = get_slopes(x_start, x_middle, x_stop, y_start, y_middle, y_stop)

            # invert z for interpolation
            z_start, z_middle, z_stop = 1/(z_start +1e-32), 1/(z_middle + 1e-32), 1/(z_stop +1e-32)
            z_slope_1, z_slope_2, z_slope_3 = get_slopes(z_start, z_middle, z_stop, y_start, y_middle, y_stop)

            for y in range(max(0, int(y_start)), min(SCREEN_H, int(y_stop+1))):
                delta_y = y - y_start
                x1 = x_start + int(delta_y*x_slope_1)
                z1 = z_start + delta_y*z_slope_1

                if y < y_middle:
                    x2 = x_start + int(delta_y*x_slope_2)
                    z2 = z_start + delta_y*z_slope_2

                else:
                    delta_y = y - y_middle
                    x2 = x_middle + int(delta_y*x_slope_3)
                    z2 = z_middle + delta_y*z_slope_3
                
                if x1 > x2:
                    x1, x2 = x2, x1 # lower x should be on the left
                    z1, z2 = z2, z1
                    
                xx1, xx2 = max(0, min(SCREEN_W-1, int(x1))), max(0, min(SCREEN_W-1, int(x2+1)))
                if xx1 != xx2:
                    if min(z_buffer[xx1:xx2, y]) == 1e32:
                        z_buffer[xx1:xx2, y] = 2/(z1 + z2)
                        frame[xx1:xx2, y] = color
                    else:
                        z_slope = (z2 - z1)/(x2 - x1 + 1e-32)
                        for x in range(xx1, xx2):
                            z = 1/(z1 + (x - x1)*z_slope + 1e-32) # retrive z
                            if z < z_buffer[x][y]: # check z buffer
                                z_buffer[x][y] = z
                                frame[x, y] = color          
    
if __name__ == '__main__':
    main()
    pg.quit()
