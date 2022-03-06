from re import S
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
    skybox = pg.image.load('skybox0.jpg').convert()
    skybox = pg.transform.smoothscale(skybox, (SCREEN_W*8, SCREEN_H*6))
    skybox = pg.surfarray.array3d(skybox)

    model1 = Model('obj models/teapot.obj', '')
    model1.translate(10, 0, 20)

    model2 = Model('obj models/mountains.obj', '')
    model2.translate(0, -1, 0)

    model3 = Model('obj models/finfet.obj', '')
    model3.translate(-10, 20, -10)

    model4 = Model('obj models/cube text.obj', 'finfet.png')

    model5 = Model('obj models/Babycrocodile.obj', 'obj models/BabyCrocodileGreen.png')
    model5.translate(5, 0.5, 30)

    model6 = Model('obj models/cottage_obj.obj', 'obj models/cottage_diffuse.png')
    model6.translate(-20, 0, 20)

    model7 = Model('obj models/ah64d.obj', 'obj models/ah64.png')
    model7.translate(10, 30, 10)

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
        h1 = int(SCREEN_W*camera[3]/FOV_H)
        v1 = int(SCREEN_H*camera[4]/FOV_V + 3*SCREEN_H)
        frame[:,:,:]  = skybox[h1:h1+SCREEN_W,v1:v1+SCREEN_H,:]
        z_buffer[:,:] = 1e32 # start with some big value
        
        for model in Model._registry:
            project_points(model.points, camera)
            draw_model(frame, model.points, model.triangles, camera, light_dir, z_buffer, model.textured,
                       model.texture_uv, model.texture_map, model.texture)
        
        surf = pg.surfarray.make_surface(frame)
        screen.blit(surf, (0,0)); pg.display.update()
        pg.display.set_caption(str(round(1/(elapsed_time+1e-32), 1)) + ' ' + str(camera))
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

class Model:
    _registry = []

    def __init__(self, path_obj, path_texture):
        self._registry.append(self)
        self.points, self.triangles, self.texture_uv, self.texture_map, self.textured =  read_obj(path_obj)
        if self.textured:
            self.texture = pg.surfarray.array3d(pg.image.load(path_texture))
        else:
            self.texture_uv, self.texture_map = np.ones((2,2)), np.random.randint(1, 2, (2,3))
            self.texture = np.random.randint(0, 255, (10, 10,3))
    
    def translate(self, x, y, z):
        self.points[:,0] = self.points[:,0] + x
        self.points[:,1] = self.points[:,1] + y
        self.points[:,2] = self.points[:,2] + z


@njit()
def project_points(points, camera):

    cos_hor = np.cos(-camera[3]+np.pi/2) # add 90° to align with z axis
    sin_hor = np.sin(-camera[3]+np.pi/2) # negative (counter rotation)
    
    cos_ver = np.cos(-camera[4])
    sin_ver = np.sin(-camera[4])

    hor_fov_adjust = 0.5*SCREEN_W/ np.tan(FOV_H * 0.5) 
    ver_fov_adjust = 0.5*SCREEN_H/ np.tan(FOV_V * 0.5)
    
    # translate to have camera as origin
    points[:,3] = points[:,0] - camera[0]
    points[:,4] = points[:,1] - camera[1]
    points[:,5] = points[:,2] - camera[2]

    points2 = points.copy() # copy for rotations
    
    # rotate to camera horizontal direction
    points2[:,3] = points[:,3]*cos_hor - points[:,5]*sin_hor
    points2[:,5] = points[:,3]*sin_hor + points[:,5]*cos_hor

    # rotate to camera vertical direction
    points[:,4] = points2[:,4]*cos_ver - points2[:,5]*sin_ver
    points[:,5] = points2[:,4]*sin_ver + points2[:,5]*cos_ver
    
    # jump over 0 to avoid zero division ¯\_(ツ)_/¯
    points[:,5][(points[:,5] < 0.001) & (points[:,5] > -0.001)] = -0.001 
    points[:,3] = (-hor_fov_adjust*points2[:,3]/points[:,5] + 0.5*SCREEN_W).astype(np.int32)
    points[:,4] = (-ver_fov_adjust*points[:,4]/points[:,5] + 0.5*SCREEN_H).astype(np.int32)

    # for point in points:

    #     # translate to have camera as origin
    #     translate = point[:3] - camera[:3]

    #     # rotate to camera horizontal direction
    #     new_x = translate[0]*cos_hor - translate[2]*sin_hor
    #     new_z = translate[0]*sin_hor + translate[2]*cos_hor
    #     translate[0], translate[2] = new_x, new_z

    #     # rotate to camera vertical direction
    #     new_y = translate[1]*cos_ver - translate[2]*sin_ver
    #     new_z = translate[1]*sin_ver + translate[2]*cos_ver
    #     translate[1], translate[2] = new_y, new_z
        
    #     if translate[2] <  0.001 and translate[2] >  - 0.001: # jump over 0 to avoid zero division ¯\_(ツ)_/¯
    #         translate[2] = - 0.001

    #     point[3] = int(-hor_fov_adjust*translate[0]/translate[2] + 0.5*SCREEN_W)
    #     point[4] = int(-ver_fov_adjust*translate[1]/translate[2] + 0.5*SCREEN_H)
    #     point[5] = translate[2] # np.sqrt(translate[0]*translate[0] + translate[1]*translate[1] + translate[2]*translate[2])


@njit()
def dot_3d(arr1, arr2):
    return arr1[0]*arr2[0] + arr1[1]*arr2[1] + arr1[2]*arr2[2]

@njit()
def draw_model(frame, points, triangles, camera, light_dir, z_buffer, textured, texture_uv, texture_map, texture):
    text_size = [len(texture)-1, len(texture[0])-1]
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

            shade = 0.5*dot_3d(light_dir, normal) + 0.5 #  directional lighting

            proj_points = points[triangle][:,3:]
            sorted_y = proj_points[:,1].argsort()

            start = proj_points[sorted_y[0]]
            middle = proj_points[sorted_y[1]]
            stop = proj_points[sorted_y[2]]

            x_slopes = get_slopes(start[0], middle[0], stop[0], start[1], middle[1], stop[1])

            if textured:
                z0, z1, z2 = 1/proj_points[0][2], 1/proj_points[1][2], 1/proj_points[2][2]
                uv_points = texture_uv[texture_map[index]]
                uv_points[0], uv_points[1], uv_points[2] = uv_points[0]*z0, uv_points[1]*z1, uv_points[2]*z2
                draw_text_triangles(frame, z_buffer, texture, proj_points, start, middle, stop, uv_points, x_slopes, shade, text_size, z0, z1, z2)

            else:
                color = shade*np.abs(points[triangles[index][0]][:3])*color_scale + 25
                start[2], middle[2], stop[2] = 1/start[2], 1/middle[2], 1/stop[2]
                z_slopes = get_slopes(start[2], middle[2], stop[2], start[1], middle[1], stop[1])
                draw_flat_triangle(frame, z_buffer, color, start, middle, stop, x_slopes, z_slopes)

@njit()
def draw_text_triangles(frame, z_buffer, texture, proj_points, start, middle, stop, uv_points, x_slopes, shade, text_size, z0, z1, z2):
    # barycentric denominator, based on https://codeplea.com/triangular-interpolation
    denominator = ((proj_points[1][1] - proj_points[2][1])*(proj_points[0][0] - proj_points[2][0]) +
                    (proj_points[2][0] - proj_points[1][0])*(proj_points[0][1] - proj_points[2][1]) + 1e-32)
    
    for y in range(max(0, start[1]), min(SCREEN_H, stop[1]+1)):
        x1 = start[0] + int((y-start[1])*x_slopes[0])
        if y < middle[1]:
            x2 = start[0] + int((y-start[1])*x_slopes[1])
        else:
            x2 = middle[0] + int((y-middle[1])*x_slopes[2])
        minx, maxx = max(0, min(x1, x2, SCREEN_W)), min(SCREEN_W, max(0, x1+1, x2+1))
        
        for x in range(minx, maxx):
            # barycentric weights
            w0 = ((proj_points[1][1]-proj_points[2][1])*(x - proj_points[2][0]) + (proj_points[2][0]-proj_points[1][0])*(y - proj_points[2][1]))/denominator
            w1 = ((proj_points[2][1]-proj_points[0][1])*(x - proj_points[2][0]) + (proj_points[0][0]-proj_points[2][0])*(y - proj_points[2][1]))/denominator
            w2 = 1 - w0 - w1

            z = 1/(w0*z0 + w1*z1 + w2*z2)
            u = ((w0*uv_points[0][0] + w1*uv_points[1][0] + w2*uv_points[2][0])*z)
            v = ((w0*uv_points[0][1] + w1*uv_points[1][1] + w2*uv_points[2][1])*z)

            if z < z_buffer[x, y] and min(u,v) >= 0 and max(u,v) < 1:
                z_buffer[x, y] = z
                frame[x, y] = shade*texture[int(u*text_size[0])][int(v*text_size[1])]

@njit()
def draw_flat_triangle(frame, z_buffer, color, start, middle, stop, x_slopes, z_slopes):

    for y in range(max(0, int(start[1])), min(SCREEN_H, int(stop[1]+1))):
        delta_y = y - start[1]
        x1 = start[0] + int(delta_y*x_slopes[0])
        z1 = start[2] + delta_y*z_slopes[0]

        if y < middle[1]:
            x2 = start[0] + int(delta_y*x_slopes[1])
            z2 = start[2] + delta_y*z_slopes[1]

        else:
            delta_y = y - middle[1]
            x2 = middle[0] + int(delta_y*x_slopes[2])
            z2 = middle[2] + delta_y*z_slopes[2]
        
        if x1 > x2: # lower x should be on the left
            x1, x2 = x2, x1
            z1, z2 = z2, z1

        xx1, xx2 = max(0, min(SCREEN_W, int(x1))), max(0, min(SCREEN_W, int(x2+1)))
        if xx1 != xx2:
            z_slope = (z2 - z1)/(x2 - x1 + 1e-32)
            if min(z_buffer[xx1:xx2, y]) == 1e32: # check z buffer, fresh pixels
                z_buffer[xx1:xx2, y] = 1/((np.arange(xx1, xx2)-x1)*z_slope + z1)
                frame[xx1:xx2, y] = color

            else:
                for x in range(xx1, xx2):
                    z = 1/(z1 + (x - x1)*z_slope + 1e-32) # retrive z
                    if z < z_buffer[x][y]: # check z buffer
                        z_buffer[x][y] = z
                        frame[x, y] = color

@njit()
def get_slopes(num_start, num_middle, num_stop, den_start, den_middle, den_stop):
    slope_1 = (num_stop - num_start)/(den_stop - den_start + 1e-32) # + 1e-32 avoid zero division ¯\_(ツ)_/¯
    slope_2 = (num_middle - num_start)/(den_middle - den_start + 1e-32)
    slope_3 = (num_stop - num_middle)/(den_stop - den_middle + 1e-32)

    return np.asarray([slope_1, slope_2, slope_3])

@njit()
def filter_triangles(z_min, normal, CameraRay, xxs, yys): #TODO replace filtering with proper clipping
    # only points on +z, facing the camera, check triangle bounding box
    if z_min > 0 and dot_3d(normal, CameraRay) < 0 and max(xxs) >= 0 and min(xxs) < SCREEN_W and max(yys) >= 0 and min(yys) < SCREEN_H:
        return True
    else:
        return False        
    
if __name__ == '__main__':
    main()
    pg.quit()
