import pygame as pg
import numpy as np
from numba import njit
from objLoader import read_obj

SCREEN_W = 800 # frame width
SCREEN_H = 2*int(SCREEN_W*0.375)
FOV_V = np.pi/4 # 45 degrees vertical fov
FOV_H = FOV_V*SCREEN_W/SCREEN_H # proportional horizontal fov
SKY_BLUE = np.asarray([50,127,200]).astype('uint8')

def main():
    pg.init()
    screen = pg.display.set_mode((800, int(800*SCREEN_H/SCREEN_W)))
    running = True
    clock = pg.time.Clock()
    surf = pg.surface.Surface((SCREEN_W, SCREEN_H))
    frame= np.ones((SCREEN_W, SCREEN_H, 3)).astype('uint8')
    z_buffer = np.ones((SCREEN_W, SCREEN_H))
    shadow_map = np.ones((SCREEN_W, SCREEN_H))

    skybox = pg.image.load('skybox1.jpg').convert()
    skybox = pg.transform.smoothscale(skybox, (SCREEN_W*(np.pi*2/FOV_H+1), SCREEN_H*(np.pi/FOV_V+1)))
    skybox = pg.surfarray.array3d(skybox)

    model1 = Model('obj models/teapot.obj', '')
    model1.change_position(10, 2, 20)

    model2 = Model('obj models/mountains.obj', '')
    model2.change_position(0, -1, 0)

    model3 = Model('obj models/finfet.obj', '')
    model3.change_position(-10, 20, -10)

    model4 = Model('obj models/cube text.obj', 'finfet.png')
    model4.change_position(y=20)

    model5 = Model('obj models/Babycrocodile.obj', 'obj models/BabyCrocodileGreen.png')
    model5.change_position(5, 0.5, 30)

    model6 = Model('obj models/cottage_obj.obj', 'obj models/cottage_diffuse.png')
    model6.change_position(-30, -0.8, 20)

    model7 = Model('obj models/ah64d.obj', 'obj models/ah64.png')
    model7.change_position(10, 30, 10)

    camera = np.asarray([11., 14.5, 68, 4.4, 0])
    light_camera = np.asarray([-500000., 1000000., -500000., 0.8, 1.])#/10
    pg.mouse.set_visible(0)

    toggle = 1

    while running:
        pg.mouse.set_pos(SCREEN_W/2, SCREEN_H/2)
        elapsed_time = clock.tick()*0.001
        
        # model3.change_position(camera[0]+ 10*np.cos(camera[3]), camera[1], camera[2] + 10*np.sin(camera[3]), reset=1)
        # model3.change_position(light_camera[0], light_camera[1], light_camera[2], scale = 10000, reset=1)
        model4.change_position(rotx=elapsed_time*.1, roty=elapsed_time*.2, rotz=elapsed_time*.5)
        model5.change_position(roty=elapsed_time*.7)

        light_camera[0] = light_camera[1]*np.sin(pg.time.get_ticks()/4500)
        light_camera[2] = light_camera[1]*np.cos(pg.time.get_ticks()/5000)

        for event in pg.event.get():
            if event.type == pg.QUIT: running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE: running = False
                if event.key == pg.K_SPACE: toggle = not(toggle)
                if event.key == pg.K_DELETE: camera = np.asarray([11., 14.5, 68, 4.4, 0]) # reset camera
        
        # frame[:,:,:] = SKY_BLUE
        h1 = int(SCREEN_W*camera[3]/FOV_H)
        v1 = int(SCREEN_H*camera[4]/FOV_V + 2*SCREEN_H)
        frame[:,:,:]  = skybox[h1:h1+SCREEN_W,v1:v1+SCREEN_H,:] # crude cylindrical skybox

        z_buffer[:,:] = 1e32 # start with some big value
        shadow_map[:,:] = 1e32
        
        render_frame(camera, frame, z_buffer, shadow_map, light_camera)

        if toggle:
            surf = pg.surfarray.make_surface(frame)        
        else:
            shadow_map[shadow_map == 1e32] = 1000
            shadow_map = shadow_map - np.min(shadow_map)
            surf = pg.surfarray.make_surface(255*shadow_map/np.max(shadow_map))
        surf = pg.transform.scale(surf, screen.get_size())
        screen.blit(surf, (0,0)); pg.display.update()
        pg.display.set_caption(str(round(1/(elapsed_time+1e-32), 1)) + ' ' + str(np.round(camera,1)))
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

    def __init__(self, path_obj, path_texture=''):
        self._registry.append(self)
        self.pointsog, self.triangles, self.texture_uv, self.texture_map, self.textured =  read_obj(path_obj)
        self.position = np.asarray([0, 0, 0, 0, 0, 0, 1]) # x y z rotx roty rotz scale
        self.points = self.pointsog.copy()
        self.shadow_points = self.points.copy()

        if self.textured and path_texture != '':
            self.texture = pg.surfarray.array3d(pg.image.load(path_texture))
        else:
            self.textured = False # set dummy texture data
            self.texture_uv, self.texture_map = np.ones((2,2)), np.random.randint(1, 2, (2,3))
            self.texture = np.random.randint(0, 255, (10, 10,3))
    
    def change_position(self, x=0, y=0, z=0, rotx=0, roty=0, rotz=0, scale=1, reset=0):

        if reset:
            self.position = np.asarray([x, y, z, rotx, roty, rotz, scale])
        else:
            self.position = self.position + np.asarray([x, y, z, rotx, roty, rotz, scale])

        self.points = self.pointsog.copy()

        # scale
        if self.position[6] != 1:
            self.points = self.points*scale

        # rotate around x axis
        if self.position[3] != 0:
            temp_points = self.points.copy()
            self.points[:,1] = temp_points[:,1]*np.cos(self.position[3]) - temp_points[:,2]*np.sin(self.position[3])
            self.points[:,2] = temp_points[:,1]*np.sin(self.position[3]) + temp_points[:,2]*np.cos(self.position[3])

        # rotate around y axis
        if self.position[4] != 0:
            temp_points = self.points.copy()
            self.points[:,0] = temp_points[:,0]*np.cos(self.position[4]) - temp_points[:,2]*np.sin(self.position[4])
            self.points[:,2] = temp_points[:,0]*np.sin(self.position[4]) + temp_points[:,2]*np.cos(self.position[4])

        # rotate around z axis
        if self.position[5] != 0:
            temp_points = self.points.copy()
            self.points[:,0] = temp_points[:,0]*np.cos(self.position[5]) - temp_points[:,1]*np.sin(self.position[5])
            self.points[:,1] = temp_points[:,0]*np.sin(self.position[5]) + temp_points[:,1]*np.cos(self.position[5])
        
        if self.position[0] != 0:
            self.points[:,0] = self.points[:,0] + self.position[0]
        
        if self.position[1] != 0:
            self.points[:,1] = self.points[:,1] + self.position[1]

        if self.position[2] != 0:
            self.points[:,2] = self.points[:,2] + self.position[2]
        
        self.shadow_points = self.points.copy()

def render_frame(camera, frame, z_buffer, shadow_map, light_camera):

    light_vx = camera[0] + 30*np.cos(camera[3]) - light_camera[0]
    light_vy = -light_camera[1]
    light_vz = camera[2] + 30*np.sin(camera[3]) - light_camera[2]

    light_vector = np.asarray([light_vx, light_vy, light_vz])
    lenght = np.linalg.norm(light_vector)

    h_vector = np.asarray([light_vector[0], light_vector[2]])/np.sqrt(light_vector[0]*light_vector[0] + light_vector[2]*light_vector[2])
    light_camera[3] = np.arccos(dot_product(h_vector, np.asarray([1, 0])))
    light_camera[4] = np.arcsin(dot_product(light_vector/lenght, np.asarray([0, -1, 0])))

    if np.sign(np.sin(light_camera[3])) != np.sign(light_vz): # not sure, sometimes points in the wrong direction
        light_camera[3] = -light_camera[3]

    for model in Model._registry:
        project_points(model.shadow_points, light_camera, shadow_mod = 0.01*lenght)
        render_shadow_map(model.shadow_points, model.triangles, light_camera, shadow_map)

    for model in Model._registry:
        project_points(model.points, camera)
        draw_model(frame, model.points, model.triangles, camera, light_camera, z_buffer, model.textured,
                       model.texture_uv, model.texture_map, model.texture, model.shadow_points, shadow_map)

@njit()
def render_shadow_map(points, triangles, light_camera, shadow_map):
    for index in range(len(triangles)):
        
        triangle = triangles[index]

        # Use Cross-Product to get surface normal
        vet1 = points[triangle[1]][:3]  - points[triangle[0]][:3]
        vet2 = points[triangle[2]][:3] - points[triangle[0]][:3]

        # backface culling with dot product between normal and camera ray
        normal = np.cross(vet1, vet2)
        normal = normal/np.sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
        CameraRay = (points[triangle[0]][:3] - light_camera[:3])/points[triangle[0]][5]

        # get projected 2d points for crude filtering of offscreen triangles
        xxs = [points[triangle[0]][3],  points[triangle[1]][3],  points[triangle[2]][3]]
        yys = [points[triangle[0]][4],  points[triangle[1]][4],  points[triangle[2]][4]]
        z_min = min([points[triangle[0]][5],  points[triangle[1]][5],  points[triangle[2]][5]])

        # check valid values
        if filter_triangles(z_min, normal, -CameraRay, xxs, yys):

            proj_points = points[triangle][:,3:]
            sorted_y = proj_points[:,1].argsort()

            start = proj_points[sorted_y[0]]
            middle = proj_points[sorted_y[1]]
            stop = proj_points[sorted_y[2]]

            x_slopes = get_slopes(start[0], middle[0], stop[0], start[1], middle[1], stop[1])
            z_slopes = get_slopes(start[2], middle[2], stop[2], start[1], middle[1], stop[1])
            
            for y in range(max(1, int(start[1])), min(SCREEN_H-1, int(stop[1]+1))):
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

                xx1, xx2 = max(1, min(SCREEN_W-1, int(x1))), max(1, min(SCREEN_W-1, int(x2+1)))
                if xx1 != xx2:
                    z_slope = (z2 - z1)/(x2 - x1 + 1e-32)
                    if min(shadow_map[xx1:xx2, y]) == 1e32: # check z buffer, fresh pixels
                        shadow_map[xx1:xx2, y] = (np.arange(xx1, xx2)-x1)*z_slope + z1
                    else:
                        for x in range(xx1, xx2):
                            z = z1 + (x - x1)*z_slope + 1e-32 # retrive z
                            if z < shadow_map[x][y]: # check z buffer
                                shadow_map[x][y] = z

@njit()
def project_points(points, camera, shadow_mod=1):

    cos_hor = np.cos(-camera[3]+np.pi/2) # add 90° to align with z axis
    sin_hor = np.sin(-camera[3]+np.pi/2) # negative (counter rotation)
    
    cos_ver = np.cos(-camera[4])
    sin_ver = np.sin(-camera[4])

    hor_fov_adjust = 0.5*SCREEN_W/ np.tan(FOV_H * 0.5/shadow_mod) 
    ver_fov_adjust = 0.5*SCREEN_H/ np.tan(FOV_V * 0.5/shadow_mod)
    
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

@njit()
def dot_product(arr1, arr2):
    if len(arr1) == len(arr2) == 2:
        return arr1[0]*arr2[0] + arr1[1]*arr2[1]

    elif len(arr1) == len(arr2) == 3:
        return arr1[0]*arr2[0] + arr1[1]*arr2[1] + arr1[2]*arr2[2]

@njit()
def draw_model(frame, points, triangles, camera, light_camera, z_buffer, textured, texture_uv, texture_map, texture, shadow_points, shadow_map):
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
            lightRay = (points[triangle[0]][:3] - light_camera[:3])/points[triangle[0]][5]
            lightRay = lightRay/np.sqrt(lightRay[0]*lightRay[0] + lightRay[1]*lightRay[1] + lightRay[2]*lightRay[2])
            shade1 =  0.2 + 0.8*(- 0.5*dot_product(lightRay, normal) + 0.5) #  directional lighting

            proj_points = points[triangle][:,3:]
            proj_shadows = shadow_points[triangle][:,3:]

            sorted_y = proj_points[:,1].argsort()

            start = proj_points[sorted_y[0]]
            middle = proj_points[sorted_y[1]]
            stop = proj_points[sorted_y[2]]

            x_slopes = get_slopes(start[0], middle[0], stop[0], start[1], middle[1], stop[1])

            min_z = min(proj_points[0][2], proj_points[1][2], proj_points[2][2])

            z0, z1, z2 = 1/proj_points[0][2], 1/proj_points[1][2], 1/proj_points[2][2]

            if textured:
                uv_points = texture_uv[texture_map[index]]
                uv_points[0], uv_points[1], uv_points[2] = uv_points[0]*z0, uv_points[1]*z1, uv_points[2]*z2
            else:
                color0 = (np.abs(points[triangles[index][0]][:3])*color_scale + 25)*z0
                color1 = (np.abs(points[triangles[index][1]][:3])*color_scale + 25)*z1
                color2 = (np.abs(points[triangles[index][2]][:3])*color_scale + 25)*z2

            # barycentric denominator, based on https://codeplea.com/triangular-interpolation
            denominator = 1/((proj_points[1][1] - proj_points[2][1])*(proj_points[0][0] - proj_points[2][0]) +
                             (proj_points[2][0] - proj_points[1][0])*(proj_points[0][1] - proj_points[2][1]) + 1e-32)

            proj_shadows[0] = proj_shadows[0]*z0
            proj_shadows[1] = proj_shadows[1]*z1
            proj_shadows[2] = proj_shadows[2]*z2

            for y in range(max(0, int(start[1])), min(SCREEN_H, int(stop[1]+1))):
                x1 = start[0] + int((y-start[1])*x_slopes[0])
                if y < middle[1]:
                    x2 = start[0] + int((y-start[1])*x_slopes[1])
                else:
                    x2 = middle[0] + int((y-middle[1])*x_slopes[2])
                minx, maxx = max(0, min(x1, x2, SCREEN_W)), min(SCREEN_W, max(0, x1+1, x2+1))
                
                for x in range(int(minx), int(maxx)):
                    # barycentric weights
                    w0 = ((proj_points[1][1]-proj_points[2][1])*(x - proj_points[2][0]) + (proj_points[2][0]-proj_points[1][0])*(y - proj_points[2][1]))*denominator
                    w1 = ((proj_points[2][1]-proj_points[0][1])*(x - proj_points[2][0]) + (proj_points[0][0]-proj_points[2][0])*(y - proj_points[2][1]))*denominator
                    w2 = 1 - w0 - w1
                    
                    z = 1/(w0*z0 + w1*z1 + w2*z2 + 1e-32)
                    if z < z_buffer[x, y] and z >= min_z:
                        z_buffer[x, y] = z

                        shade2 = 1
                        if shade1 < 0.6:
                            shade2 = shade1
                        else:
                            point = (w0*proj_shadows[0] + w1*proj_shadows[1] + w2*proj_shadows[2])*z
                            lx, ly = max(0, min(SCREEN_W-1, int(point[0]))), max(0, min(SCREEN_H-1, int(point[1])))

                            if point[2] > shadow_map[lx][ly]:
                                shade2 = min(0.9, 2.5/np.sum(point[2] > shadow_map[lx-1:lx+1,ly-1:ly+1]))

                        if textured:
                            u = int((w0*uv_points[0][0] + w1*uv_points[1][0] + w2*uv_points[2][0])*z*text_size[0])
                            v = int((w0*uv_points[0][1] + w1*uv_points[1][1] + w2*uv_points[2][1])*z*text_size[1])
                            if min(u,v) >= 0 and u < text_size[0] and v < text_size[1]:
                                frame[x, y] = shade1*shade2*texture[u][v]
                        else:
                            color =  (w0*color0 + w1*color1 + w2*color2)*z
                            frame[x, y] = shade1*shade2*color

@njit()
def get_slopes(num_start, num_middle, num_stop, den_start, den_middle, den_stop):
    slope_1 = (num_stop - num_start)/(den_stop - den_start + 1e-32) # + 1e-32 avoid zero division ¯\_(ツ)_/¯
    slope_2 = (num_middle - num_start)/(den_middle - den_start + 1e-32)
    slope_3 = (num_stop - num_middle)/(den_stop - den_middle + 1e-32)

    return np.asarray([slope_1, slope_2, slope_3])

@njit()
def filter_triangles(z_min, normal, CameraRay, xxs, yys, scale=1): #TODO replace filtering with proper clipping
    # only points on +z, facing the camera, check triangle bounding box
    if z_min > 0 and dot_product(normal, CameraRay) < 0 and max(xxs) >= 0 and min(xxs) < SCREEN_W*scale and max(yys) >= 0 and min(yys) < SCREEN_H*scale:
        return True
    else:
        return False        
    
if __name__ == '__main__':
    main()
    pg.quit()
