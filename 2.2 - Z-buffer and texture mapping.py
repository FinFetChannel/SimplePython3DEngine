import pygame as pg
import numpy as np
from numba import njit

SCREEN_W, SCREEN_H = 800, 600
SCALE = 2
FOV_V = np.pi/4 # 45 degrees vertical fov
FOV_H = FOV_V*SCREEN_W/SCREEN_H
SKY_BLUE = np.asarray([50,127,200]).astype('uint8')


def main():
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    running = True
    clock = pg.time.Clock()
    surf = pg.surface.Surface((SCREEN_W, SCREEN_H))
    frameblue = np.ones((SCREEN_W, SCREEN_H, 3)).astype('uint8')
    frameblue[:,:,0], frameblue[:,:,1], frameblue[:,:,2]  = SKY_BLUE[0], SKY_BLUE[1], SKY_BLUE[2]
    texture = pg.surfarray.array3d(pg.image.load('finfet.png'))

    points = 10*np.asarray([[0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1], 
                            [0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1]])
    
    texture_coord = np.asarray([[0,1], [1,1], [0,0], [1,0]])

    triangles = np.asarray([[0,1,2], [0,2,3],[3,2,6], [3,6,7], [1,5,6], [1,6,2], [0,3,7], [0,7,4], [1,0,4], [1,4,5], [6,5,4], [6,4,7]])
    texture_map = np.asarray([[2,0,1], [2,1,3], [2,0,1], [2,1,3], [2,0,1], [2,1,3], [2,0,1], [2,1,3], [2,0,1], [2,1,3], [2,0,1], [2,1,3],])
    #points, triangles =  read_obj('mountains.obj')

    color_scale = 230/np.max(np.abs(points))

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
        
        project_points(points, camera)
        frame = frameblue.copy()

        z_buffer = np.ones((SCREEN_W, SCREEN_H)) + 999999 #start with some big value

        draw_triangles(frame, points, triangles, camera, light_dir, z_buffer, texture_coord, texture_map, texture)
        
        surf = pg.surfarray.make_surface(frame)

        screen.blit(surf, (0,0)); pg.display.update()
        pg.display.set_caption(str(round(1/(elapsed_time+1e-16), 1)) + ' ' + str(camera))
        movement(camera, elapsed_time*10)

def movement(camera, elapsed_time):

    if pg.mouse.get_focused():
        p_mouse = pg.mouse.get_pos()
        camera[3] = (camera[3] + 10*elapsed_time*np.clip((p_mouse[0]-SCREEN_W/2)/SCREEN_W, -0.2, .2))%(2*np.pi)
        camera[4] = camera[4] + 10*elapsed_time*np.clip((p_mouse[1]-SCREEN_H/2)/SCREEN_H, -0.2, .2)
        camera[4] = np.clip(camera[4], -.3, .3)
    
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

@njit() #TODO: better vertical projection
def project_points(points, camera):

    for point in points:
        # Calculate xy angle of vector from camera point to projection point
        h_angle_camera_point = np.arctan((point[2]-camera[2])/(point[0]-camera[0] + 1e-16))
        
        # Check if it isn't pointing backwards
        if abs(camera[0]+np.cos(h_angle_camera_point)-point[0]) > abs(camera[0]-point[0]):
            h_angle_camera_point = (h_angle_camera_point - np.pi)%(2*np.pi)

        # Calculate difference between camera angle and pointing angle
        h_angle = (h_angle_camera_point-camera[3])%(2*np.pi)
        
        # Bring to -pi to pi range
        if h_angle > np.pi: h_angle =  h_angle - 2*np.pi
        
        # Calculate the point horizontal screen coordinate
        point[3] = int(SCREEN_W*h_angle/FOV_H + SCREEN_W/2)

        # Calculate xy distance from camera point to projection point
        distance = np.sqrt((point[0]-camera[0])**2 + (point[1]-camera[1])**2 + (point[2]-camera[2])**2)
        
        # keep value for z_buffer
        point[5] = distance
        
        # Calculate angle to xy plane
        v_angle_camera_point = np.arcsin((camera[1]-point[1])/distance)

        # Calculate difference between camera verticam angle and pointing vertical angle
        v_angle = (v_angle_camera_point - camera[4])%(2*np.pi)
        if v_angle > np.pi: v_angle =  v_angle - 2*np.pi

        # Bring to -pi to pi range
        if v_angle > np.pi: v_angle =  v_angle - 2*np.pi

        # Calculate the point vertical screen coordinate
        point[4] = int(SCREEN_H*v_angle/FOV_V + SCREEN_H/2)

@njit()
def dot_3d(arr1, arr2):
    return arr1[0]*arr2[0] + arr1[1]*arr2[1] + arr1[2]*arr2[2]

@njit()
def sort_triangles(points, triangles, camera, light_dir, z_order, shade):

    for i in range(len(triangles)):
        triangle = triangles[i]

        # Use Cross-Product to get surface normal
        vet1 = points[triangle[1]][:3]  - points[triangle[0]][:3]
        vet2 = points[triangle[2]][:3] - points[triangle[0]][:3]

        # backface culling with dot product between normal and camera ray
        normal = np.cross(vet1, vet2)
        normal = normal/np.sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
        
        CameraRay = points[triangle[0]][:3] - camera[:3]
        dist2cam = np.sqrt(CameraRay[0]*CameraRay[0] + CameraRay[1]*CameraRay[1] + CameraRay[2]*CameraRay[2])
        CameraRay = CameraRay/dist2cam

        # get projected 2d points for filtering of offscreen triangles
        xxs = [points[triangle[0]][3:5][0],  points[triangle[1]][3:5][0],  points[triangle[2]][3:5][0]]
        yys = [points[triangle[0]][3:5][1],  points[triangle[1]][3:5][1],  points[triangle[2]][3:5][1]]

        # check valid values
        if (dot_3d(normal, CameraRay) < 0   and ((xxs[0] >= -SCREEN_W and xxs[0] < 2*SCREEN_W and yys[0] >= -SCREEN_H and yys[0] < 2*SCREEN_H) or
                                                 (xxs[1] >= -SCREEN_W and xxs[1] < 2*SCREEN_W and yys[1] >= -SCREEN_H and yys[1] < 2*SCREEN_H) or
                                                 (xxs[2] >= -SCREEN_W and xxs[2] < 2*SCREEN_W and yys[2] >= -SCREEN_H and yys[2] < 2*SCREEN_H))):            
            # order triangles by distance
            z_order[i] = -dist2cam
            # calculate shading, normalize, dot and to 0 - 1 range
            shade[i] = 0.5*dot_3d(light_dir, normal) + 0.5

        # big value for last positions in sort
        else: z_order[i] = 9999

def read_obj(fileName):
    vertices = []
    triangles = []
    f = open(fileName)
    for line in f:
        if line[:2] == "v ":
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)
            
            vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]), 1, 1, 1]
            vertices.append(vertex)

        elif line[0] == "f":
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)

            triangles.append([int(line[index1:index2]) - 1, int(line[index2:index3]) - 1, int(line[index3:-1]) - 1])

    f.close()

    return np.asarray(vertices), np.asarray(triangles)

@njit()
def draw_triangles(frame, points, triangles, camera, light_dir, z_buffer, texture_coord, texture_map, texture):
    text_size = [len(texture)-1, len(texture[0])-1]
    for index in range(len(triangles)):
        
        triangle = triangles[index]

        # Use Cross-Product to get surface normal
        vet1 = points[triangle[1]][:3]  - points[triangle[0]][:3]
        vet2 = points[triangle[2]][:3] - points[triangle[0]][:3]

        # backface culling with dot product between normal and camera ray
        normal = np.cross(vet1, vet2)
        normal = normal/np.sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
        
        CameraRay = points[triangle[0]][:3] - camera[:3]
        CameraRay = CameraRay/points[triangle[0]][5]

        # get projected 2d points for crude filtering of offscreen triangles
        xxs = [points[triangle[0]][3:5][0],  points[triangle[1]][3:5][0],  points[triangle[2]][3:5][0]]
        yys = [points[triangle[0]][3:5][1],  points[triangle[1]][3:5][1],  points[triangle[2]][3:5][1]]

        # check valid values
        # if (dot_3d(normal, CameraRay) < 0   and ((xxs[0] >= -SCREEN_W and xxs[0] < 2*SCREEN_W and yys[0] >= -SCREEN_H and yys[0] < 2*SCREEN_H) or
        if (((xxs[0] >= -SCREEN_W and xxs[0] < 2*SCREEN_W and yys[0] >= -SCREEN_H and yys[0] < 2*SCREEN_H) or
                                                 (xxs[1] >= -SCREEN_W and xxs[1] < 2*SCREEN_W and yys[1] >= -SCREEN_H and yys[1] < 2*SCREEN_H) or
                                                 (xxs[2] >= -SCREEN_W and xxs[2] < 2*SCREEN_W and yys[2] >= -SCREEN_H and yys[2] < 2*SCREEN_H))):            
            # calculate shading, normalize, dot and to 0 - 1 range
            shade = 0.5*dot_3d(light_dir, normal) + 0.5

            point0 = list(points[triangle[0]][3:])
            point1 = list(points[triangle[1]][3:])
            point2 = list(points[triangle[2]][3:])

            triangle_text = texture_map[index]
            text0 = texture_coord[triangle_text[0]]
            text1 = texture_coord[triangle_text[1]]
            text2 = texture_coord[triangle_text[2]]

            proj_points = np.asarray([point0, point1, point2])
            text_points = np.asarray([list(text0), list(text1), list(text2)])

            sorted_y = proj_points[:,1].argsort()
            x_start, y_start, z_start = proj_points[sorted_y[0]]
            x_middle, y_middle, z_middle = proj_points[sorted_y[1]]
            x_stop, y_stop, z_stop = proj_points[sorted_y[2]]
            
            slope_1 = (x_stop - x_start)/(y_stop - y_start + 1e-16)
            slope_2 = (x_middle - x_start)/(y_middle - y_start + 1e-16)
            slope_3 = (x_stop - x_middle)/(y_stop - y_middle + 1e-16)
            
            slope_z1 = (z_stop - z_start)/(y_stop - y_start + 1e-16)
            slope_z2 = (z_middle - z_start)/(y_middle - y_start + 1e-16)
            slope_z3 = (z_stop - z_middle)/(y_stop - y_middle + 1e-16)

            for y in range(max(0, y_start), min(SCREEN_H, y_stop)):
                x1 = x_start + int((y-y_start)*slope_1)
                z1 = z_start + int((y-y_start)*slope_z1)
                uv_inter1 = text_points[sorted_y[0]] + (text_points[sorted_y[2]] - text_points[sorted_y[0]])*(y-y_start)/(y_stop-y_start+1e-16)
                if y < y_middle:
                    x2 = x_start + int((y-y_start)*slope_2)
                    z2 = z_start + int((y-y_start)*slope_z2)
                    uv_inter2 = text_points[sorted_y[0]] + (text_points[sorted_y[1]] - text_points[sorted_y[0]])*(y-y_start)/(y_middle-y_start+1e-16)

                else:
                    x2 = x_middle + int((y-y_middle)*slope_3)
                    z2 = z_middle + int((y-y_middle)*slope_z3)
                    uv_inter2 = text_points[sorted_y[1]] + (text_points[sorted_y[2]] - text_points[sorted_y[1]])*(y-y_middle)/(y_stop - y_middle+1e-16)

                if x1 > x2:
                    x1, x2 = x2, x1
                    z1, z2 = z2, z1
                    uv_inter1, uv_inter2 = uv_inter2, uv_inter1
                xx1, xx2 = max(0, min(x1, SCREEN_W)), max(0, min(x2, SCREEN_W))
                
                for x in range(xx1, xx2):
                    
                    uv = uv_inter1 + (uv_inter2 - uv_inter1)*(x - x1)/(x2 - x1 + 1e-16)
                    z = z1 + (z2 - z1)*(x - x1)/(x2 - x1 + 1e-16)

                    if z < z_buffer[x, y]:
                        z_buffer[x, y] = z
                        frame[x, y] = shade*texture[int(uv[0]*text_size[0])][int(uv[1]*text_size[1])]
            

if __name__ == '__main__':
    main()
    pg.quit()
