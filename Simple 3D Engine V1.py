import pygame as pg
import numpy as np
from numba import njit

SCREEN_W, SCREEN_H = 800, 600
FOV_V = np.pi/4 # 45 degrees vertical fov
FOV_H = FOV_V*SCREEN_W/SCREEN_H


def main():
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    running = True
    clock = pg.time.Clock()
    surf = pg.surface.Surface((SCREEN_W, SCREEN_H))

    # points = np.asarray([[1, 1, 1, 1, 1], [4, 2, 0, 1, 1], [1, .5, 3, 1, 1]])
    # triangles = np.asarray([[0,1,2]])
    points, triangles =  read_obj('teapot.obj')

    camera = np.asarray([13, 0.5, 2, 3.3, 0])
    
    z_order = np.zeros(len(triangles))
    shade = z_order.copy()

    while running:
        pg.mouse.set_pos(SCREEN_W/2, SCREEN_H/2)
        surf.fill([50,127,200])
        elapsed_time = clock.tick()*0.001
        light_dir = np.asarray([np.sin(pg.time.get_ticks()/1000), 1, 1])
        light_dir = light_dir/np.linalg.norm(light_dir)

        for event in pg.event.get():
            if event.type == pg.QUIT: running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: running = False
        
        project_points(points, camera)
        sort_triangles(points, triangles, camera, light_dir, z_order, shade) 
        
        for index in np.argsort(z_order):
            if z_order[index] == 9999: break
            triangle = [points[triangles[index][0]][3:], points[triangles[index][1]][3:], points[triangles[index][2]][3:]]
            color = shade[index]*np.abs(points[triangles[index][0]][:3])*45 +25
            pg.draw.polygon(surf, color, triangle)                

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
        point[3] = SCREEN_W*h_angle/FOV_H + SCREEN_W/2

        # Calculate xy distance from camera point to projection point
        distance = np.sqrt((point[0]-camera[0])**2 + (point[1]-camera[1])**2 + (point[2]-camera[2])**2)
        
        # Calculate angle to xy plane
        v_angle_camera_point = np.arcsin((camera[1]-point[1])/distance)

        # Calculate difference between camera verticam angle and pointing vertical angle
        v_angle = (v_angle_camera_point - camera[4])%(2*np.pi)
        if v_angle > np.pi: v_angle =  v_angle - 2*np.pi

        # Bring to -pi to pi range
        if v_angle > np.pi: v_angle =  v_angle - 2*np.pi

        # Calculate the point vertical screen coordinate
        point[4] = SCREEN_H*v_angle/FOV_V + SCREEN_H/2

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
        xxs = np.asarray([points[triangle[0]][3:5][0],  points[triangle[1]][3:5][0],  points[triangle[2]][3:5][0]])
        yys = np.asarray([points[triangle[0]][3:5][1],  points[triangle[1]][3:5][1],  points[triangle[2]][3:5][1]])

        # check valid values
        if (dot_3d(normal, CameraRay) < 0   and np.min(xxs) > - SCREEN_W and np.max(xxs) < 2*SCREEN_W
                                            and np.min(yys) > - SCREEN_H and np.max(yys) < 2*SCREEN_H):
            
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
            
            vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]), 1, 1]
            vertices.append(vertex)

        elif line[0] == "f":
            index1 = line.find(" ") + 1
            index2 = line.find(" ", index1 + 1)
            index3 = line.find(" ", index2 + 1)

            triangles.append([int(line[index1:index2]) - 1, int(line[index2:index3]) - 1, int(line[index3:-1]) - 1])

    f.close()

    return np.asarray(vertices), np.asarray(triangles)


if __name__ == '__main__':
    main()
    pg.quit()
