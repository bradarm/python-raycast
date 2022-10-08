import math

import numpy as np
import pygame
from numba import njit
from numba.typed import List
from pygame.locals import \
    K_DOWN, \
    K_ESCAPE, \
    K_LEFT, \
    K_RIGHT, \
    K_UP, \
    KEYDOWN, \
    KEYUP

keys = {
    K_DOWN: False,
    K_ESCAPE: False,
    K_LEFT: False,
    K_RIGHT: False,
    K_UP: False,
    KEYDOWN: False,
    KEYUP: False,
}
MAP = np.array([[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 3, 0, 0, 2],
            [2, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 3, 1, 0, 0, 2, 0, 0, 0, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [2, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 1, 2, 0, 1],
            [1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 2],
            [2, 3, 1, 0, 0, 2, 0, 0, 2, 1, 3, 2, 0, 2, 0, 0, 3, 0, 3, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 2, 0, 0, 2],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 3, 0, 1, 2, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 2],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]])

WIDTH = 800
HEIGHT = 800

ROTATION_SPEED = 0.02
MOVEMENT_SPEED = 0.03

LEFT_ROTATION_MATRIX = np.array([
        [math.cos(-ROTATION_SPEED), -math.sin(-ROTATION_SPEED)],
        [math.sin(-ROTATION_SPEED), math.cos(-ROTATION_SPEED)]
    ])
RIGHT_ROTATION_MATRIX = np.array([
        [math.cos(ROTATION_SPEED), -math.sin(ROTATION_SPEED)],
        [math.sin(ROTATION_SPEED), math.cos(ROTATION_SPEED)]
    ])


def close(): 
    # Closes the program 
    pygame.display.quit()
    pygame.quit()


def get_fps(clock, font):
    clock.tick()
    fps = str(int(clock.get_fps()))
    fps_text = font.render(fps, 1, pygame.Color('coral'))
    return fps_text


@njit
def _draw_scene(position, orientation, plane):
    columns = np.arange(0, WIDTH, 2)
    colors = np.zeros((int(WIDTH / 2), 3))
    draw_starts = np.zeros(int(WIDTH / 2))
    draw_ends = np.zeros(int(WIDTH / 2))
    for i in np.arange(int(WIDTH / 2)):
        column = columns[i]
        cameraX = 2.0 * column / WIDTH - 1.0
        ray_position = position
        ray_orientation = np.array([0.0, 0.0])
        ray_orientation[0] = orientation[0] + plane[0] * cameraX
        ray_orientation[1] = orientation[1] + plane[1] * cameraX + .000000000000001  # avoid ZDE

        map_position = np.array([int(ray_position[0]), int(ray_position[1])])

        # Delta distance calculation
        delta = np.array([0.0, 0.0])
        delta[0] = math.sqrt(1.0 + (ray_orientation[1] * ray_orientation[1]) / (ray_orientation[0] * ray_orientation[0]))
        delta[1] = math.sqrt(1.0 + (ray_orientation[0] * ray_orientation[0]) / (ray_orientation[1] * ray_orientation[1]))

        # We need side_distance[0] and Y for distance calculation. Checks quadrant
        side_distance = np.array([0.0, 0.0])
        step = np.array([0.0, 0.0])
        if (ray_orientation[0] < 0):
            step[0] = -1
            side_distance[0] = (ray_position[0] - map_position[0]) * delta[0]

        else:
            step[0] = 1
            side_distance[0] = (map_position[0] + 1.0 - ray_position[0]) * delta[0]

        if (ray_orientation[1] < 0):
            step[1] = -1
            side_distance[1] = (ray_position[1] - map_position[1]) * delta[1]

        else:
            step[1] = 1
            side_distance[1] = (map_position[1] + 1.0 - ray_position[1]) * delta[1]

        # Finding distance to a wall
        hit = False
        while (not hit):
            if (side_distance[0] < side_distance[1]):
                side_distance[0] += delta[0]
                map_position[0] += step[0]
                side = True
                
            else:
                side_distance[1] += delta[1]
                map_position[1] += step[1]
                side = False
                
            if (MAP[map_position[0]][map_position[1]] > 0):
                hit = True

        # Correction against fish eye effect
        perp_wall_distance = 0.0
        if (side):
            perp_wall_distance = abs((map_position[0] - ray_position[0] + ( 1.0 - step[0] ) / 2.0) / ray_orientation[0])
        else:
            perp_wall_distance = abs((map_position[1] - ray_position[1] + ( 1.0 - step[1] ) / 2.0) / ray_orientation[1])

        # Calculating HEIGHT of the line to draw
        line_height = abs(int(HEIGHT / (perp_wall_distance+.0000001)))
        draw_starts[i] = -line_height / 2.0 + HEIGHT / 2.0

        # if drawStat < 0 it would draw outside the screen
        if (draw_starts[i] < 0):
            draw_starts[i] = 0

        draw_ends[i] = line_height / 2.0 + HEIGHT / 2.0

        if (draw_ends[i] >= HEIGHT):
            draw_ends[i] = HEIGHT - 1

        # Wall colors 0 to 3
        wall_colors = np.array([[0, 0, 0], [150,0,0], [0,150,0], [0,0,150]])
        colors[i] = wall_colors[MAP[map_position[0]][map_position[1]]]

    return columns, colors, draw_starts, draw_ends


def draw_scene(clock, font, screen, position, orientation, plane):
    screen.fill((25,25,25))
    pygame.draw.rect(screen, (50,50,50), (0, HEIGHT/2, WIDTH, HEIGHT/2))

    columns, colors, draw_starts, draw_ends = _draw_scene(List(position), List(orientation), List(plane))

    # Drawing the graphics
    for i in range(int(WIDTH / 2)):
        pygame.draw.line(screen, colors[i], (columns[i], draw_starts[i]), (columns[i], draw_ends[i]), 2)

    # Drawing HUD
    screen.blit(get_fps(clock, font), (10, 0))

    # Updating display
    pygame.event.pump()
    pygame.display.flip()
    return


def turn_left(orientation, plane):
    orientation = LEFT_ROTATION_MATRIX.dot(orientation)
    plane = LEFT_ROTATION_MATRIX.dot(plane)
    return orientation, plane


def turn_right(orientation, plane):
    orientation = RIGHT_ROTATION_MATRIX.dot(orientation)
    plane = RIGHT_ROTATION_MATRIX.dot(plane)
    return orientation, plane


@njit
def move_forward(position, orientation):
    if not MAP[int(position[0] + orientation[0] * MOVEMENT_SPEED)][int(position[1])]:
        position[0] += orientation[0] * MOVEMENT_SPEED
    if not MAP[int(position[0])][int(position[1] + orientation[1] * MOVEMENT_SPEED)]:
        position[1] += orientation[1] * MOVEMENT_SPEED
    return position


@njit
def move_backward(position, orientation):
    if not MAP[int(position[0] - orientation[0] * MOVEMENT_SPEED)][int(position[1])]:
        position[0] -= orientation[0] * MOVEMENT_SPEED
    if not MAP[int(position[0])][int(position[1] - orientation[1] * MOVEMENT_SPEED)]:
        position[1] -= orientation[1] * MOVEMENT_SPEED
    return position


def main():
    # Initialize PyGame Window
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)
    info = pygame.display.Info()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Initialize Position and orientation
    position = np.array([3.0, 7.0])
    orientation = np.array([1.0, 0.0])
    plane = np.array([0.0, 0.5])

    # PyGame Loop
    while True:
        # Catches user input
        # Sets keys[key] to True or False
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    close()
                    return
                keys[event.key] = True
            elif event.type == KEYUP:
                keys[event.key] = False

        if keys[K_ESCAPE]:
            close()

        if keys[K_LEFT]:
            orientation, plane = turn_left(orientation, plane)

        if keys[K_RIGHT]:
            orientation, plane = turn_right(orientation, plane)

        if keys[K_UP]:
            position = move_forward(position, orientation)
                
        if keys[K_DOWN]:
            position = move_backward(position, orientation)

        draw_scene(clock, font, screen, position, orientation, plane)
    return


if __name__ == '__main__':
    main()