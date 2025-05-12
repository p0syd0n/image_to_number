import pygame

pygame.init()
pygame.display.set_caption("Pixel Drawing")

GRID_SIZE = 28
SCALE = 20
WINDOW_SIZE = GRID_SIZE * SCALE
BRUSH_SIZE = 2

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
clock = pygame.time.Clock()

pixels = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

def apply_brush(grid_x, grid_y, intensity=50):
    half = BRUSH_SIZE // 2
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            x = grid_x + dx
            y = grid_y + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                dist = (dx ** 2 + dy ** 2) ** 1
                fade = max(0, intensity - int(dist * 40))
                pixels[y][x] = min(255, pixels[y][x] + fade)

running = True
while running:
    mousePressed = pygame.mouse.get_pressed()[0]
    mousePos = pygame.math.Vector2(pygame.mouse.get_pos())
    grid_x = int(mousePos.x // SCALE)
    grid_y = int(mousePos.y // SCALE)

    if mousePressed:
        apply_brush(grid_x, grid_y, 75)

    screen.fill((0, 0, 0))

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            val = pixels[y][x]
            color = (val, val, val)
            pygame.draw.rect(screen, color, (x * SCALE, y * SCALE, SCALE, SCALE))

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            pixels = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
