import pygame
import sys
import os
import random

# Game Variables
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
BIRD_X = 50
GRAVITY = 0.1
TILT_DELAY = 500  # Delay in milliseconds before tilting down



# Get the absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_asset_path(filename):
    return os.path.join(BASE_DIR, 'sprites', filename)

class Bird:
    def __init__(self):
        self.y = SCREEN_HEIGHT / 2
        self.speed = 0
        self.angle = 0  # Initial angle of the bird
        self.max_angle = 25  # Maximum tilt angle
        self.min_angle = -25  # Minimum tilt angle
        self.tilt_up_speed = 2  # Speed at which the bird tilts up
        self.tilt_down_speed = 1  # Speed at which the bird tilts down
        self.sprite = pygame.image.load(get_asset_path('flap.png'))
        self.sprite = pygame.transform.scale(self.sprite, (45, 45))
        self.last_flap_time = 0  # Time when the space bar was last pressed

    def update(self, space_pressed, current_time):
        # Update speed based on gravity
        self.speed += GRAVITY
        self.y += self.speed

        # Keep bird on screen
        if self.y > SCREEN_HEIGHT - self.sprite.get_height():
            self.y = SCREEN_HEIGHT - self.sprite.get_height()
            self.speed = 0
            self.angle = 0

        # Gradually tilt up if space is pressed
        if space_pressed:
            self.last_flap_time = current_time
            if self.angle < self.max_angle:
                self.angle += self.tilt_up_speed + 30
        else:
            # Check if enough time has passed since the last flap
            if current_time - self.last_flap_time > TILT_DELAY:
                self.angle -= self.tilt_down_speed
                if self.angle < self.min_angle:
                    self.angle = self.min_angle

    def flap(self):
        # Set the bird's speed to jump up
        self.speed = -3

    def draw(self, screen):
        # Rotate the sprite based on the angle
        rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        # Get the rect of the rotated sprite to use for positioning
        rect = rotated_sprite.get_rect(center=self.sprite.get_rect(topleft=(BIRD_X, int(self.y))).center)
        # Draw the rotated sprite onto the screen
        screen.blit(rotated_sprite, rect.topleft)

    def get_rect(self):
        # Return a rectangle that surrounds the bird's sprite
        return pygame.Rect(BIRD_X, int(self.y), self.sprite.get_width(), self.sprite.get_height())

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 50  # Width of the pipes
        self.height = random.randint(200, 350)  # Random height of the top pipe
        self.gap = random.randint(100, 200)  # Gap between the top and bottom pipes
        self.passed = False
        self.top_image = pygame.image.load(get_asset_path('pipe_top.png'))
        self.bottom_image = pygame.image.load(get_asset_path('pipe_bottom.png'))
        self.top_image = pygame.transform.scale(self.top_image, (self.width, self.height))
        self.bottom_image = pygame.transform.scale(self.bottom_image, (self.width, SCREEN_HEIGHT - self.height - self.gap))

    def update(self):
        self.x -= 2  # Move the pipes to the left

    def draw(self, screen):
        screen.blit(self.top_image, (self.x, 0))  # Draw the top pipe
        screen.blit(self.bottom_image, (self.x, self.height + self.gap))  # Draw the bottom pipe

    def get_rects(self):
        # Adjust the pipe rectangle dimensions to match the actual pipe sizes
        top_rect = pygame.Rect(self.x + 5, 0, self.width - 20, self.height - 10)
        bottom_rect = pygame.Rect(self.x + 5, self.height + self.gap, self.width - 20, SCREEN_HEIGHT - self.height - self.gap)
        return [top_rect, bottom_rect]

class Button:
    def __init__(self, x, y, width, height, text, font, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.is_hovered = False

    def draw(self, screen):
        pygame.draw.rect(screen, self.hover_color if self.is_hovered else self.color, self.rect)
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        screen.blit(text_surface, (self.rect.centerx - text_surface.get_width() // 2, self.rect.centery - text_surface.get_height() // 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered:
            return True
        return False
def check_collision(bird, pipes):
    bird_rect = bird.get_rect()
    for pipe in pipes:
        pipe_rects = pipe.get_rects()
        for rect in pipe_rects:
            if bird_rect.colliderect(rect):
                return True
    return False

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 55)

    record_file = os.path.join(BASE_DIR, 'record.txt')
    try:
        with open(record_file, "r") as file:
            record = int(file.read())
    except FileNotFoundError:
        record = 0

    def reset_game():
        nonlocal bird, pipes, game_over, score
        bird = Bird()
        pipes = [Pipe(SCREEN_WIDTH + i * 200) for i in range(5)]
        game_over = False
        score = 0

    bird = Bird()
    pipes = [Pipe(SCREEN_WIDTH + i * 200) for i in range(5)]
    game_over = False
    score = 0

    restart_button = Button(SCREEN_WIDTH // 2 - 75, SCREEN_HEIGHT // 2 + 50, 150, 50, "Restart", font, (255, 255, 255), (200, 200, 200))

    while True:
        space_pressed = False
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not game_over:
                bird.flap()
                space_pressed = True
            if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                space_pressed = False
            if game_over and restart_button.handle_event(event):
                reset_game()

        if not game_over:
            bird.update(space_pressed, current_time)
            for pipe in pipes:
                pipe.update()
                if not pipe.passed and pipe.x < BIRD_X:
                    pipe.passed = True
                    score += 1

            pipes = [pipe for pipe in pipes if pipe.x + 50 > 0]
            if len(pipes) < 5:
                pipes.append(Pipe(pipes[-1].x + 200))

            if check_collision(bird, pipes):
                game_over = True

        screen.fill((0, 120, 255))

        for pipe in pipes:
            pipe.draw(screen)

        bird.draw(screen)

        if game_over:
            text = font.render("You Died", True, (255, 255, 255))
            screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 4))

            if score > record:
                record = score
                with open(record_file, "w") as file:
                    file.write(str(record))

            record_text = font.render(f"Record: {record}", True, (255, 255, 255))
            screen.blit(record_text, (SCREEN_WIDTH // 2 - record_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))

            restart_button.draw(screen)

        score_text = font.render(str(score), True, (255, 255, 255))
        screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 0))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
