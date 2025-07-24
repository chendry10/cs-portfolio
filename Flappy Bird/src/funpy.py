import pygame
import sys
import random

# --- Constants ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GROUND_HEIGHT = 50
FPS = 60

BIRD_RADIUS = 20
BIRD_X = SCREEN_WIDTH // 4
GRAVITY = 0.5
JUMP_VELOCITY = -10

PIPE_WIDTH = 80
PIPE_GAP = 150
PIPE_SPEED = 3
PIPE_INTERVAL = 1500  # ms between pipes

BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50

# --- Classes ---
class Bird:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = BIRD_X
        self.y = SCREEN_HEIGHT // 2
        self.vel = 0

    def jump(self):
        self.vel = JUMP_VELOCITY

    def move(self):
        self.vel += GRAVITY
        self.y += self.vel

    def get_rect(self):
        return pygame.Rect(self.x - BIRD_RADIUS,
                           self.y - BIRD_RADIUS,
                           BIRD_RADIUS * 2,
                           BIRD_RADIUS * 2)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), BIRD_RADIUS)

class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.height = random.randint(50, SCREEN_HEIGHT - GROUND_HEIGHT - PIPE_GAP - 50)
        self.scored = False

    def reset(self):
        self.__init__()

    def move(self):
        self.x -= PIPE_SPEED

    def off_screen(self):
        return self.x + PIPE_WIDTH < 0

    def collides_with(self, bird_rect):
        top = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom = pygame.Rect(self.x,
                             self.height + PIPE_GAP,
                             PIPE_WIDTH,
                             SCREEN_HEIGHT - GROUND_HEIGHT - (self.height + PIPE_GAP))
        return bird_rect.colliderect(top) or bird_rect.colliderect(bottom)

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), (self.x, 0, PIPE_WIDTH, self.height))
        pygame.draw.rect(screen, (0, 255, 0),
                         (self.x,
                          self.height + PIPE_GAP,
                          PIPE_WIDTH,
                          SCREEN_HEIGHT - GROUND_HEIGHT - (self.height + PIPE_GAP)))

def draw_button(screen, font, text, center):
    text_surf = font.render(text, True, (255, 255, 255))
    btn_rect = pygame.Rect(0, 0, BUTTON_WIDTH, BUTTON_HEIGHT)
    btn_rect.center = center
    pygame.draw.rect(screen, (70, 70, 70), btn_rect)
    pygame.draw.rect(screen, (255, 255, 255), btn_rect, 3)
    screen.blit(text_surf,
                (btn_rect.centerx - text_surf.get_width()//2,
                 btn_rect.centery - text_surf.get_height()//2))
    return btn_rect

# --- Main Game ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 48)

    bird = Bird()
    pipes = []
    score = 0

    ADDPIPE = pygame.USEREVENT + 1
    pygame.time.set_timer(ADDPIPE, PIPE_INTERVAL)

    game_active = True
    running = True

    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if game_active:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    bird.jump()
                if event.type == ADDPIPE:
                    pipes.append(Pipe())
            else:
                # In game-over state: restart on R or click button
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_r, pygame.K_SPACE):
                    game_active = True
                    bird.reset()
                    pipes.clear()
                    score = 0
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if restart_btn.collidepoint(event.pos):
                        game_active = True
                        bird.reset()
                        pipes.clear()
                        score = 0

        if game_active:
            # --- Update ---
            bird.move()
            for pipe in pipes:
                pipe.move()
                if not pipe.scored and pipe.x + PIPE_WIDTH < bird.x:
                    score += 1
                    pipe.scored = True
            pipes = [p for p in pipes if not p.off_screen()]

            # Collision?
            bird_rect = bird.get_rect()
            if (bird.y - BIRD_RADIUS <= 0 or
                bird.y + BIRD_RADIUS >= SCREEN_HEIGHT - GROUND_HEIGHT or
                any(p.collides_with(bird_rect) for p in pipes)):
                game_active = False

            # --- Draw Play ---
            screen.fill((135, 206, 235))
            for pipe in pipes:
                pipe.draw(screen)
            pygame.draw.rect(screen, (222, 184, 135),
                             (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))
            bird.draw(screen)
            score_surf = font.render(str(score), True, (255, 255, 255))
            screen.blit(score_surf,
                        (SCREEN_WIDTH//2 - score_surf.get_width()//2, 20))
        else:
            # --- Draw Game Over & Button ---
            screen.fill((0, 0, 0))
            over_surf = font.render("Game Over", True, (255, 0, 0))
            screen.blit(over_surf,
                        (SCREEN_WIDTH//2 - over_surf.get_width()//2,
                         SCREEN_HEIGHT//2 - 100))
            restart_btn = draw_button(screen, font, "Restart (R)", (SCREEN_WIDTH//2, SCREEN_HEIGHT//2))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
