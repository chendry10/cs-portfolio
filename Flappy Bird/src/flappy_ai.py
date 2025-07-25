import pygame
import sys
import os
import random
import numpy as np
from bird import Bird
from pipe import Pipe
from nn import NeuralNetwork
import argparse

# ─── Constants ────────────────────────────────────────────────────────────────
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
BIRD_X = 50
FPS = 240
#changes traced
GRAVITY       = 400.0    # px/sec²
FLAP_VELOCITY = -200.0   # px/sec (instant jump)
PIPE_SPEED    = 120.0    # px/sec

TILT_DELAY    = 500      # ms before tilting down
PIPE_SPACING  = 200      # horizontal space between pipes

# ─── Assets Path ─────────────────────────────────────────────────────────────
def get_asset_path(fn):
    return os.path.join(os.path.dirname(__file__), "sprites", fn)

# ─── Neural Network & GA Helpers ────────────────────────────────────────────

def crossover(w1, w2):
    mask = np.random.rand(len(w1)) < 0.5
    return np.where(mask, w1, w2)

def mutate(w, rate=0.1, scale=0.5):
    for i in range(len(w)):
        if random.random() < rate:
            w[i] += np.random.randn() * scale

# ─── Game Entities ───────────────────────────────────────────────────────────

def check_collision(bird, pipes):
    b_mask, b_rect = bird.get_mask()
    for p in pipes:
        if b_mask.overlap(p.top_mask,     (p.x - b_rect.x, 0 - b_rect.y)):      return True
        if b_mask.overlap(p.bottom_mask,  (p.x - b_rect.x, p.height+p.gap - b_rect.y)): return True
    return False

# ─── Evolution & Simulation ─────────────────────────────────────────────────
def eval_population(pop, display=False):
    pygame.init()

    if display:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        font   = pygame.font.SysFont(None, 30)

        # ── Load & scale background after display init ─────────
        raw_bg = pygame.image.load(get_asset_path('background.png')).convert()
        bg_img = pygame.transform.scale(raw_bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        # load ground and key out the sky‐fill color
        ground_img = pygame.image.load(get_asset_path('ground.png')).convert()
        ground_img.set_colorkey((112, 192, 238))
        GROUND_Y = SCREEN_HEIGHT - ground_img.get_height()
    else:
        screen = None
        bg_img = ground_img = None
        GROUND_Y = SCREEN_HEIGHT - 100

    clock   = pygame.time.Clock()
    sim_now = 0.0
    SIM_FPS = 60.0

    pipes = [Pipe(SCREEN_WIDTH + i * PIPE_SPACING, display=display) for i in range(5)]
    bg_scroll = ground_scroll = 0
    scroll_speed = PIPE_SPEED

    run = True
    while run and any(b.alive for b in pop):
        if display:
            dt_ms = clock.tick(FPS)
            sim_now = pygame.time.get_ticks()
        else:
            dt_ms = 1000.0 / SIM_FPS
            sim_now += dt_ms

        dt_s = dt_ms / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        # ── Scroll updates ─────────────────────────────────────────
        bg_scroll     = (bg_scroll - scroll_speed * dt_s * 0.5) % SCREEN_WIDTH
        ground_scroll = (ground_scroll - scroll_speed * dt_s)       % SCREEN_WIDTH

        # ── Pipes update ───────────────────────────────────────────
        rem = []; add = False
        for p in pipes:
            p.update(dt_s)
            if not p.passed and p.x < BIRD_X:
                p.passed = True
                add = True
                for b in pop:
                    if b.alive: b.fitness += 1
            if p.x + p.width < 0:
                rem.append(p)
        for r in rem:
            pipes.remove(r)
        if add:
            pipes.append(Pipe(pipes[-1].x + PIPE_SPACING, display=display))

        # ── Birds update ───────────────────────────────────────────
        for b in pop:
            if b.alive:
                b.think(pipes)
                b.update(dt_s, False, sim_now, GROUND_Y)
                if check_collision(b, pipes):
                    b.alive = False
                # Simple bounding box collision for headless mode
                if not display:
                    for p in pipes:
                        bird_left = BIRD_X
                        bird_right = BIRD_X + getattr(b, 'width', 34)
                        pipe_left = p.x
                        pipe_right = p.x + getattr(p, 'width', 52)
                        # Check horizontal overlap
                        if bird_right > pipe_left and bird_left < pipe_right:
                            # Check vertical overlap with top pipe
                            if b.y < p.height:
                                b.alive = False
                            # Check vertical overlap with bottom pipe
                            if b.y + getattr(b, 'height', 24) > p.height + p.gap:
                                b.alive = False
                # Ground collision check (works for both display and headless)
                if b.y + getattr(b, 'height', 24) >= GROUND_Y:
                    b.alive = False

        # ── Drawing ────────────────────────────────────────────────
        if display:
            # draw tiled background
            for x in range(-int(bg_scroll), SCREEN_WIDTH, bg_img.get_width()):
                screen.blit(bg_img, (x, 0))

            # draw pipes
            for p in pipes:
                p.draw(screen)

            # draw birds
            for b in pop:
                if b.alive and not b.is_elite:
                    b.draw(screen)
            for b in pop:
                if b.alive and b.is_elite:
                    b.draw(screen)

            # draw tiled ground
            for x in range(-int(ground_scroll), SCREEN_WIDTH, ground_img.get_width()):
                screen.blit(ground_img, (x, GROUND_Y))

            # UI
            best = max((b.fitness for b in pop), default=0)
            txt  = font.render(f"Pipes: {best}", True, (255,255,255))
            screen.blit(txt, (10,10))

            pygame.display.flip()

        if not any(b.alive for b in pop):
            run = False

    if display:
        pygame.quit()

def next_gen(old, hall_of_fame, elite_k=3, reinject_k=3, mrate=0.01):
    graded    = sorted(old, key=lambda b: b.fitness, reverse=True)
    top_group = graded[:elite_k]
    new_pop   = []

    # elitism
    for i, parent in enumerate(top_group):
        nn = NeuralNetwork()
        nn.set_weights(parent.brain.get_weights())
        new_pop.append(Bird(brain=nn, is_elite=(i==0)))

    # reinject hall of fame
    for champ in hall_of_fame[:reinject_k]:
        weights = tuple(champ.brain.get_weights().tolist())
        if not any(tuple(b.brain.get_weights().tolist())==weights for b in new_pop):
            nn = NeuralNetwork()
            nn.set_weights(champ.brain.get_weights())
            new_pop.append(Bird(brain=nn))

    # breed children
    while len(new_pop) < len(old):
        p1, p2 = random.sample(top_group, 2)
        w1, w2  = p1.brain.get_weights(), p2.brain.get_weights()
        child = crossover(w1, w2)
        mutate(child, rate=mrate)
        nn = NeuralNetwork()
        nn.set_weights(child)
        new_pop.append(Bird(brain=nn))

    return new_pop

def main():
    parser = argparse.ArgumentParser(description="Flappy Bird AI Trainer")
    parser.add_argument("--headless", action="store_true", help="Run in headless (no graphics) mode")
    args = parser.parse_args()

    # Set VISUAL_EVERY based on headless argument
    if args.headless:
        VISUAL_EVERY = 0
    else:
        VISUAL_EVERY = 1

    POP_SIZE, GENS = 150, 40
    hall_of_fame   = []

    pop = [Bird(brain=NeuralNetwork()) for _ in range(POP_SIZE)]
    for g in range(GENS):
        do_display = (VISUAL_EVERY and ((g % VISUAL_EVERY == 0) or (g == GENS-1)))
        print(f"Gen {g+1}/{GENS} — display={'ON' if do_display else 'OFF'}")
        eval_population(pop, display=do_display)

        fits = [b.fitness for b in pop]
        print(f"  Best: {max(fits)}   Avg: {sum(fits)/len(fits):.1f}")

        # update hall of fame
        candidates = hall_of_fame + pop
        unique = {}
        for b in sorted(candidates, key=lambda b: b.fitness, reverse=True):
            key = tuple(b.brain.get_weights().tolist())
            if key not in unique and len(unique) < 5:
                unique[key] = b
        hall_of_fame = list(unique.values())

        pop = next_gen(pop, hall_of_fame, elite_k=3, reinject_k=3, mrate=0.04)

    if not VISUAL_EVERY:
        champ = max(pop, key=lambda b: b.fitness)
        eval_population([champ], display=True)

if __name__ == "__main__":
    main()
