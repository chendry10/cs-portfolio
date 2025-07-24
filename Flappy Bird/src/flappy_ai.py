import pygame
import sys
import os
import random
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
BIRD_X = 50
FPS = 240

GRAVITY       = 400.0    # px/sec²
FLAP_VELOCITY = -200.0   # px/sec (instant jump)
PIPE_SPEED    = 120.0    # px/sec

TILT_DELAY    = 500      # ms before tilting down
PIPE_SPACING  = 200      # horizontal space between pipes

# ─── Assets Path ─────────────────────────────────────────────────────────────
def get_asset_path(fn):
    return os.path.join(os.path.dirname(__file__), "sprites", fn)

# ─── Neural Network & GA Helpers ────────────────────────────────────────────
class NeuralNetwork:
    def __init__(self, in_sz=3, hid_sz=6, out_sz=1):
        self.W1 = np.random.randn(hid_sz, in_sz)
        self.b1 = np.zeros((hid_sz, 1))
        self.W2 = np.random.randn(out_sz, hid_sz)
        self.b2 = np.zeros((out_sz, 1))

    def forward(self, x):
        x = x.reshape(-1, 1)
        a1 = np.tanh(self.W1 @ x + self.b1)
        z2 = self.W2 @ a1 + self.b2
        return 1 / (1 + np.exp(-z2))

    def get_weights(self):
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten(),
        ])

    def set_weights(self, flat):
        i = 0
        for mat in (self.W1, self.b1, self.W2, self.b2):
            size = mat.size
            mat[:] = flat[i:i+size].reshape(mat.shape)
            i += size

def crossover(w1, w2):
    mask = np.random.rand(len(w1)) < 0.5
    return np.where(mask, w1, w2)

def mutate(w, rate=0.1, scale=0.5):
    for i in range(len(w)):
        if random.random() < rate:
            w[i] += np.random.randn() * scale

# ─── Game Entities ───────────────────────────────────────────────────────────
class Bird:
    def __init__(self, brain=None, is_elite=False):
        self.y = SCREEN_HEIGHT / 2
        self.speed = 0.0
        self.angle = 0
        self.last_flap_time = 0

        # unchanged sprite load
        self.sprite = pygame.transform.scale(
            pygame.image.load(get_asset_path('flap.png')),
            (45, 45)
        )

        self.brain = brain or NeuralNetwork()
        self.alive = True
        self.fitness = 0
        self.is_elite = is_elite

    def flap(self):
        self.speed = FLAP_VELOCITY
        now = pygame.time.get_ticks()
        self.last_flap_time = now
        self.angle = min(self.angle + 32, 25)

    def think(self, pipes):
        nxt = next((p for p in pipes if p.x + p.width > BIRD_X), None)
        if not nxt: return
        inp = np.array([
            self.y / SCREEN_HEIGHT,
            (nxt.height + nxt.gap/2) / SCREEN_HEIGHT,
            (nxt.x - BIRD_X) / SCREEN_WIDTH
        ])
        if self.brain.forward(inp)[0,0] > 0.5:
            self.flap()

    def update(self, dt_s, space_pressed, now, GROUND_Y):
        self.speed += GRAVITY * dt_s
        self.y += self.speed * dt_s

        # die if you hit ground line or fly off top
        if self.y > GROUND_Y - self.sprite.get_height() or self.y < 0:
            self.alive = False

        if space_pressed:
            self.last_flap_time = now
            self.angle = min(self.angle + 32, 25)
        elif now - self.last_flap_time > TILT_DELAY:
            self.angle = max(self.angle - 1, -25)

    def draw(self, screen):
        rot = pygame.transform.rotate(self.sprite, self.angle)
        rect = rot.get_rect(
            center=self.sprite.get_rect(topleft=(BIRD_X, int(self.y))).center
        )
        screen.blit(rot, rect.topleft)
        if self.is_elite:
            pygame.draw.circle(
                screen, (255,0,0), rect.center,
                max(rect.width, rect.height)//2 + 4, 3
            )

    def get_mask(self):
        rot = pygame.transform.rotate(self.sprite, self.angle)
        rect = rot.get_rect(
            center=self.sprite.get_rect(topleft=(BIRD_X, int(self.y))).center
        )
        return pygame.mask.from_surface(rot), rect
class Pipe:
    # Minimum visible bottom‐pipe height (above the ground)
    MIN_BOTTOM_HEIGHT = 50  

    def __init__(self, x):
        self.x = x
        self.width = 50
        self.passed = False

        # ── Load ground to compute its height ──────────────────────
        ground_img = pygame.image.load(get_asset_path('ground.png')).convert_alpha()
        ground_h   = ground_img.get_height()

        # ── Pick height & gap such that bottom pipe ≥ MIN_BOTTOM_HEIGHT ──
        while True:
            height = random.randint(200, 350)
            gap    = random.randint(135, 200)
            bottom_h = SCREEN_HEIGHT - (height + gap) - ground_h
            if bottom_h >= Pipe.MIN_BOTTOM_HEIGHT:
                break

        self.height = height
        self.gap    = gap

        # ── Top pipe ───────────────────────────────────────────────
        top_raw = pygame.image.load(get_asset_path('pipe_top.png')).convert_alpha()
        self.top_img  = pygame.transform.scale(top_raw, (self.width, self.height))
        self.top_mask = pygame.mask.from_surface(self.top_img)

        # ── Bottom pipe ────────────────────────────────────────────
        bot_raw = pygame.image.load(get_asset_path('pipe_bottom.png')).convert_alpha()
        self.bottom_img  = pygame.transform.scale(bot_raw, (self.width, bottom_h))
        self.bottom_mask = pygame.mask.from_surface(self.bottom_img)

    def update(self, dt_s):
        self.x -= PIPE_SPEED * dt_s

    def draw(self, screen):
        # Draw top at the top of the screen
        screen.blit(self.top_img, (self.x, 0))
        # Draw bottom just below the gap
        screen.blit(self.bottom_img, (self.x, self.height + self.gap))



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

    pipes = [Pipe(SCREEN_WIDTH + i * PIPE_SPACING) for i in range(5)]
    bg_scroll = ground_scroll = 0
    scroll_speed = PIPE_SPEED

    run = True
    while run and any(b.alive for b in pop):
        if display:
            dt_ms = clock.tick(FPS)
            sim_now = pygame.time.get_ticks()
        else:
            dt_ms = 20000.0 / SIM_FPS
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
            pipes.append(Pipe(pipes[-1].x + PIPE_SPACING))

        # ── Birds update ───────────────────────────────────────────
        for b in pop:
            if b.alive:
                b.think(pipes)
                b.update(dt_s, False, sim_now, GROUND_Y)
                if check_collision(b, pipes):
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
    POP_SIZE, GENS = 150, 40
    VISUAL_EVERY   = 1
    hall_of_fame   = []

    pop = [Bird() for _ in range(POP_SIZE)]
    for g in range(GENS):
        do_display = (g % VISUAL_EVERY == 0) or (g == GENS-1)
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

    if not do_display:
        champ = max(pop, key=lambda b: b.fitness)
        eval_population([champ], display=True)

if __name__ == "__main__":
    main()
