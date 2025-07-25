import pygame
import os


class Bird:
    """
    Represents a Flappy Bird agent controlled by a neural network.

    Attributes:
        y (float): Vertical position of the bird.
        speed (float): Current vertical speed.
        angle (int): Current tilt angle for drawing.
        last_flap_time (int): Time of last flap (ms).
        sprite (pygame.Surface): Bird image.
        brain: Neural network controlling the bird.
        alive (bool): Whether the bird is alive.
        fitness (int): Fitness score for evolution.
        is_elite (bool): Marks if bird is elite in population.
        flap_velocity (float): Velocity applied when flapping.
        gravity (float): Gravity constant.
        tilt_delay (int): Delay before tilting down (ms).
        bird_x (int): Horizontal position for drawing.
        screen_height (int): Height of the game screen.
    """
    def __init__(self, brain, is_elite=False, screen_height=600, bird_x=50, flap_velocity=-200.0, gravity=400.0, tilt_delay=500):
        """
        Initialize a Bird agent.

        Args:
            brain: Neural network controlling the bird's actions.
            is_elite (bool): Whether the bird is elite in the population.
            screen_height (int): Height of the game screen.
            bird_x (int): Horizontal position for drawing.
            flap_velocity (float): Velocity applied when flapping.
            gravity (float): Gravity constant.
            tilt_delay (int): Delay before tilting down (ms).
        """
        self.y = screen_height / 2
        self.speed = 0.0
        self.angle = 0
        self.last_flap_time = 0
        self.sprite = pygame.transform.scale(
            pygame.image.load(os.path.join(os.path.dirname(__file__), "sprites", 'flap.png')),
            (45, 45)
        )
        self.brain = brain
        self.alive = True
        self.fitness = 0
        self.is_elite = is_elite
        self.flap_velocity = flap_velocity
        self.gravity = gravity
        self.tilt_delay = tilt_delay
        self.bird_x = bird_x
        self.screen_height = screen_height

    def flap(self):
        """
        Instantly applies upward velocity and tilts the bird up.
        """
        self.speed = self.flap_velocity
        now = pygame.time.get_ticks()
        self.last_flap_time = now
        self.angle = min(self.angle + 32, 25)

    def think(self, pipes):
        """
        Decides whether to flap based on neural network output and pipe positions.

        Args:
            pipes (list): List of Pipe objects in the game.
        """
        nxt = next((p for p in pipes if p.x + p.width > self.bird_x), None)
        if not nxt: return
        import numpy as np
        inp = np.array([
            self.y / self.screen_height,
            (nxt.height + nxt.gap/2) / self.screen_height,
            (nxt.x - self.bird_x) / 400
        ])
        if self.brain.forward(inp)[0,0] > 0.5:
            self.flap()

    def update(self, dt_s, space_pressed, now, ground_y):
        """
        Updates the bird's position, speed, and angle.

        Args:
            dt_s (float): Time delta in seconds.
            space_pressed (bool): Whether the space key is pressed.
            now (int): Current time in ms.
            ground_y (int): Y position of the ground.
        """
        self.speed += self.gravity * dt_s
        self.y += self.speed * dt_s
        if self.y > ground_y - self.sprite.get_height() or self.y < 0:
            self.alive = False
        if space_pressed:
            self.last_flap_time = now
            self.angle = min(self.angle + 32, 25)
        elif now - self.last_flap_time > self.tilt_delay:
            self.angle = max(self.angle - 1, -25)

    def draw(self, screen):
        """
        Draws the bird on the screen, with rotation and elite highlight if applicable.

        Args:
            screen (pygame.Surface): The game screen to draw on.
        """
        rot = pygame.transform.rotate(self.sprite, self.angle)
        rect = rot.get_rect(
            center=self.sprite.get_rect(topleft=(self.bird_x, int(self.y))).center
        )
        screen.blit(rot, rect.topleft)
        if self.is_elite:
            pygame.draw.circle(
                screen, (255,0,0), rect.center,
                max(rect.width, rect.height)//2 + 4, 3
            )

    def get_mask(self):
        """
        Returns a mask and rect for pixel-perfect collision detection.

        Returns:
            tuple: (pygame.Mask, pygame.Rect)
        """
        rot = pygame.transform.rotate(self.sprite, self.angle)
        rect = rot.get_rect(
            center=self.sprite.get_rect(topleft=(self.bird_x, int(self.y))).center
        )
        return pygame.mask.from_surface(rot), rect
