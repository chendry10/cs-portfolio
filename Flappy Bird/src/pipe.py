import pygame
import os
import random

class Pipe:
    """
    Represents a pipe obstacle in Flappy Bird.
    Handles both visual and headless modes.
    """

    def __init__(self, x, display=True, screen_height=600, ground_height=100):
        """
        Initialize a Pipe obstacle.

        Args:
            x (float): Horizontal position of the pipe.
            display (bool): Whether to load and display the pipe images.
        """
        self.display = display
        self.x = x
        self.width = 60
        self.passed = False

        # Always randomize height and gap
        while True:
            self.height = random.randint(200, 350)
            self.gap = random.randint(120, 170)
            bottom_h = screen_height - (self.height + self.gap) - ground_height
            if bottom_h >= 50:  # MIN_BOTTOM_HEIGHT
                break

        if display:
            self.top_img = pygame.image.load(os.path.join(os.path.dirname(__file__), "sprites", "pipe_top.png")).convert_alpha()
            self.top_img = pygame.transform.scale(self.top_img, (self.width, self.height))
            self.top_mask = pygame.mask.from_surface(self.top_img)
            self.bottom_img = pygame.image.load(os.path.join(os.path.dirname(__file__), "sprites", "pipe_bottom.png")).convert_alpha()
            self.bottom_img = pygame.transform.scale(self.bottom_img, (self.width, bottom_h))
            self.bottom_mask = pygame.mask.from_surface(self.bottom_img)
        else:
            self.top_img = None
            self.bottom_img = None
            self.top_mask = pygame.mask.Mask((self.width, self.height))
            self.bottom_mask = pygame.mask.Mask((self.width, bottom_h))

    def update(self, dt):
        """
        Updates the pipe's horizontal position.

        Args:
            dt (float): Time delta in seconds.
        """
        self.x -= 120 * dt

    def draw(self, screen):
        """
        Draws the pipe on the given screen (only in visual mode).

        Args:
            screen (pygame.Surface): The game screen to draw on.
        """
        if self.display and self.top_img and self.bottom_img:
            screen.blit(self.top_img, (self.x, 0))
            screen.blit(self.bottom_img, (self.x, self.height + self.gap))
