import time
import pyautogui as pag


class InputMapper:
    def __init__(self):
        self.screen_size = pag.size()
        self.screen_centerx = self.screen_size[0] / 2
        self.screen_centery = self.screen_size[1] / 2
        self.crop = (0, 0)
        pag.FAILSAFE = False

    def move_mouse(self, pos):
        """Moves the mouse to the passed pixel coordinates."""
        dx, dy = self.distance_from_crosshairs(pos[0], pos[1])
        pag.move(dx, dy)

    def distance_from_crosshairs(self, x, y):
        """Return the difference in x and y pixels from the target point to the center of the screen.
        The screen center is calculated from the crop tuple set by set_crop()."""
        x += self.crop[0]
        y += self.crop[1]
        dx = x - self.screen_centerx
        dy = y - self.screen_centery
        return int(dx), int(dy)

    def set_crop(self, crop):
        """Sets the pixels being cropped from the (width, height) of each side of the screen
        to calculate the screen center with respect to the cropped image."""
        self.crop = crop

    def burst_fire(self, duration):
        pag.mouseDown()
        time.sleep(1)
        pag.mouseUp()
