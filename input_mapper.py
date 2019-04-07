import time
import pyautogui as pag
#import win32api

"""
NOTES
In CSGO, raw mouse input must be turned off.
(Probably gets straight from IO when raw, not through OS)

"""
class InputMapper:
    def __init__(self):
        self.screen_size = pag.size()
        self.screen_centerx = self.screen_size[0] / 2
        self.screen_centery = self.screen_size[1] / 2
        pag.FAILSAFE = False

    def move_mouse(self, pos):
        #print("Coordinates given - x: {0}, y: {1}".format(pos[0], pos[1]))
        dx, dy = self.distance_from_crosshairs(pos[0], pos[1])
        print("Moving by - dx: {0}, dy: {1}".format(dx, dy))
        pag.move(dx, dy)
        #pag.moveTo(int(pos[0]), int(pos[1]))
        #win32api.mouse_event(int(x), int(y), 0x0001 | 0x8000)

    # Return the difference in x and y pixels from the target point to the center of the screen
    def distance_from_crosshairs(self, x, y):
        dx = x - self.screen_centerx
        dy = y - self.screen_centery
        #c = pag.position()
        #dx = x - c[0]
        #dy = y - c[1]
        return int(dx), int(dy)

    def get_center(self):
        return self.screen_centerx, self.screen_centery

    def click(self):
        pag.click()

    def burst_fire(self, duration):
        pag.mouseDown()
        time.sleep(3)
        pag.mouseUp()



