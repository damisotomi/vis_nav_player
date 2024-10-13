from vis_nav_game import Player, Action
import pygame
import cv2
import matplotlib.pyplot as plt


class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.movements = []
        self.position = [0, 0]  # Starting position on the grid
        self.direction = [0, 1]  # Initial direction (facing "up", north)
        self.consecutive_presses = 0  # Track consecutive presses
        self.last_key_pressed = None  # Track the last key pressed
        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
            pygame.K_p: 'PLOT'  # Add the 'P' key to trigger plotting
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                self.plot_movements()  # Call this function on quit
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    action = self.keymap[event.key]

                    # Handle the case for 'P' key to trigger plotting
                    if action == 'PLOT':
                        self.plot_movements()  # Plot movements when 'P' is pressed
                    else:
                        # Handle consecutive key presses
                        if self.last_key_pressed == action:
                            self.consecutive_presses += 1
                        else:
                            self.consecutive_presses = 1
                            self.last_key_pressed = action

                        # Special handling for forward/backward and left/right
                        if action == Action.FORWARD or action == Action.BACKWARD:
                            self.update_position_cumulative(action)
                        elif action == Action.LEFT or action == Action.RIGHT:
                            self.update_rotation_cumulative(action)

                        self.last_act = action  # Store the last action, don't use |= here for strings

                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap and self.keymap[event.key] != 'PLOT':
                    self.last_act = Action.IDLE  # Reset to IDLE when key is released
        return self.last_act


    def update_position_cumulative(self, action):
        if self.consecutive_presses >= 3:
            # Only move after 3 consecutive presses
            if action == Action.FORWARD:
                self.position[0] += self.direction[0]
                self.position[1] += self.direction[1]
            elif action == Action.BACKWARD:
                self.position[0] -= self.direction[0]
                self.position[1] -= self.direction[1]

            # Log the movement
            self.movements.append(action)

            # Reset the consecutive presses after movement
            self.consecutive_presses = 0

    def update_rotation_cumulative(self, action):
        if self.consecutive_presses >= 7:  # Only count the rotation if pressed at least 7 times
            if 7 <= self.consecutive_presses < 25:
                # Rotate 90 degrees
                if action == Action.LEFT:
                    self.direction = [-self.direction[1], self.direction[0]]  # Turn left
                elif action == Action.RIGHT:
                    self.direction = [self.direction[1], -self.direction[0]]  # Turn right
            elif self.consecutive_presses >= 25:
                # Rotate 180 degrees
                self.direction = [-self.direction[0], -self.direction[1]]

            # Log the rotation
            self.movements.append(action)

            # Reset the consecutive presses after rotation
            self.consecutive_presses = 0
        else:
            # Do not count the rotation if consecutive presses are less than 7
            return


    def plot_movements(self):
        # Starting position
        x, y = [0], [0]

        # Recalculate positions based on movements
        pos = [0, 0]
        direction = [0, 1]  # Initially facing up (north)

        for action in self.movements:
            if action == Action.FORWARD:
                pos[0] += direction[0]
                pos[1] += direction[1]
            elif action == Action.BACKWARD:
                pos[0] -= direction[0]
                pos[1] -= direction[1]
            elif action == Action.LEFT:
                direction = [-direction[1], direction[0]]  # Turn left
            elif action == Action.RIGHT:
                direction = [direction[1], -direction[0]]  # Turn right

            x.append(pos[0])
            y.append(pos[1])

        # Plot the movements
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker='o', color='r', linestyle='-', linewidth=2)
        plt.quiver(x[:-1], y[:-1], [x2 - x1 for x1, x2 in zip(x[:-1], x[1:])],
                   [y2 - y1 for y1, y2 in zip(y[:-1], y[1:])], angles='xy', scale_units='xy', scale=1)

        plt.title('Player Movement Trajectory')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('trajectory_plot.png')
        plt.show()

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]

        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.imwrite('target.jpg', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    def pre_navigation(self) -> None:
        pass

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import logging
    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    import vis_nav_game as vng
    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')
    vng.play(the_player=KeyboardPlayerPyGame())
