import argparse
import time

import imageio
import numpy as np
import pygame
from PIL import Image

import langroom


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=None)
  parser.add_argument('--view', type=int, default=2)  # 6 for fully observed
  parser.add_argument('--length', type=int, default=200)
  parser.add_argument('--window', type=int, nargs=2, default=(800, 200))
  parser.add_argument('--resolution', type=int, default=64)
  parser.add_argument('--fps', type=int, default=2)
  args = parser.parse_args()

  keymap = {
      pygame.K_s: {'move': 1, 'talk': 0},
      pygame.K_w: {'move': 2, 'talk': 0},
      pygame.K_d: {'move': 3, 'talk': 0},
      pygame.K_a: {'move': 4, 'talk': 0},
      pygame.K_1: {'move': 0, 'talk': 1},
      pygame.K_2: {'move': 0, 'talk': 2},
      pygame.K_3: {'move': 0, 'talk': 3},
      pygame.K_4: {'move': 0, 'talk': 4},
      pygame.K_5: {'move': 0, 'talk': 5},
      pygame.K_6: {'move': 0, 'talk': 6},
      pygame.K_7: {'move': 0, 'talk': 7},
      pygame.K_8: {'move': 0, 'talk': 8},
      pygame.K_9: {'move': 0, 'talk': 9},
      pygame.K_0: {'move': 0, 'talk': 10},
  }

  env = langroom.LangRoom(
      view=args.view,
      length=args.length,
      resolution=args.resolution,
      seed=args.seed,
  )

  pygame.init()
  screen = pygame.display.set_mode(args.window)
  clock = pygame.time.Clock()
  running = True
  while running:

    action = None
    pygame.event.pump()
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False
      elif event.type == pygame.KEYDOWN and event.key in keymap.keys():
        action = keymap[event.key]
    if action is None:
      pressed = pygame.key.get_pressed()
      for key, action in keymap.items():
        if pressed[key]:
          break
      else:
        action = {'move': 0, 'talk': 0}

    action['reset'] = False
    obs = env.step(action)

    image = obs['log_image']
    image = np.array(Image.fromarray(image).resize(
        (args.window[0], args.window[1]), Image.NEAREST))
    image = image.swapaxes(0, 1)
    surface = pygame.surfarray.make_surface(image)
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(args.fps)

  pygame.quit()


if __name__ == '__main__':
  main()
