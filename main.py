import pygame as pg
import numpy as np
import math
import numba
#import taichi as ti

#settings
res = width, height = 1200, 800
offset = np.array([1.3 * width, height]) // 2 #центр фрактала
max_iter = 30
zoom = 2.2 / height

texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size()) - 1  #получим цвета по диагонали
texture_array = pg.surfarray.array3d(texture)


class Fractal:
    def __init__(self, app) -> None:
        self.app = app
        self.screen_array = np.full((width, height, 3), [0, 0, 0], dtype=np.uint8)
        self.x = np.linspace(0, width, num=width, dtype=np.float32)   #флоутовские кооринаты начала пикселей
        self.y = np.linspace(0, height, num=height, dtype=np.float32)

    '''
    def render(self):
        
        for x in range(width):
            for y in range(height):
                c = (x - offset[0]) * zoom + 1j * (y - offset[1]) * zoom #для каждого пикселя картинки создается комплексное чило  с
                z = 0 #начальное комплексное число
                num_iter = 0
                for i in range(max_iter):
                    z = z**2 + c
                    if abs(z) > 2:# если вылетели из окружности радиуса 2, эта точка не принадлежит множеству
                        break
                    num_iter += 1
                col = int(texture_size * num_iter / max_iter)  #трюк с цветом
                self.screen_array[x, y] = texture_array[col, col]
        
        
        
        векторизация
        x = (self.x - offset[0]) * zoom #массив
        y = (self.y - offset[1]) * zoom # все возможные координаты
        c = x + 1j * y[:, None] #не одномерный вектор, а совокупность двух векторов (параллелизация), None добавляет размерность

        num_iter = np.full(c.shape, max_iter) #это больше не число, а массив пространства и времени
        z = np.empty(c.shape, np.complex64)
        for i in range(max_iter):
            mask = (num_iter == max_iter)
            z[mask] = z[mask] ** 2 + c[mask]
            num_iter[mask & (z.real ** 2 + z.imag ** 2 > 4.0)] = i + 1

        col = (num_iter.T * texture_size / max_iter).astype(np.uint8)
        self.screen_array = texture_array[col, col]
        '''
    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def render(screen_array):
        for x in numba.prange(width):
            for y in numba.prange(height):
                c = (x - offset[0]) * zoom + 1j * (y - offset[1]) * zoom  # для каждого пикселя картинки создается комплексное чило  с
                z = 0  # начальное комплексное число
                num_iter = 0
                for i in numba.prange(max_iter):
                    z = z ** 2 + c
                    if abs(z) > 2:  # если вылетели из окружности радиуса 2, эта точка не принадлежит множеству
                        break
                    num_iter += 1
                col = int(texture_size * num_iter / max_iter)  # трюк с цветом
                screen_array[x, y] = texture_array[col, col]
        return screen_array

    def update(self):
        self.screen_array = self.render(self.screen_array)



    # def update(self):
    #     self.render()


    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self):
        self.update()
        self.draw()


class App:
    def __init__(self) -> None:
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)

    def run(self):
        while True:
            self.screen.fill('black')
            self.fractal.run()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps()}')


if __name__ == '__main__':
    app = App()
    app.run()
