
from pyglet.gl import *
from pyglet.window import mouse
from pyglet.window import key
from pyglet import shapes

options_var_1 = 3
options_var_2 = 4
offset = 100
square_size = 100
window_width = square_size*options_var_1 + 2*offset
window_height = square_size*options_var_2 + 2*offset


def draw_grid(group, batch, opt1, opt2):

    #dibujar lineas verticales dearriba hacia abajo

    y_1 = offset
    y_2 = offset + options_var_2*square_size
    width = 2
 
    # color = green
    color_list = ([0.3, 0.3, 0.3] * 2)
    #line = shapes.Line(1, 1, 2, 2, width, color=color, batch=batch) 

    for vertical in range(options_var_1+1):
        #x1 y x2 es igual, pero va cammbiando
        x_1 = offset + square_size*vertical
        x_2 = offset + square_size*vertical
        #batch.add(shapes.Line(x_1, y_1, x_2, y_2, width, color=color, batch=batch)) 
        batch.add(2, pyglet.gl.GL_LINES, group,
              ('v2i/static', (x_1, y_1, x_2, y_2)), ('c3f/static', color_list))

    x_1 = offset
    x_2 = offset + options_var_1*square_size

    for horizontal in range(options_var_2+1): 
        y_1 = offset + square_size*horizontal
        y_2 = offset + square_size*horizontal
        #line2 = shapes.Line(x_1, y_1, x_2, y_2, width, color=color, batch=batch) 
        batch.add(2, pyglet.gl.GL_LINES, group,
              ('v2i/static', (x_1, y_1, x_2, y_2)), ('c3f/static', color_list))

def draw_player(group,batch, var_1, var_2):


    circle_x = offset + var_1*square_size + (square_size/2)
    circle_y = offset + var_2*square_size + (square_size/2)
    
    # size of circle
    # color = green
    size_circle = (square_size/2) - 30
    
    # creating a circle
    circle1 = shapes.Circle(circle_x, circle_y, size_circle, color =(0, 0, 0), batch = batch, group=group)
    
    # changing opacity of the circle1
    # opacity is visibility (0 = invisible, 255 means visible)
    circle1.opacity = 250
    circle1.circle_x = offset + 2*square_size + (square_size/2)

    circle1.draw()










def draw_axis_names(group,batch, var_name_1, var_name_2):

    label = pyglet.text.Label(var_name_1,
                          font_name='Times New Roman',
                          font_size=12,
                          x=window_width//2, y=window_height-100,
                          anchor_x='center', anchor_y='center',
                          color=(0, 0, 0, 0), batch=batch, group=group)

    label1 = pyglet.text.Label(var_name_2,
                          font_name='Times New Roman',
                          font_size=12,
                          x=window_width-100, y=window_height//2,
                          anchor_x='center', anchor_y='center',
                          color=(255, 255, 255, 255), batch=batch, group=group)

    label.draw()




def draw_var_opt(var_opt_1, var_opt_2):
    #Debe dibujar los valores para cada opcion de la grilla

    return

class MainWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) #inicializa una clase superior window para acceder a sus metodos
        #self.set_minimum_size(400,300)

    def on_draw(self):
        #screen = pyglet.canvas.get_display().get_default_screen()
        #window_width = int(min(screen.width, screen.height) * 2 / 3)

        pyglet.gl.glClearColor(255, 255, 255, 255)
        window.clear()

        #pyglet.gl.glLineWidth(3)
        batch = pyglet.graphics.Batch()
        grid = pyglet.graphics.OrderedGroup(0)
        labels = pyglet.graphics.OrderedGroup(1)
        player = pyglet.graphics.OrderedGroup(2)

        # draw the grid and labels

        draw_grid(grid,batch, options_var_1, options_var_2)
        draw_axis_names(labels, batch,'cant_fundadores', 'horas_dedicacion')
        draw_player(player, batch, 2, 2)

        batch.draw()

        #self.triangle.vertices.draw(GL_TRIANGLE_FAN)

if __name__== "__main__":

                

    window = MainWindow(window_width, window_height, "PygetTutorials")
    label = pyglet.text.Label('Hola',
                          font_name ='Times New Roman',
                          font_size = 28,
                          x = 20, y = window.height//2,
                            color=(0,0,0,0) )
    window.on_draw()
    window.clear()
    label.draw()

    #window2 = MainWindow(300, 200, "hola")

    pyglet.app.run()

'''

import pygame


def grid(window, size, rows):


    distanceBtwRows = size // rows
    x = 0
    y = 0
    for l in range(rows):
        x += distanceBtwRows
        y +=distanceBtwRows

        pygame.draw.line(window, (0,0,0), (x,0), (x, size))
        pygame.draw.line(window, (0,0,0), (0,y), (size, y))

def redraw(window):
    global size, rows
    window.fill((255,255,255))
    grid(window, size, rows)
    pygame.display.update()


def main():
    global size, rows
    size = 500
    rows = 20 
    window = pygame.display.set_mode((size, size))
    play = True

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        redraw(window)


main()

'''