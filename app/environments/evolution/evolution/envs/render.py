def grid(window, size, rows, cols):

    offset = size #offset grilla

    total_y = size * rows #tamano total eje y
    total_x = size * cols #tamano total eje x
    # distanceBtwRows = size // rows  esto es size ahora

    x = offset
    y = offset

    print('dibujando lineas horizontales')
    cont_cols = 0
    cont_rows = 0
    for l in range(rows+1):
        print('x1', 'y1',x, y)
        print('x2', 'y2',total_x, y)


        pygame.draw.line(window, (0,0,0), (x,y), (offset + total_x, y))
        y += size
        cont_cols+=1

    x = offset
    y = offset

    print('dibujando lineas verticales')

    for z in range(cols+1):

        print('x1', 'y1',x, y)
        print('x2', 'y2',x, total_y)

        pygame.draw.line(window, (0,0,0), (x,y), (x, offset + total_y))
        x += size
        cont_rows+=1



def circle(window, size, var_1, var_2):


    circle_x = size + var_2*size + (size/2)
    circle_y = size + var_1*size + (size/2)
    pygame.draw.circle(window, (0, 255, 0),
                   [circle_x, circle_y], 10, 2)

def label (window, size, rows_name, cols_name):

    myfont = pygame.font.SysFont("monospace", 15)

    # render text
    label = myfont.render(cols_name, 1, (0,0,0))
    label2 = myfont.render(rows_name, 2, (0,0,0))
    window.blit(label, (size, size/4))
    window.blit(label2, (0, size))

def options(window, size, options_1, options_2):


    myfont =  pygame.font.SysFont("monospace", 10)
    x = size
    y = size/2
    for i, value in enumerate(options_2):
        label = myfont.render(value, 1, (0,0,0))
        window.blit(label, (x*(i+1) + size/2, y))

    x = size/2
    y = size

    for i, value in enumerate(options_1):
        label = myfont.render(value, 1, (0,0,0))
        window.blit(label, (x, y*(i+1) + size/2))

def draw_grid(window, size, row, cols, labels, opt1, opt2, x, y):

    window.fill((255,255,255))
    grid(window, size, rows, cols)
    circle(window, size, x, y)
    label(window, size, labels[0], labels[1])
    options(window, size, opt1, opt2)
    pygame.display.update()


def game(running,size, rows, cols, x, y):
    window_width = cols*size + 2*size
    window_height = rows*size + 2*size 
    pygame.init()

    self.window = pygame.display.set_mode((window_width, window_height))
    labels = [self.board1.get_var_1(), self.board1.get_var_2()]
    opt1 = self.board1.get_options_var_1()
    opt2 = self.board1.get_options_var_2()
    draw_grid(self.window, size, rows, cols, labels, opt1, opt2, 0,0)

    while running.value:
    
        # Check for event if user has pushed
        # any event in queue
        for event in pygame.event.get():
        
            # if event is of type quit then set
            # running bool to false
            if event.type == pygame.QUIT:
                print('finalizado')
                running.value = False
        draw_grid(self.window, size, rows, cols, labels, opt1, opt2, x.value,y.value)

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