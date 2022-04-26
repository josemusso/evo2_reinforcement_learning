
import multiprocessing  
import time

def function():
    print('perrito')


def square_list(running, x, y):
    # import pygame package
    import pygame
    
    # initializing imported module
    pygame.init()
    
    # displaying a window of height
    # 500 and width 400
    pygame.display.set_mode((400, 500))
    
    # Setting name for window
    pygame.display.set_caption(str(x.value))
    
    # creating a bool value which checks
    # if game is running
    running.value = True

    function()
    # Game loop
    # keep game running till running is true
    while running.value:
    
        # Check for event if user has pushed
        # any event in queue
        for event in pygame.event.get():
        
            # if event is of type quit then set
            # running bool to false
            if event.type == pygame.QUIT:
                print('finalizado')
                running = False
        pygame.display.set_caption(str(x.value))
        pygame.display.update()
  
if __name__ == "__main__":
    # input list
    running = multiprocessing.Value('i')
  
  
    # creating Value of int data type
    x = multiprocessing.Value('i')
    y = multiprocessing.Value('i')

    x.value = 1
    y.value = 2
  
    # creating new process
    p1 = multiprocessing.Process(target=square_list, args=(running, x, y))
  
    # starting process
    p1.start()

    time.sleep(6)

    x.value = 3000

    running.value = 0

'''
  
def square_list(mylist, result, square_sum):
    """
    function to square a given list
    """
    # append squares of mylist to result array
    for idx, num in enumerate(mylist):
        result[idx] = num * num
  
    # square_sum value
    square_sum.value = sum(result)
  
    # print result Array
    print("Result(in process p1): {}".format(result[:]))
  
    # print square_sum Value
    print("Sum of squares(in process p1): {}".format(square_sum.value))
  
if __name__ == "__main__":
    # input list
    mylist = [1,2,3,4]
  
    # creating Array of int data type with space for 4 integers
    result = multiprocessing.Array('i', 4)
  
    # creating Value of int data type
    square_sum = multiprocessing.Value('i')
  
    # creating new process
    p1 = multiprocessing.Process(target=square_list, args=(mylist, result, square_sum))
  
    # starting process
    p1.start()
  
    # wait until the process is finished
    p1.join()
  
    # print result array
    print("Result(in main program): {}".format(result[:]))
  
    # print square_sum Value
    print("Sum of squares(in main program): {}".format(square_sum.value))
    '''