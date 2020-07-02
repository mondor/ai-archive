import multiprocessing
import os

def print_cube(shared):
    shared.append(os.getpid())
    print("Cube {}".format(os.getpid()))
    print(shared)
    
    
def print_square(shared):
    shared.append(os.getpid())    
    print("Square {}".format(os.getpid()))
    print(shared)
    
    
if __name__ == "__main__":
    with multiprocessing.Manager() as manager:
        shared = manager.list([])
        p1 = multiprocessing.Process(target=print_cube, args=(shared,))
        p2 = multiprocessing.Process(target=print_square, args=(shared,))
        
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        
        