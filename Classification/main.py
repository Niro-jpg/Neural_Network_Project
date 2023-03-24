from Test import Test
from Plots import Plot
from Load import Load

def main():
   
   if "-t":
      Test()
   
   elif "-i":
      Plot()
      
   
   elif "-r":
      Load()


if __name__ == "__main__":
   main()

