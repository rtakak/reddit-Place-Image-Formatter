from place_formatter import place
import os 
import time
clear = lambda: os.system('cls')


menu_options = {
    0: 'Requirements',
    1: 'Tutorial',
    2: 'Make my Sprite',
    3: 'Credit',
    4: 'Exit',
}

def print_menu():
    print('|| Reddit Place Image Formatter ||')
    print('')    
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )
        print('')

def option1():
     clear()
     print('')
     time.sleep(1)
     print("To create a pixelated image it's very simple ! (Really)")
     print("Simply add your image in the software file source.")
     print('***It is recommended to use a png image')
     time.sleep(5)
     print('')
     print('')

def option0():
     clear()
     print('')
     time.sleep(1)
     print("||| Mandatory dependencies |||")
     print('numpy = 1.22.3')
     print('')
     print("opencv_python = 4.5.5.64")
     print('')
     print('scikit_learn = 1.0.2')
     time.sleep(5)
     print('')
     print('')

# IMPORTANT
def option2():
     clear()
     nameFile = ''
     extensionFile = ''
     print('')
     nameFile = input('Please enter the name of your file : ')
     print('')
     extensionFile = input('Please enter the extension of your file : ')
     PATH = './' + nameFile + '.' + extensionFile
     if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print("We will now update your file")
        image_path = r""+ nameFile + '.' + extensionFile # name of file if it is in same directory or path to file
        width_pixel_size = 64  # size of width of output by pixels
        show_grids_at_output = True  # option to turn on or off grids at output
        place(image_path, width_pixel_size, show_grids_at_output)  # main function call
     else:
        print("Either the file is missing or not readable")
        time.sleep(3)
        option2()

def option3():
     clear()
     print('')
     time.sleep(1)
     print('Created by : rtakak & mikesingleton')
     print('')
     print('Modify by : Maxime66410')
     time.sleep(3)
     print('')
     print('')

if __name__ == "__main__":
 while(True):
        print_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            clear()
            print('')
            print('Wrong input. Please enter a number ...')
            print('')
        if option == 1:
           option1()
        elif option == 2:
            option2()
        elif option == 3:
            option3()
        elif option == 4:
            clear()
            print('Thank you for using the software and contributing to reddit Place !')
            time.sleep(3)
            exit()
        elif option == 0:
            option0()
        else:
            clear()
            print('')
            print('Invalid option. Please enter a number between 1 and 4.')
            print('')


