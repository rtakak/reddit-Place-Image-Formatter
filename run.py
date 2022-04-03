from place_formatter import place

if __name__ == "__main__":
    image_path = r"user2.png"  # name of file if it is in same directory or path to file
    width_pixel_size = 48  # size of width of output by pixels
    show_grids_at_output = True  # option to turn on or off grids at output
    place(image_path, width_pixel_size, show_grids_at_output)  # main function call
