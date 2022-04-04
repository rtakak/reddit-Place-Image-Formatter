from math import sqrt
import cv2
import numpy as np
from sklearn.cluster import KMeans


def color_clustering(idx, img, k):
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])

    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])

    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))

    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]

    return imgC


def kmeans_image(image, k):
    idx = segment_img(image, k)
    return color_clustering(idx, image, k)


def segment_img(img, k):
    imgC = np.copy(img)

    h = img.shape[0]
    w = img.shape[1]

    imgC.shape = (img.shape[0] * img.shape[1], 3)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_
    kmeans.shape = (h, w)

    return kmeans


def pixelate(img, bit_res, k):
    width = int(img.shape[1])
    height = int(img.shape[0])
    img = cv2.resize(
        img, (bit_res, int(bit_res * height / width)), interpolation=cv2.INTER_LINEAR
    )
    img = cv2.resize(
        img, (bit_res, int(bit_res * height / width)), interpolation=cv2.INTER_NEAREST
    )
    return kmeans_image(img, k)


def dict_palette(bgr_colors, color_names):
    palette_colors = {}
    for color, name in zip(bgr_colors, color_names):
        palette_colors[name] = color

    start = 0
    line = 0
    rect = np.zeros((100, 600, 3), dtype=np.uint8)
    for color in palette_colors:
        end = start + 50
        cv2.rectangle(
            rect,
            (int(start), line + 0),
            (int(end), line + 50),
            palette_colors[color],
            -1,
        )
        if end == 600:
            start = 0
            line = 50
        else:
            start = end

    return palette_colors, rect


def encountered_colors(img):
    encountered_colors_list = []
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if not any(
                    [
                        (img[x, y] == countered_color).all()
                        for countered_color in encountered_colors_list
                    ]
            ):
                b, g, r = img[x, y]
                encountered_colors_list.append([b, g, r])
    return encountered_colors_list


def match_colors_w_palette(encountered_colors_list, palette, DEBUG, color_space="LAB"):
    recoloring = []
    encountered_color_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    palette_color_bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    np.set_printoptions(precision=1)

    for encountered_color in encountered_colors_list:
        comparison = []
        for palette_color in palette:
            palette_color_bgr[:, :] = palette[palette_color]
            encountered_color_bgr[:, :] = encountered_color
            if color_space == "LAB":
                palette_color_lab = cv2.cvtColor(
                    palette_color_bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab
                )
                encountered_color_lab = cv2.cvtColor(
                    encountered_color_bgr.astype(np.float32) / 255, cv2.COLOR_BGR2Lab
                )
                l1, a1, b1 = encountered_color_lab[0][0]
                l2, a2, b2 = palette_color_lab[0][0]
                diff = sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)
            elif color_space == "HSV":
                palette_color_lab = cv2.cvtColor(
                    palette_color_bgr.astype(np.float32) / 255, cv2.COLOR_BGR2HSV
                )
                encountered_color_lab = cv2.cvtColor(
                    encountered_color_bgr.astype(np.float32) / 255, cv2.COLOR_BGR2HSV
                )
                h1, s1, v1 = encountered_color_lab[0][0]
                h2, s2, v2 = palette_color_lab[0][0]
                diff = sqrt((h2 - h1) ** 2 + (s2 - s1) ** 2 + (v2 - v1) ** 2)
            else:
                raise ValueError
            if DEBUG:
                print(
                    f"encountered: {encountered_color_lab[0][0]}, palette {palette_color}:\t{palette_color_lab[0][0]}, diff: {diff:.1f}"
                )
            comparison.append([diff, palette_color])
        comparison.sort()
        if DEBUG:
            print(comparison[0][1])
            print(
                f"best pick, encountered lab, bgr: {encountered_color_lab[0][0]}, {encountered_color}, color {comparison[0][1]}, {palette[comparison[0][1]]}, diff: {comparison[0][0]:.1f}\n--------------------------"
            )
        recoloring.append(
            [
                encountered_color,
                comparison[0][1],
                palette[comparison[0][1]],
                int(comparison[0][0]),
            ]
        )
    return recoloring


def recolor(img, recoloring, n, DEBUG):
    matched_rect = np.zeros((66, 33 * n, 3), dtype=np.uint8)
    start = 0
    for recolor in recoloring:
        if DEBUG:
            print(recolor)
        end = start + 33
        cv2.rectangle(
            matched_rect, (start, 0), (end, 33), list(map(int, recolor[0])), -1
        )
        cv2.rectangle(
            matched_rect, (start, 33), (end, 66), list(map(int, recolor[2])), -1
        )
        start = end
        recolor_index = np.where(img[:, :] == recolor[0])

        img[np.all(img == recolor[0], axis=-1)] = list(map(int, recolor[2]))
    return img, matched_rect


def place(
        image_path,
        width_size,
        grid=True,
        show_process_detail=False,
        color_n=24,
        DEBUG=False,
):
    bgr_colors = [
        [26, 0, 109],
        [57, 0, 190],
        [0, 69, 255],
        [0, 168, 255],
        [53, 214, 255],
        [184, 248, 255],
        [104, 163, 0],
        [120, 204, 0],
        [86, 237, 126],
        [111, 117, 0],
        [170, 158, 0],
        [192, 204, 0],
        [164, 80, 36],
        [234, 144, 54],
        [244, 233, 81],
        [193, 58, 73],
        [255, 92, 106],
        [255, 179, 148],
        [159, 30, 129],
        [192, 74, 180],
        [255, 171, 228],
        [127, 16, 222],
        [129, 56, 255],
        [170, 153, 255],
        [47, 72, 109],
        [38, 105, 156],
        [112, 180, 255],
        [0, 0, 0],
        [82, 82, 81],
        [144, 141, 137],
        [255, 255, 255],
    ]
    color_names = [
        "burgundy",
        "dark red",
        "red",
        "orange",
        "yellow",
        "pale yellow",
        "dark green",
        "green",
        "light green",
        "dark teal",
        "teal",
        "light teal",
        "dark blue",
        "blue",
        "light blue",
        "indigo",
        "periwinkle",
        "lavender",
        "dark purple",
        "purple",
        "pale purple",
        "magenta",
        "pink",
        "light pink",
        "dark brown",
        "brown",
        "beige",
        "black",
        "dark gray",
        "gray",
        "light gray",
        "white",
    ]

    original = cv2.imread(image_path)
    pixel = pixelate(original, width_size, color_n)
    output = pixel.copy()

    palette, palette_visual = dict_palette(bgr_colors, color_names)

    encountered_colors_list = encountered_colors(output)
    if DEBUG:
        print(palette)
        print(encountered_colors_list)
    recoloring = match_colors_w_palette(encountered_colors_list, palette, DEBUG)
    output, matches_visual = recolor(
        output, recoloring, len(encountered_colors_list), DEBUG
    )

    width = int(output.shape[1])
    height = int(output.shape[0])
    dim = (width * 10, height * 10)
    pixel = cv2.resize(pixel, dim, interpolation=cv2.INTER_AREA)
    output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)
    if grid:
        color = (0, 0, 0)
        thickness = 1
        for x in range(width):
            start_point = [10 * x, 0]
            end_point = [10 * x, height * 10]
            output = cv2.line(output, start_point, end_point, color, thickness)
        for y in range(height):
            start_point = [0, 10 * y]
            end_point = [width * 10, 10 * y]
            output = cv2.line(output, start_point, end_point, color, thickness)

    if show_process_detail:
        cv2.imshow("Palette", palette_visual)
        cv2.imshow("Color Matches", matches_visual)
        cv2.imshow("pixel", pixel)
        cv2.imshow("matched palette", output)
        cv2.imshow("original", original)
        cv2.waitKey(0)
    name = image_path.split("\\")
    name = "pixel_" + name[-1]
    cv2.imwrite(name, output)
    print("My Job is done, bye!")


if __name__ == "__main__":
    image_path = r"user2.png"  # name of file if it is in same directory or path to file
    width_pixel_size = 48  # size of width of output by pixels
    show_grids_at_output = True  # option to turn on or off grids at output
    place(image_path, width_pixel_size, show_grids_at_output)  # main function call
