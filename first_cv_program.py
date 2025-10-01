import cv2

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'Координаты точки: {x}, {y}')
        b = image[y, x, 0]
        g = image[y, x, 1]
        r = image[y, x, 2]
        print(f'BGR-код точки: {b}, {g}, {r}\n')

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f'{x}, {y}', (x, y),
                    font, 1, (0, 0, 0), 2)
        cv2.imshow('image', image)

    if event == cv2.EVENT_RBUTTONDOWN:
        b = image[y, x, 0]
        g = image[y, x, 1]
        r = image[y, x, 2]

        print(f'Координаты точки: {x}, {y}')
        print(f'BGR-код точки: {b}, {g}, {r}\n')

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f'{b}, {g}, {r}', (x, y),
                    font, 1, (0, 0, 0), 2)
        cv2.imshow('image', image)

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier('cv_xmls/haarcascade_frontalface_default.xml')

    image_2 = cv2.imread('images/test_image_4.jpg')
    gray_image = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('found_faces', image_2)
    cv2.waitKey(0)
    cv2.imwrite('images/found_faces.jpg', image_2)

    image = cv2.imread('images/logo.jpg', 1)

    cv2.imshow('image', image)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.imwrite('images/new_logo.jpg', image)

    blue, green, red = image[10, 350]
    print(f'RGB-код пикселя: {red}, {green}, {blue}')

    width = image.shape[1]
    for width_coordinate in range(width):
        for frame_coordinate in range(0, 10):
            image[frame_coordinate, width_coordinate] = [0, 0, 0]
            image[-frame_coordinate, width_coordinate] = [0, 0, 0]

    height = image.shape[0]
    for height_coordinate in range(height):
        for frame_coordinate in range(0, 10):
            image[height_coordinate, frame_coordinate] = [0, 0, 0]
            image[height_coordinate, -frame_coordinate] = [0, 0, 0]

    reduced_image = cv2.resize(image, (int(width/2), int(height/2)), cv2.INTER_AREA)
    cv2.imshow('reduced_logo', reduced_image)
    cv2.waitKey(0)
    cv2.imwrite('images/reduced_logo.jpg', reduced_image)

    cropped_image = image[0:368, 200:600]
    cv2.imshow('cropped_logo', cropped_image)
    cv2.waitKey(0)
    cv2.imwrite('images/cropped_logo.jpg', cropped_image)

    matrix = cv2.getRotationMatrix2D((int(width/2), int(height/2)), 60, 0.8)
    rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    cv2.imshow('rotated_logo', rotated_image)
    cv2.waitKey(0)
    cv2.imwrite('images/rotated_logo.jpg', rotated_image)