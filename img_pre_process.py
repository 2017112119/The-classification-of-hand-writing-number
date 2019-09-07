import cv2 as cv

img1 = cv.imread("1.png")

#灰度
img1_gray = cv.cvtColor(img1,cv.COLOR_RGB2GRAY)
#img1_gray = cv.resize(img1_gray,(32,32))
print(img1)
print(img1_gray)
cv.imshow("1",img1)
cv.imshow("2",img1_gray)

#二值化
binary = cv.adaptiveThreshold(img1_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,25,10)
#cv.imshow("3",binary)
print(binary)

#二值化
ret,binary1 = cv.threshold(img1_gray,0,255,cv.THRESH_BINARY|cv.THRESH_TRIANGLE)
for i in range(200):
    for j in range(188):
        binary1[i][j]=255-binary1[i][j]
print("threshold value %s"%ret)
#cv.imshow("4",binary1)

#平滑滤波
blu_img = cv.blur(binary1,(15,15))
print(blu_img)
cv.imshow("blur",blu_img)

#高斯滤波
gauss_img = cv.GaussianBlur(img1_gray,(5,5),1.5)
cv.imshow("gauss_img",gauss_img)


cv.waitKey(0)

cv.destroyAllWindows()
