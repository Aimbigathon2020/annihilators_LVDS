import no_plate as p
import type as t
import cv2
img=cv2.imread('DVLA-number-plates-2017-67-new-car-847566.jpg')
t.typevehicle(img)
p.noplateno(img)
cv2.destroyAllWindows()