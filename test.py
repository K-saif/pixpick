import cv2
import pixpick

# select ROI interactively
region = pixpick.box("1.jpeg")

print("\n=== RAW ===")
print(region.to_raw())

print("\n=== XYXY ===")
print(region.xyxy)

print("\n=== XYWH ===")
print(region.xywh)

print("\n=== NORMALIZED ===")
print(region.normalized)

print("\n=== YOLO ===")
print(region.yolo_region())

print("\n=== CENTER ===")
print(region.center)

print("\n=== AREA ===")
print(region.area)

# save
region.save("roi.json")

# load
loaded = pixpick.load("roi.json")

print("\n=== LOADED ===")
print(loaded)

# visualize
img = cv2.imread("1.jpeg")
vis = loaded.visualize(img)

cv2.imshow("box Selection", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()








print("\n============== POLYGON ==============")

# select ROI interactively
region = pixpick.polygon("1.jpeg")

print("\n=== RAW ===")
print(region.to_raw())

print("\n=== NUMPY ===")
print(region.as_numpy)

print("\n=== NORMALIZED NUMPY ===")
print(region.normalized_numpy)

print("\n=== NORMALIZED ===")
print(region.normalized)

print("\n=== BOUNDING BOX ===")
print(region.bounding_box)

print("\n=== NUMBER OF POINTS ===")
print(region.n_points)

print("\n=== SUPERVISION ===")
print(region.to_supervision())

# save
region.save("roi-poly.json")

# load
loaded = pixpick.load("roi-poly.json")

print("\n=== LOADED ===")
print(loaded)

# visualize
img = cv2.imread("1.jpeg")
vis = loaded.visualize(img)

cv2.imshow("polygon Selection", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()










