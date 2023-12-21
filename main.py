from harris_detector import *
from sift_detector import *
from orb_detector import *
from orb_descriptor import *
from comparison import *
from sift_descriptor import *

image1 = 'images/img19.jpg'
image2 = 'images/img20.jpg'

# Harris Detector
harrisDetector(image1)

# SIFT Detector
siftDetector(image1)

# ORB Detector
orbDetector(image1)

# ORB Descriptor
orbDescriptor(image1, image2)

#SIFT Descriptor
siftDescriptor(image1, image2)

# Comparison
comparison(image1)
