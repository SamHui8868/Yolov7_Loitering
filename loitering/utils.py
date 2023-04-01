import json
import cv2
import numpy as np


# Assume 1 annotation file contains only 1 alram zone
def alarmzone_anno_reader(json_path,alarmzone_id=1):
    import json

    # Open the file for reading
    with open(json_path, 'r') as f:
        # Load the contents of the file as a JSON object
        json_data = json.load(f)
    # Get category ID of alarm zone
    category_id = alarmzone_id
    # Find annotation with relevant category ID
    annotation = None
    for ann in json_data['annotations']:
        if ann['category_id'] == category_id:
            annotation = ann
            break
    # Extract polygon vertices from segmentation field
    polygon_vertices = annotation['segmentation'][0]
    # Reshape polygon vertices into a numpy array
    polygon_vertices = np.array(polygon_vertices, np.int32).reshape((-1, 2))
    return polygon_vertices

# Draw Bounding Box and Label with OpenCV
def bounding_box_drawer(img,class_names,class_id, confidence, x, y, x_plus_w, y_plus_h,in_zone,color=(255,0,255)):

    label = str(class_names[int(class_id)])

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label+' '+str(int(confidence*100))+' '+str(in_zone), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Draw Bounding Box of Tracking and Label with OpenCV
def bounding_box_drawer_track(img,track_id,x, y, x_plus_w, y_plus_h,in_zone,color=(255,0,255)):
    if(in_zone):
        state='IN'
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,0,255), 2)
        cv2.putText(img, 'Target '+state, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    else:
        state='OUT'
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (255,0,255), 2)
        cv2.putText(img, 'Target '+state, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        
# Identify point in polygon by OpenCV     
def point_polygon_tester(point,polygon_vertices):
    # Check if point is inside polygon
    result = cv2.pointPolygonTest(polygon_vertices, tuple(point), False)
    if result >= 0:
         # Points within or Lie on polygon edge will return true
        return 1
    else:
         # Points outside polygon will return false
        return 0
        