import xml.etree.ElementTree as ET


def convert_to_yolo_format(xml_file, output_file, class_id=-1):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Open the output file for writing all frames' annotations
    with open(output_file, "w") as f:
        # Loop through each frame in the XML file
        for frame in root.findall('frame'):
            frame_num = frame.get('num')

            # Loop through all targets in the frame
            for target in frame.findall('.//target'):
                # Get the bounding box coordinates
                box = target.find('box')
                left = float(box.get('left'))
                top = float(box.get('top'))
                width = float(box.get('width'))
                height = float(box.get('height'))


                # Write the YOLO format annotation with frame_num, class_id, and normalized values
                f.write(f"{frame_num},{class_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},-1,-1,-1\n")


# Example usage
video_index='MVI_40773'
xml_file = 'F:/0_Joy_Data/1_Work/Projects/PyCharmProjects/1_dataset/Video/UA-DETRAC/Test-XML/DETRAC-Test-Annotations-XML/'+video_index+'.xml'
output_file = '../labels/'+video_index+'.txt'
convert_to_yolo_format(xml_file, output_file)

