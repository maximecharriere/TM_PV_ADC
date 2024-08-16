import json
from xml.dom.minidom import Document
import os

PARENT_DATA_DIR = os.getenv('PARENT_DATA_DIR')
dataDirpath = PARENT_DATA_DIR + r"\PRiOT\dataExport_2"  # "/Applications/Documents/TM Maxime/dataExport_3400_daily"#
metadataFilepath = os.path.join(dataDirpath, "metadata.json")

valid_systems = ['2026250', '2026251', '2026258', '2026269', '2026271', 'a001017', 'a001018', 'a001020', 'a001021', 'a001022', 'a001023', 'a001025', 'a001026', 'a001027', 'a001031', 'a001035', 'a001036', 'a001039', 'a001040', 'a001041', 'a001043', 'a001044', 'a001045', 'a001048', 'a001070', 'a001071', 'a001072', 'a001073', 'a001074', 'a001075', 'a001076', 'a001077', 'a001080', 'a001081', 'a001082', 'a001084', 'a001085', 'a001086', 'a001088', 'a001089', 'a001091', 'a001094', 'a001095', 'a001096', 'a001097', 'a001098', 'a001099', 'a001101', 'a001102', 'a001104', 'a001105', 'a001106', 'a001107', 'a001108', 'a001109', 'a001110', 'a001112', 'a001113', 'a001114', 'a001115', 'a001117', 'a001119', 'a001120', 'a001121', 'a001123', 'a001124', 'a001125', 'a001126', 'a001127', 'a001128', 'a001129', 'a001130', 'a001131', 'a001132', 'a001133', 'a001138', 'a001139', 'a001140', 'a001141', 'a001142', 'a001144', 'a001146', 'a001147', 'a001148', 'a001149', 'a001152', 'a001153', 'a001155', 'a001156', 'a001157', 'a001158', 'a001159', 'a001160', 'a001161', 'a001162', 'a001163', 'a001166', 'a001168', 'a001179', 'a001180', 'a001182', 'a001184', 'a001185', 'a001186', 'a001188', 'a001190', 'a001192', 'a001193', 'a001194', 'a001195', 'a001196', 'a001197', 'a001200', 'a001202', 'a001204', 'a001205', 'a001206', 'a001207', 'a001208', 'a001209', 'a001210', 'a001214', 'a001215', 'a001216', 'a001217', 'a001218', 'a001219', 'a001221', 'a001224', 'a001225', 'a001228', 'a001229', 'a001230', 'a001231', 'a001233', 'a001234', 'a001235', 'a001236', 'a001238', 'a001239', 'a001240', 'a001241', 'a001242', 'a001243', 'a001244', 'a001246', 'a001247', 'a001248', 'a001250', 'a001252', 'a001255', 'a001256', 'a001260', 'a001262', 'a001263', 'a001264', 'a001265', 'a001266', 'a001267', 'a001268', 'a001270', 'a001271', 'a001272', 'a001273', 'a001274', 'a001275', 'a001276', 'a001277', 'a001278', 'a001279', 'a001282', 'a001284', 'a001286', 'a001287', 'a001289', 'a001290', 'a001291', 'a001292', 'a001293', 'a001294', 'a001295', 'a001296', 'a001297', 'a001298', 'a001299', 'a001300', 'a001301', 'a001302', 'a001303', 'a001304', 'a001305', 'a001306', 'a001308', 'a001309', 'a001310', 'a001311', 'a001312', 'a001313', 'a001314', 'a001315', 'a001316', 'a001317', 'a001318', 'a001319', 'a001320', 'a001321', 'a001322', 'a001323', 'a001325', 'a001326', 'a001327', 'a001328', 'a001329', 'a001330', 'a001331', 'a001333', 'a001334', 'a001336', 'a001337', 'a001339', 'a001345', 'a001349', 'a001355', 'a001358', 'a001360', 'a001364', 'a001367', 'a001369', 'a001374', 'a001375', 'a001376', 'a001377', 'a001378', 'a001379', 'a001381', 'a001383', 'a001384', 'a001385', 'a001386', 'a001387', 'a001388', 'a001391', 'a001394', 'a001396', 'a001397', 'a001398', 'a001399', 'a001400', 'a001401', 'a001403', 'a001405', 'a001406', 'a001409', 'a001410', 'a001414', 'a001416', 'a001417', 'a001418', 'a001420', 'a001424', 'a001428', 'a001429', 'a001430', 'a001431', 'a001436', 'a001438', 'a001439', 'a001440', 'a001441', 'a001443', 'a001444', 'a001445', 'a001448', 'a001449', 'a001452', 'a001453', 'a001455', 'a001457', 'a001458', 'a001461', 'a001463', 'a001464', 'a001465', 'a001467', 'a001468', 'a001473', 'a001474', 'a001475', 'a001479', 'a001481', 'a001482', 'a001483', 'a001484', 'a001485', 'a001487', 'a001490', 'a001491', 'a001492', 'a001493', 'a001494', 'a001496', 'a001497', 'a001498', 'a001506', 'a001510', 'a001515', 'a001516', 'a001527', 'a001535', 'a001559', 'a001591', 'a001600', 'a001623', 'a001624', 'a001625', 'g001002', 'g001003', 'g001004', 'g001005', 'g001006']


# Load the JSON data
with open(metadataFilepath, 'r') as file:
    data = json.load(file)

# Create a new KML document
doc = Document()

# Create the <kml> base element
kml = doc.createElement('kml')
kml.setAttribute('xmlns', 'http://www.opengis.net/kml/2.2')
doc.appendChild(kml)

# Create a <Document> element under <kml>
document = doc.createElement('Document')
kml.appendChild(document)

# Create a <Style> element for the red circle icon
style = doc.createElement('Style')
style.setAttribute('id', 'redCircleIcon')

icon_style = doc.createElement('IconStyle')
icon = doc.createElement('Icon')
href = doc.createElement('href')
href.appendChild(doc.createTextNode('http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'))
icon.appendChild(href)
icon_style.appendChild(icon)

color = doc.createElement('color')
color.appendChild(doc.createTextNode('ff0000ff'))  # Red color in ABGR format
icon_style.appendChild(color)

scale = doc.createElement('scale')
scale.appendChild(doc.createTextNode('1.2'))  # Scale of the icon
icon_style.appendChild(scale)

style.appendChild(icon_style)
document.appendChild(style)

# Iterate through the JSON data and create a placemark for each location
for key, value in data.items():
    if key in valid_systems:
        # Create a <Placemark> element
        placemark = doc.createElement('Placemark')

        # Assign the style to the placemark
        style_url = doc.createElement('styleUrl')
        style_url.appendChild(doc.createTextNode('#redCircleIcon'))
        placemark.appendChild(style_url)

        # Create a <Point> element
        point = doc.createElement('Point')
        placemark.appendChild(point)

        # Create a <coordinates> element and set its value to the longitude, latitude, and altitude
        coordinates = doc.createElement('coordinates')
        coords = f"{value['metadata']['loc_longitude']},{value['metadata']['loc_latitude']},{value['metadata']['loc_altitude']}"
        coordinates.appendChild(doc.createTextNode(coords))
        point.appendChild(coordinates)

        # Append the placemark to the document
        document.appendChild(placemark)

# Write the KML document to a file
with open('locations.kml', 'w') as kml_file:
    kml_file.write(doc.toprettyxml(indent="  "))

print("KML file with red circle icons created successfully!")
