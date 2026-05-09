"""Add inertial data from YAML to panda URDF."""
import xml.etree.ElementTree as ET
import yaml
import os

YAML_PATH = "/tmp/franka_ros/franka_description/robots/common/inertial.yaml"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "panda.urdf")

# Mapping from YAML keys to URDF link names
NAME_MAP = {
    "link0": "panda_link0",
    "link1": "panda_link1",
    "link2": "panda_link2",
    "link3": "panda_link3",
    "link4": "panda_link4",
    "link5": "panda_link5",
    "link6": "panda_link6",
    "link7": "panda_link7",
    "hand": "panda_hand",
    "leftfinger": "panda_leftfinger",
    "rightfinger": "panda_rightfinger",
}

# Read YAML
with open(YAML_PATH, "r") as f:
    inertial_data = yaml.safe_load(f)

# Read URDF
tree = ET.parse(URDF_PATH)
root = tree.getroot()

# Build lookup: urdf_link_name -> yaml entry
urdf_to_yaml = {}
for yaml_key, urdf_name in NAME_MAP.items():
    if yaml_key in inertial_data:
        urdf_to_yaml[urdf_name] = inertial_data[yaml_key]

# Process each link element
for link_elem in root.findall("link"):
    link_name = link_elem.get("name")
    if link_name not in urdf_to_yaml:
        continue

    data = urdf_to_yaml[link_name]

    # Remove existing inertial if any
    existing = link_elem.find("inertial")
    if existing is not None:
        link_elem.remove(existing)

    # Build <inertial> block
    inertial_elem = ET.SubElement(link_elem, "inertial")

    # <origin xyz="..." rpy="..." />
    origin = data["origin"]
    xyz_str = f"{float(origin['xyz'].split()[0])} {float(origin['xyz'].split()[1])} {float(origin['xyz'].split()[2])}" if isinstance(origin["xyz"], str) else f"{origin['xyz']}"
    rpy_str = f"{float(origin['rpy'].split()[0])} {float(origin['rpy'].split()[1])} {float(origin['rpy'].split()[2])}" if isinstance(origin["rpy"], str) else f"{origin['rpy']}"
    ET.SubElement(inertial_elem, "origin", xyz=xyz_str, rpy=rpy_str)

    # <mass value="..." />
    ET.SubElement(inertial_elem, "mass", value=str(float(data["mass"])))

    # <inertia ixx="..." iyy="..." izz="..." ixy="..." ixz="..." iyz="..." />
    inertia = data["inertia"]
    ET.SubElement(inertial_elem, "inertia",
                  ixx=str(float(inertia["xx"])),
                  iyy=str(float(inertia["yy"])),
                  izz=str(float(inertia["zz"])),
                  ixy=str(float(inertia["xy"])),
                  ixz=str(float(inertia["xz"])),
                  iyz=str(float(inertia["yz"])))

# Indent the XML for readability (Python 3.9+)
ET.indent(tree, space="    ")

# Write back
tree.write(URDF_PATH, xml_declaration=True, encoding="unicode")

print(f"Inertial data added to {URDF_PATH}")
print(f"Links updated: {sorted(urdf_to_yaml.keys())}")
