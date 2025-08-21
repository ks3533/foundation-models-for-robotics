from robosuite.models.robots import PandaOmron
from lxml import etree

# Load the robot
robot = PandaOmron()

# Parse the XML string into an lxml Element
robot_xml_str = robot.get_xml()
model_xml = etree.fromstring(robot_xml_str)  # ✅ this is the fix

# Create a new minimal MuJoCo model
world = etree.Element("mujoco", model="panda_clean")
asset = etree.SubElement(world, "asset")
worldbody = etree.SubElement(world, "worldbody")

# Copy robot's assets and body into the clean world
asset.extend(model_xml.find("asset"))
worldbody.append(model_xml.find("worldbody").find("body"))

# Save the new MJCF file
etree.ElementTree(world).write("pandaomron_clean.xml", pretty_print=True)
