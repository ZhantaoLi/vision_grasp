from pathlib import Path
import importlib.util
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_XML = PACKAGE_ROOT / "package.xml"

EXPECTED_RUNTIME_DEPENDENCIES = {
    "rclpy",
    "sensor_msgs",
    "geometry_msgs",
    "visualization_msgs",
    "cv_bridge",
    "tf2_ros",
    "ament_index_python",
    "launch",
    "launch_ros",
    "robot_state_publisher",
    "rviz2",
    "python3-opencv",
    "python3-numpy",
    "python3-serial",
}

EXPECTED_TEST_DEPENDENCIES = {
    "python3-pytest",
    "launch_testing",
    "launch_testing_ros",
}

EXPECTED_DELIVERY_FILES = {
    "README.md",
    "LICENSE",
    "CHANGELOG.rst",
}


def _package_xml_values(tag_name):
    root = ET.parse(PACKAGE_XML).getroot()
    return {element.text for element in root.findall(tag_name)}


def _load_launch_module(filename):
    launch_path = PACKAGE_ROOT / "launch" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node_parameters_by_executable(launch_filename):
    module = _load_launch_module(launch_filename)
    launch_description = module.generate_launch_description()
    params_by_executable = {}
    for entity in launch_description.entities:
        if type(entity).__name__ != "Node":
            continue
        params_by_executable[entity.node_executable] = list(
            entity.__dict__.get("_Node__parameters", [])
        )
    return params_by_executable


def test_package_declares_runtime_dependencies():
    runtime_dependencies = _package_xml_values("depend") | _package_xml_values("exec_depend")
    missing = EXPECTED_RUNTIME_DEPENDENCIES - runtime_dependencies
    assert not missing, f"Missing runtime dependencies in package.xml: {sorted(missing)}"


def test_package_declares_test_dependencies():
    test_dependencies = _package_xml_values("test_depend")
    missing = EXPECTED_TEST_DEPENDENCIES - test_dependencies
    assert not missing, f"Missing test dependencies in package.xml: {sorted(missing)}"


def test_package_contains_basic_delivery_files():
    missing = [name for name in EXPECTED_DELIVERY_FILES if not (PACKAGE_ROOT / name).exists()]
    assert not missing, f"Missing delivery files: {missing}"


def test_pipeline_launch_passes_config_to_tf_transformer_node():
    params_by_executable = _node_parameters_by_executable("pipeline.launch.py")
    assert params_by_executable["tf_transformer_node"], (
        "pipeline.launch.py should pass params.yaml to tf_transformer_node"
    )


def test_demo_launch_passes_config_to_tf_transformer_node():
    params_by_executable = _node_parameters_by_executable("demo.launch.py")
    assert params_by_executable["tf_transformer_node"], (
        "demo.launch.py should pass params.yaml to tf_transformer_node"
    )
