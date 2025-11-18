import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _dict_to_xml(tag: str, data: dict[str, Any]) -> ET.Element:
    """Convert a dictionary to an XML Element recursively."""
    element = ET.Element(tag)

    for key, value in data.items():
        child = ET.SubElement(element, str(key))
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                sub_child = ET.SubElement(child, str(sub_key))
                sub_child.text = str(sub_value)
        elif isinstance(value, list):
            for item in value:
                item_element = ET.SubElement(child, 'item')
                if isinstance(item, dict):
                    for sub_key, sub_value in item.items():
                        sub_child = ET.SubElement(item_element, str(sub_key))
                        sub_child.text = str(sub_value)
                else:
                    item_element.text = str(item)
        else:
            child.text = str(value)

    return element


def _xml_to_dict(element: ET.Element) -> dict[str, Any] | list[Any] | str:
    """Convert an XML Element to a dictionary recursively."""
    # If element has no children, return its text
    if len(element) == 0:
        return element.text or ''

    result: dict[str, Any] = {}
    for child in element:
        child_data = _xml_to_dict(child)

        # Handle list items
        if child.tag == 'item':
            if element.tag not in result:
                result[element.tag] = []
            if isinstance(result[element.tag], list):
                result[element.tag].append(child_data)
        # Handle regular elements
        elif child.tag in result:
            # Convert to list if duplicate tags
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data

    return result


def load_xml(path: str | Path) -> dict[str, Any]:
    """Load XML data from file and convert to dictionary.

    Args:
        path: Path to the XML file

    Returns:
        Dictionary representation of the XML data

    Note:
        This uses xml.etree.ElementTree which is not secure against maliciously
        constructed data. For untrusted data, consider using defusedxml.

    """
    tree = ET.parse(str(path))  # noqa: S314
    root = tree.getroot()
    result = _xml_to_dict(root)

    # Wrap in root tag name if result is not already wrapped
    if isinstance(result, dict) and root.tag not in result:
        return {root.tag: result}
    return result if isinstance(result, dict) else {root.tag: result}


def save_as_xml(
    data: dict[str, Any],
    path: str | Path,
    root_tag: str = 'root',
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    """Save dictionary data as XML to file.

    Args:
        data: Dictionary data to save
        path: Path where the XML file should be saved
        root_tag: Tag name for the root element (default: 'root')
        parents: If True, create parent directories as needed
        exist_ok: If True, don't raise error if directory exists

    """
    target = Path(path)
    target.parent.mkdir(parents=parents, exist_ok=exist_ok)

    # If data has single key, use it as root tag
    if len(data) == 1:
        root_tag = next(iter(data.keys()))
        root_data = data[root_tag]
        if isinstance(root_data, dict):
            root = _dict_to_xml(root_tag, root_data)
        else:
            root = ET.Element(root_tag)
            root.text = str(root_data)
    else:
        root = _dict_to_xml(root_tag, data)

    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(str(target), encoding='utf-8', xml_declaration=True)


class XmlFileHandler:
    """XML file handler implementing FileHandler protocol."""

    def __init__(self, root_tag: str = 'root') -> None:
        """Initialize XML handler.

        Args:
            root_tag: Default root tag name for saving (default: 'root')

        """
        self.root_tag = root_tag

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load XML data from file."""
        return load_xml(path)

    def save(
        self,
        data: dict[str, Any],
        path: str | Path,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Save data as XML to file."""
        save_as_xml(data, path, root_tag=self.root_tag, parents=parents, exist_ok=exist_ok)
