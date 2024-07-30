""" Handle compatibility withing version """
import warnings
import re

from ..exceptions.lib_exceptions import VersionNotCompatibleException

FACTORY_COMPATIBILITIES = [
    ("", "0.4.6"),
    ("0.4.7", "")
]

VERSION_REGEX = re.compile(r'^(\d+\.\d+\.\d+)')

def _get_tuple_version(version : str):
    """
    Extracts the version in x.x.x format from a version string.

    Args:
        version (str): The version string to extract from.

    Returns:
        tuple: The extracted version in x.x.x format or None if invalid.
    """
    print(version)
    match = VERSION_REGEX.match(version)
    print(match)
    if not match:
        raise VersionNotCompatibleException("Version doesn't match x.x.x")

    return tuple(map(int, match.group(1).split('.')))

def are_versions_compatible(current_version, other_version):
    """
    Check if the given current version and other version are compatible based on the FACTORY_COMPATIBILITIES list.

    Args:
        current_version (str): The current version of the library.
        other_version (str): The other version to check compatibility with.

    Returns:
        bool: True if the versions are compatible, False otherwise.
    """
    current_version = _get_tuple_version(current_version)
    other_version = _get_tuple_version(other_version)

    if other_version == (0,0,0):
        warnings.warn(
            f"Found factory saved on version {other_version}. "
            f"Current version is {current_version}. Factory might be broken, either install "
            "correct version or use at your own risk")
        return True

    if current_version == other_version:
        return True

    for lower_bound, upper_bound in FACTORY_COMPATIBILITIES:
        if check_rule(lower_bound, upper_bound, current_version) and check_rule(lower_bound, upper_bound, other_version):
            warnings.warn(
                    f"Found factory saved on version {other_version}. "
                    f"Current version is {current_version}. Factory might be broken, either install "
                    "correct version or use at your own risk")
            return True
    return False

def check_rule(lower_bound, upper_bound, version):
    """
    Helper function to check if the version satisfies the given rule.

    Args:
        lower_bound (str): The lower bound of the rule.
        upper_bound (str): The upper bound of the rule.
        version (tuple): The version to check.

    Returns:
        bool: True if the version satisfies the rule, False otherwise.
    """
    lower_check = True
    upper_check = True

    if lower_bound:
        lower_check = _get_tuple_version(lower_bound) <= version
        print("Lower check : ", lower_check, version, ">", lower_bound)
    if upper_bound:
        upper_check = version <= _get_tuple_version(upper_bound)
        print("Upper check : ", upper_check, version , "=<", upper_bound)
    return lower_check and upper_check
