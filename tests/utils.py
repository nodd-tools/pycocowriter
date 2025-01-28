import doctest
import difflib
import json

def compare_dicts(d1, d2):
    d = difflib.Differ()
    return list(d.compare(
        json.dumps(d1, indent=4, sort_keys=True).splitlines(),
        json.dumps(d2, indent=4, sort_keys=True).splitlines()
    ))

def doctests(module, tests):
    """
    A helper function to combine unittest discovery with doctest suites.

    Args:
        module: The module to add doctests from.
        tests: The existing TestSuite discovered by unittest.

    Returns:
        A combined TestSuite with doctests added.
    """
    tests.addTests(doctest.DocTestSuite(module))
    return tests