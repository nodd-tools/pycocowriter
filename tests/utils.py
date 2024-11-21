import doctest

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