import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

PROCEDURES = {}


def add_procedure(name, procedure):
    """Adds a procedure"""
    PROCEDURES[name] = procedure


def iterative_importer(path):
    """
    Imports every python script from a given path
    Ignores hidden files, files that begin with "." and "_" are considered hidden

    Path is relative to the working directory
    """
    # Returns True if the file is hidden

    check_hidden = lambda x: x.startswith('_') or x.startswith('.')
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not check_hidden(d)]

        for script in files:
            if not check_hidden(script) and script.endswith('.py'):
                spec = importlib.util.spec_from_file_location(
                    script.rstrip('.py'), os.path.join(root, script)
                )
                spec.loader.exec_module(importlib.util.module_from_spec(spec))


def initialise():
    """Run main initialisation steps, returns an app context"""
    iterative_importer("script_folder")
    return PROCEDURES
