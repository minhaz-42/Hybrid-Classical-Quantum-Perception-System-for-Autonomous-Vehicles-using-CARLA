#!/usr/bin/env python
"""
Convenience wrapper — delegates to quantumdrive/manage.py so you can run
`python3 manage.py runserver` from the project root directory.
"""
import os
import sys

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantumdrive"))
    sys.path.insert(0, os.getcwd())
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "quantumdrive.settings")
    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)
