# The following lines are replaced during the release process.
# START_VERSION_BLOCK
VERSION_MAJOR = 0
VERSION_MINOR = 4
VERSION_BUILD = 2
VERSION_ALPHA = 0
# END_VERSION_BLOCK

__version__ = "{}.{}.{}{}".format(VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD,
                                   "a{}".format(VERSION_ALPHA) if VERSION_ALPHA else "")

if __name__ == "__main__":
    print(__version__)
