import os
from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def get_version():
    """ Find the version of the package"""
    version = None
    version_file = os.path.join(BASEDIR, 'ovos_ww_plugin_openwakeword', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if alpha and int(alpha) > 0:
        version += f"a{alpha}"
    return version


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=').replace('~=', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


PLUGIN_ENTRY_POINT = 'ovos-ww-plugin-openwakeword=ovos_ww_plugin_openwakeword:OwwHotwordPlugin'
setup(
    name='ovos-ww-plugin-openwakeword',
    version=get_version(),
    packages=['ovos_ww_plugin_openwakeword'],
    url='https://github.com/OpenVoiceOS/ovos-ww-plugin-openwakeword',
    license='Apache-2.0',
    author='dscripka',
    install_requires=required("requirements/requirements.txt"),
    author_email='david.scripka@gmail.com',
    description='A wakeword plugin for OpenVoiceOS for the openWakeWord library',
    keywords='mycroft ovos plugin',
    entry_points={'mycroft.plugin.wake_word': PLUGIN_ENTRY_POINT}
)
